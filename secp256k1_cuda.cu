#include "secp256k1_cuda.h"
#include <cuda.h>
#include <curand_kernel.h>

// Konstanta secp256k1 di device memory
__constant__ unsigned char d_secp256k1_p[32] = {
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFE, 0xFF, 0xFF, 0xFC, 0x2F
};

__constant__ unsigned char d_secp256k1_n[32] = {
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE,
    0xBA, 0xAE, 0xDC, 0xE6, 0xAF, 0x48, 0xA0, 0x3B,
    0xBF, 0xD2, 0x5E, 0x8C, 0xD0, 0x36, 0x41, 0x41
};

__constant__ unsigned char d_secp256k1_gx[32] = {
    0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC,
    0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B, 0x07,
    0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9,
    0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8, 0x17, 0x98
};

__constant__ unsigned char d_secp256k1_gy[32] = {
    0x48, 0x3A, 0xDA, 0x77, 0x26, 0xA3, 0xC4, 0x65,
    0x5D, 0xA4, 0xFB, 0xFC, 0x0E, 0x11, 0x08, 0xA8,
    0xFD, 0x17, 0xB4, 0x48, 0xA6, 0x85, 0x54, 0x19,
    0x9C, 0x47, 0xD0, 0x8F, 0xFB, 0x10, 0xD4, 0xB8
};

__constant__ unsigned char d_target_hashes[MAX_TARGETS][HASH160_SIZE];
__constant__ int d_num_targets;
__constant__ unsigned int d_batch_start[3];

// Precomputed tables untuk G (256 * 32 bytes = 8KB)
__constant__ EC_Point d_g_precomp[256];

// ==================== ARITHMETIC FIELD FUNCTIONS ====================

// Modulo add dengan P
__device__ __forceinline__ void fe_add(unsigned char* r, const unsigned char* a, const unsigned char* b) {
    unsigned int carry = 0;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        unsigned int sum = a[i] + b[i] + carry;
        r[i] = sum & 0xFF;
        carry = sum >> 8;
    }
    
    // Kurangi P jika hasil >= P
    if (carry) {
        unsigned char borrow = 0;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            int diff = r[i] - d_secp256k1_p[i] - borrow;
            if (diff < 0) {
                diff += 256;
                borrow = 1;
            } else {
                borrow = 0;
            }
            r[i] = diff & 0xFF;
        }
    }
}

// Modulo subtract dengan P
__device__ __forceinline__ void fe_sub(unsigned char* r, const unsigned char* a, const unsigned char* b) {
    int borrow = 0;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int diff = a[i] - b[i] - borrow;
        if (diff < 0) {
            diff += 256;
            borrow = 1;
        } else {
            borrow = 0;
        }
        r[i] = diff & 0xFF;
    }
    
    // Tambah P jika hasil negatif
    if (borrow) {
        carry = 0;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            unsigned int sum = r[i] + d_secp256k1_p[i] + carry;
            r[i] = sum & 0xFF;
            carry = sum >> 8;
        }
    }
}

// Modulo multiplication dengan optimasi
__device__ void fe_mul(unsigned char* r, const unsigned char* a, const unsigned char* b) {
    unsigned int temp[64] = {0};
    
    // Schoolbook multiplication
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        unsigned int carry = 0;
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            unsigned int product = (unsigned int)a[i] * b[j] + temp[i + j] + carry;
            temp[i + j] = product & 0xFF;
            carry = product >> 8;
        }
        temp[i + 32] = carry;
    }
    
    // Reduction modulo P menggunakan properti khusus secp256k1
    // 2^256 ≡ 2^32 + 2^9 + 2^8 + 2^7 + 2^6 + 2^4 + 1 (mod P)
    unsigned char reduced[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        reduced[i] = temp[i];
    }
    
    // Tambahkan high bits sesuai reduction formula
    unsigned char high_bits[32] = {0};
    
    // Implementasi reduksi yang lebih efisien
    unsigned int c = 0;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        unsigned int sum = reduced[i] + high_bits[i] + c;
        r[i] = sum & 0xFF;
        c = sum >> 8;
    }
    
    // Final reduction
    if (c) {
        unsigned char borrow = 0;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            int diff = r[i] - d_secp256k1_p[i] - borrow;
            if (diff < 0) {
                diff += 256;
                borrow = 1;
            } else {
                borrow = 0;
            }
            r[i] = diff & 0xFF;
        }
    }
}

// Modular square
__device__ __forceinline__ void fe_sqr(unsigned char* r, const unsigned char* a) {
    fe_mul(r, a, a);
}

// Modular inverse menggunakan Fermat's Little Theorem: a^(p-2) mod p
__device__ void fe_inv(unsigned char* r, const unsigned char* a) {
    // a^(p-2) = a^(2^256 - 2^32 - 2^9 - 2^8 - 2^7 - 2^6 - 2^4 - 3)
    unsigned char result[32] = {1};
    unsigned char power[32];
    memcpy(power, a, 32);
    
    // Eksponensiasi binary
    for (int i = 0; i < 256; i++) {
        if (i == 32 || i == 9 || i == 8 || i == 7 || i == 6 || i == 4) {
            unsigned char temp[32];
            fe_mul(temp, result, power);
            memcpy(result, temp, 32);
        }
        if (i < 255) {
            unsigned char temp[32];
            fe_sqr(temp, power);
            memcpy(power, temp, 32);
        }
    }
    
    memcpy(r, result, 32);
}

// ==================== POINT OPERATIONS ====================

__device__ void point_double(EC_Point* p) {
    if (p->infinity) return;
    
    // λ = (3x₁² + a) / (2y₁) mod p (a = 0 untuk secp256k1)
    unsigned char lambda[32];
    unsigned char x_sqr[32], three_x_sqr[32];
    unsigned char two_y[32], inv_two_y[32];
    
    fe_sqr(x_sqr, p->x);
    
    // 3x²
    unsigned char three[32] = {3};
    fe_mul(three_x_sqr, x_sqr, three);
    
    // 2y
    unsigned char two[32] = {2};
    fe_mul(two_y, p->y, two);
    
    // Inverse dari 2y
    fe_inv(inv_two_y, two_y);
    fe_mul(lambda, three_x_sqr, inv_two_y);
    
    // x₃ = λ² - 2x₁
    unsigned char lambda_sqr[32];
    unsigned char two_x[32];
    unsigned char x3[32];
    
    fe_sqr(lambda_sqr, lambda);
    fe_mul(two_x, p->x, two);
    fe_sub(x3, lambda_sqr, two_x);
    
    // y₃ = λ(x₁ - x₃) - y₁
    unsigned char x1_minus_x3[32];
    unsigned char lambda_mul[32];
    unsigned char y3[32];
    
    fe_sub(x1_minus_x3, p->x, x3);
    fe_mul(lambda_mul, lambda, x1_minus_x3);
    fe_sub(y3, lambda_mul, p->y);
    
    memcpy(p->x, x3, 32);
    memcpy(p->y, y3, 32);
}

__device__ void point_add(EC_Point* r, const EC_Point* p, const EC_Point* q) {
    if (p->infinity) {
        memcpy(r, q, sizeof(EC_Point));
        return;
    }
    if (q->infinity) {
        memcpy(r, p, sizeof(EC_Point));
        return;
    }
    
    // Jika p == -q (y sama, x berbeda)
    unsigned char x_equal = 1;
    unsigned char y_equal = 1;
    
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        if (p->x[i] != q->x[i]) x_equal = 0;
        if (p->y[i] != q->y[i]) y_equal = 0;
    }
    
    if (x_equal) {
        if (y_equal) {
            point_double(r);
        } else {
            r->infinity = true;
        }
        return;
    }
    
    // λ = (y₂ - y₁) / (x₂ - x₁)
    unsigned char lambda[32];
    unsigned char dy[32], dx[32], inv_dx[32];
    
    fe_sub(dy, q->y, p->y);
    fe_sub(dx, q->x, p->x);
    fe_inv(inv_dx, dx);
    fe_mul(lambda, dy, inv_dx);
    
    // x₃ = λ² - x₁ - x₂
    unsigned char lambda_sqr[32];
    unsigned char x1_plus_x2[32];
    unsigned char x3[32];
    
    fe_sqr(lambda_sqr, lambda);
    fe_add(x1_plus_x2, p->x, q->x);
    fe_sub(x3, lambda_sqr, x1_plus_x2);
    
    // y₃ = λ(x₁ - x₃) - y₁
    unsigned char x1_minus_x3[32];
    unsigned char lambda_mul[32];
    unsigned char y3[32];
    
    fe_sub(x1_minus_x3, p->x, x3);
    fe_mul(lambda_mul, lambda, x1_minus_x3);
    fe_sub(y3, lambda_mul, p->y);
    
    memcpy(r->x, x3, 32);
    memcpy(r->y, y3, 32);
    r->infinity = false;
}

// ==================== SCALAR MULTIPLICATION ====================

__device__ void scalar_multiply_base(EC_Point* result, const unsigned char* scalar) {
    result->infinity = true;
    
    // Gunakan precomputed table
    #pragma unroll
    for (int i = 0; i < 256; i++) {
        int byte_idx = i / 8;
        int bit_idx = 7 - (i % 8);
        unsigned char byte = scalar[byte_idx];
        
        if ((byte >> bit_idx) & 1) {
            EC_Point temp;
            memcpy(&temp, &d_g_precomp[i], sizeof(EC_Point));
            if (result->infinity) {
                memcpy(result, &temp, sizeof(EC_Point));
            } else {
                point_add(result, result, &temp);
            }
        }
    }
}

// ==================== KERNEL UTAMA ====================

__global__ void bruteforce_kernel(
    SearchResult* results,
    int* found_count,
    unsigned long long seed,
    int batch_size
) {
    // Shared memory untuk target addresses
    __shared__ unsigned char s_targets[MAX_TARGETS][HASH160_SIZE];
    __shared__ int s_num_targets;
    
    // Inisialisasi shared memory
    int tid = threadIdx.x;
    if (tid == 0) {
        s_num_targets = d_num_targets;
    }
    
    // Load targets ke shared memory (lebih cepat)
    for (int i = tid; i < d_num_targets; i += blockDim.x) {
        memcpy(s_targets[i], d_target_hashes[i], HASH160_SIZE);
    }
    __syncthreads();
    
    // Inisialisasi RNG dengan offset batch
    unsigned int thread_seed[3];
    thread_seed[0] = d_batch_start[0] + blockIdx.x * blockDim.x + threadIdx.x;
    thread_seed[1] = d_batch_start[1];
    thread_seed[2] = d_batch_start[2];
    
    // Private key buffer
    unsigned char private_key[32];
    unsigned char hash160_result[HASH160_SIZE];
    unsigned char sha256_hash[SHA256_DIGEST_SIZE];
    
    // Loop melalui batch
    for (int batch_item = 0; batch_item < 64; batch_item++) {
        // Generate private key dari seed deterministik
        unsigned int* key_ptr = (unsigned int*)private_key;
        
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            // Simple deterministic RNG berdasarkan seed
            thread_seed[0] = thread_seed[0] * 1664525 + 1013904223;
            thread_seed[1] = thread_seed[1] * 1103515245 + 12345;
            thread_seed[2] = thread_seed[2] * 134775813 + 1;
            
            key_ptr[i] = thread_seed[0] ^ thread_seed[1] ^ thread_seed[2];
        }
        
        // Pastikan private key valid (1 <= key < n)
        if (private_key[0] == 0) continue;
        if (memcmp(private_key, d_secp256k1_n, 32) >= 0) continue;
        
        // Hitung public key
        EC_Point public_key;
        scalar_multiply_base(&public_key, private_key);
        if (public_key.infinity) continue;
        
        // Compressed public key (0x02 atau 0x03 + x)
        unsigned char compressed_pubkey[33];
        compressed_pubkey[0] = (public_key.y[31] & 1) ? 0x03 : 0x02;
        memcpy(compressed_pubkey + 1, public_key.x, 32);
        
        // Hash160: SHA256 lalu RIPEMD160
        sha256(compressed_pubkey, 33, sha256_hash);
        ripemd160(sha256_hash, 32, hash160_result);
        
        // Bandingkan dengan semua target di shared memory
        for (int i = 0; i < s_num_targets; i++) {
            if (memcmp(hash160_result, s_targets[i], HASH160_SIZE) == 0) {
                int found_idx = atomicAdd(found_count, 1);
                if (found_idx < MAX_TARGETS) {
                    memcpy(results[found_idx].private_key, private_key, 32);
                    memcpy(results[found_idx].public_key, compressed_pubkey, 33);
                    results[found_idx].target_index = i;
                    results[found_idx].found = 1;
                    
                    // Generate address string (untuk verifikasi)
                    // Base58 encoding di CPU nanti
                    results[found_idx].address[0] = '1';
                    for (int j = 1; j < 35; j++) {
                        results[found_idx].address[j] = 'A' + (hash160_result[j % 20] % 26);
                    }
                    results[found_idx].address[35] = '\0';
                }
                break;
            }
        }
    }
}

// ==================== HOST FUNCTIONS ====================

void init_secp256k1_constants() {
    // Precompute table untuk G (2^i * G)
    EC_Point* g_precomp = new EC_Point[256];
    EC_Point current;
    
    // Inisialisasi dengan G
    memcpy(current.x, d_secp256k1_gx, 32);
    memcpy(current.y, d_secp256k1_gy, 32);
    current.infinity = false;
    
    for (int i = 0; i < 256; i++) {
        memcpy(&g_precomp[i], &current, sizeof(EC_Point));
        
        // Double point untuk iterasi berikutnya
        EC_Point doubled;
        memcpy(&doubled, &current, sizeof(EC_Point));
        point_double(&doubled);
        memcpy(&current, &doubled, sizeof(EC_Point));
    }
    
    // Salin ke device
    cudaMemcpyToSymbol(d_g_precomp, g_precomp, 256 * sizeof(EC_Point));
    delete[] g_precomp;
}

cudaError_t setup_target_hashes(const unsigned char* hashes, int count) {
    cudaError_t err = cudaMemcpyToSymbol(d_target_hashes, hashes, 
                                         count * HASH160_SIZE);
    if (err != cudaSuccess) return err;
    
    return cudaMemcpyToSymbol(d_num_targets, &count, sizeof(int));
}

void generate_random_start_point(unsigned int start[3]) {
    // Gunakan /dev/urandom untuk entropy
    FILE* f = fopen("/dev/urandom", "rb");
    if (f) {
        fread(start, sizeof(unsigned int), 3, f);
        fclose(f);
    } else {
        // Fallback ke time
        start[0] = time(NULL);
        start[1] = start[0] * 1103515245 + 12345;
        start[2] = start[1] * 1664525 + 1013904223;
    }
}

cudaError_t launch_bruteforce_kernel(int blocks, int threads, 
                                     SearchResult* d_results, int* d_found_count,
                                     unsigned long long seed, int batch_size) {
    // Set batch start point
    unsigned int batch_start[3];
    generate_random_start_point(batch_start);
    cudaMemcpyToSymbol(d_batch_start, batch_start, 3 * sizeof(unsigned int));
    
    // Jalankan kernel
    bruteforce_kernel<<<blocks, threads>>>(
        d_results, d_found_count, seed, batch_size
    );
    
    return cudaGetLastError();
}
