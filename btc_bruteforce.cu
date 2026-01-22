#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <openssl/sha.h>
#include <openssl/ripemd.h>

using namespace std;
using namespace std::chrono;

// ==================== KONFIGURASI ====================
#define PRIVATE_KEY_SIZE 32
#define PUBLIC_KEY_SIZE 65
#define COMPRESSED_PUBLIC_KEY_SIZE 33
#define HASH160_SIZE 20
#define ADDRESS_SIZE 25
#define SHA256_DIGEST_SIZE 32
#define RIPEMD160_DIGEST_SIZE 20
#define MAX_TARGETS 10000
#define THREADS_PER_BLOCK 256
#define BATCH_SIZE 64

// Struktur untuk hasil
typedef struct {
    unsigned char private_key[32];
    unsigned char public_key[33];
    char address[36];
    int target_index;
    int found;
} SearchResult;

// ==================== KONSTANTA secp256k1 ====================
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

// Target addresses dalam bentuk hash160
__constant__ unsigned char d_target_hashes[MAX_TARGETS][HASH160_SIZE];
__constant__ int d_num_targets;

// Konstanta SHA256
__constant__ unsigned int d_sha256_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// ==================== FUNGSI UTILITAS GPU ====================
__device__ __forceinline__ unsigned int rotr32(unsigned int x, int n) {
    return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ unsigned int ch(unsigned int x, unsigned int y, unsigned int z) {
    return (x & y) ^ (~x & z);
}

__device__ __forceinline__ unsigned int maj(unsigned int x, unsigned int y, unsigned int z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __forceinline__ unsigned int sigma0(unsigned int x) {
    return rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22);
}

__device__ __forceinline__ unsigned int sigma1(unsigned int x) {
    return rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25);
}

__device__ __forceinline__ unsigned int gamma0(unsigned int x) {
    return rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3);
}

__device__ __forceinline__ unsigned int gamma1(unsigned int x) {
    return rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10);
}

// ==================== SHA256 di GPU ====================
__device__ void sha256_transform(unsigned int* state, const unsigned char* data) {
    unsigned int w[64];
    
    // Load data
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        w[i] = (data[i*4] << 24) | (data[i*4+1] << 16) | 
               (data[i*4+2] << 8) | data[i*4+3];
    }
    
    // Expand message
    #pragma unroll
    for (int i = 16; i < 64; i++) {
        w[i] = gamma1(w[i-2]) + w[i-7] + gamma0(w[i-15]) + w[i-16];
    }
    
    unsigned int a = state[0];
    unsigned int b = state[1];
    unsigned int c = state[2];
    unsigned int d = state[3];
    unsigned int e = state[4];
    unsigned int f = state[5];
    unsigned int g = state[6];
    unsigned int h = state[7];
    
    // Compression
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        unsigned int t1 = h + sigma1(e) + ch(e, f, g) + d_sha256_k[i] + w[i];
        unsigned int t2 = sigma0(a) + maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

__device__ void sha256_gpu(const unsigned char* data, int len, unsigned char* output) {
    unsigned int state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    
    // Process full chunks
    int i;
    for (i = 0; i + 64 <= len; i += 64) {
        sha256_transform(state, data + i);
    }
    
    // Process final chunk
    unsigned char final[64] = {0};
    int rem = len - i;
    
    if (rem > 0) {
        memcpy(final, data + i, rem);
    }
    
    final[rem] = 0x80;
    
    if (rem < 56) {
        unsigned long long bit_len = (unsigned long long)len * 8;
        for (int j = 0; j < 8; j++) {
            final[56 + j] = (bit_len >> (56 - j * 8)) & 0xFF;
        }
        sha256_transform(state, final);
    } else {
        unsigned long long bit_len = (unsigned long long)len * 8;
        for (int j = 0; j < 8; j++) {
            final[120 + j] = (bit_len >> (56 - j * 8)) & 0xFF;
        }
        sha256_transform(state, final);
        sha256_transform(state, final + 64);
    }
    
    // Output
    #pragma unroll
    for (i = 0; i < 8; i++) {
        output[i*4] = (state[i] >> 24) & 0xFF;
        output[i*4+1] = (state[i] >> 16) & 0xFF;
        output[i*4+2] = (state[i] >> 8) & 0xFF;
        output[i*4+3] = state[i] & 0xFF;
    }
}

// ==================== RIPEMD160 di GPU (simplified) ====================
__device__ void ripemd160_gpu(const unsigned char* data, int len, unsigned char* output) {
    // Simplified version for demonstration
    unsigned char hash[20];
    
    // Simple deterministic hash from data
    for (int i = 0; i < 20; i++) {
        hash[i] = 0;
        for (int j = 0; j < len && j < 32; j++) {
            hash[i] ^= data[j] + (i * 17);
        }
    }
    
    memcpy(output, hash, 20);
}

// ==================== ARITHMETIC FIELD OPERATIONS ====================
__device__ int compare_big_int(const unsigned char* a, const unsigned char* b, int size = 32) {
    for (int i = size - 1; i >= 0; i--) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

__device__ void fe_add_mod(unsigned char* r, const unsigned char* a, const unsigned char* b) {
    unsigned int carry = 0;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        unsigned int sum = a[i] + b[i] + carry;
        r[i] = sum & 0xFF;
        carry = sum >> 8;
    }
    
    // Modulo reduction jika >= p
    if (carry || compare_big_int(r, d_secp256k1_p) >= 0) {
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

__device__ void fe_mul_mod(unsigned char* r, const unsigned char* a, const unsigned char* b) {
    unsigned int temp[64] = {0};
    
    // Multiply
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
    
    // Mod p reduction (simplified)
    // Copy low 256 bits
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        r[i] = temp[i];
    }
    
    // Add high bits multiplied by 2^32 mod p
    unsigned char high_part[32] = {0};
    #pragma unroll
    for (int i = 32; i < 64; i++) {
        // Simplified reduction
        if (temp[i] != 0) {
            high_part[0] ^= temp[i];
            high_part[4] ^= temp[i];
        }
    }
    
    fe_add_mod(r, r, high_part);
    
    // Final reduction
    if (compare_big_int(r, d_secp256k1_p) >= 0) {
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

// ==================== KERNEL BRUTEFORCE UTAMA ====================
__global__ void bruteforce_kernel(
    SearchResult* results,
    int* found_count,
    unsigned long long seed,
    int total_threads
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_threads) return;
    
    // Inisialisasi RNG
    curandState_t state;
    curand_init(seed, tid, 0, &state);
    
    // Shared memory untuk target hashes
    __shared__ unsigned char s_targets[MAX_TARGETS][HASH160_SIZE];
    __shared__ int s_num_targets;
    
    if (threadIdx.x == 0) {
        s_num_targets = d_num_targets;
    }
    __syncthreads();
    
    // Load targets ke shared memory
    for (int i = threadIdx.x; i < d_num_targets; i += blockDim.x) {
        memcpy(s_targets[i], d_target_hashes[i], HASH160_SIZE);
    }
    __syncthreads();
    
    // Buffer untuk perhitungan
    unsigned char private_key[32];
    unsigned char public_key[33];
    unsigned char sha256_hash[32];
    unsigned char ripemd160_hash[20];
    
    // Generate dan test BATCH_SIZE keys per thread
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        // Generate random private key
        for (int i = 0; i < 32; i++) {
            private_key[i] = (unsigned char)curand(&state);
        }
        
        // Pastikan private key valid (1 <= key < n)
        if (private_key[0] == 0) continue;
        if (compare_big_int(private_key, d_secp256k1_n) >= 0) continue;
        
        // SIMULASI: Untuk demo, kita buat public key sederhana
        // Dalam implementasi nyata, gunakan scalar multiplication
        
        // Compressed public key format (0x02/0x03 + x)
        public_key[0] = (private_key[0] & 1) ? 0x03 : 0x02;
        
        // Buat x coordinate dari private key (simplified)
        for (int i = 0; i < 32; i++) {
            public_key[i + 1] = private_key[i] ^ (i * 13);
        }
        
        // Hash160: SHA256 lalu RIPEMD160
        sha256_gpu(public_key, 33, sha256_hash);
        ripemd160_gpu(sha256_hash, 32, ripemd160_hash);
        
        // Bandingkan dengan semua targets
        for (int i = 0; i < s_num_targets; i++) {
            bool match = true;
            #pragma unroll
            for (int j = 0; j < HASH160_SIZE; j++) {
                if (ripemd160_hash[j] != s_targets[i][j]) {
                    match = false;
                    break;
                }
            }
            
            if (match) {
                int found_idx = atomicAdd(found_count, 1);
                if (found_idx < MAX_TARGETS) {
                    // Simpan hasil
                    memcpy(results[found_idx].private_key, private_key, 32);
                    memcpy(results[found_idx].public_key, public_key, 33);
                    results[found_idx].target_index = i;
                    results[found_idx].found = 1;
                    
                    // Buat address string sederhana
                    char addr[36];
                    addr[0] = '1';
                    for (int j = 1; j < 35; j++) {
                        addr[j] = 'A' + (ripemd160_hash[j % 20] % 26);
                    }
                    addr[35] = '\0';
                    memcpy(results[found_idx].address, addr, 36);
                }
                break;
            }
        }
    }
}

// ==================== FUNGSI CPU/OPENSSL ====================

// Base58 encoding di CPU
string base58_encode(const unsigned char* data, int len) {
    const char alphabet[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    vector<unsigned char> digits((len * 138 / 100) + 1);
    int digitslen = 1;
    
    for (int i = 0; i < len; i++) {
        unsigned int carry = data[i];
        for (int j = 0; j < digitslen; j++) {
            carry += (unsigned int)digits[j] << 8;
            digits[j] = carry % 58;
            carry /= 58;
        }
        while (carry > 0) {
            digits[digitslen++] = carry % 58;
            carry /= 58;
        }
    }
    
    string result;
    for (int i = 0; i < len; i++) {
        if (data[i] != 0) break;
        result += '1';
    }
    
    for (int i = 0; i < digitslen; i++) {
        result += alphabet[digits[digitslen - 1 - i]];
    }
    
    return result;
}

// Decode Base58 address ke hash160
bool decode_base58(const string& address, unsigned char* hash160) {
    const char alphabet[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    vector<unsigned char> digits(25);
    
    // Decode base58
    int zeros = 0;
    while (zeros < address.size() && address[zeros] == '1') zeros++;
    
    vector<unsigned char> bytes((address.size() - zeros) * 733 / 1000 + 1);
    
    for (char c : address) {
        const char* pos = strchr(alphabet, c);
        if (!pos) return false;
        
        unsigned int carry = pos - alphabet;
        for (int i = bytes.size() - 1; i >= 0; i--) {
            carry += 58 * bytes[i];
            bytes[i] = carry % 256;
            carry /= 256;
        }
    }
    
    // Skip leading zeros
    int start = zeros;
    while (start < bytes.size() && bytes[start] == 0) start++;
    
    // Verifikasi checksum
    if (bytes.size() - start != 25) return false;
    
    unsigned char checksum1[32], checksum2[32];
    SHA256(&bytes[start], 21, checksum1);
    SHA256(checksum1, 32, checksum2);
    
    if (memcmp(&bytes[start + 21], checksum2, 4) != 0) return false;
    
    // Extract hash160
    memcpy(hash160, &bytes[start + 1], 20);
    return true;
}

// Generate Bitcoin address dari public key
string generate_address(const unsigned char* pubkey, bool compressed = true) {
    unsigned char sha256_hash[32];
    unsigned char ripemd160_hash[20];
    
    if (compressed) {
        SHA256(pubkey, 33, sha256_hash);
    } else {
        SHA256(pubkey, 65, sha256_hash);
    }
    RIPEMD160(sha256_hash, 32, ripemd160_hash);
    
    // Tambah version byte (0x00 untuk mainnet)
    unsigned char version_hash160[21];
    version_hash160[0] = 0x00;
    memcpy(version_hash160 + 1, ripemd160_hash, 20);
    
    // Hitung checksum
    unsigned char checksum1[32], checksum2[32];
    SHA256(version_hash160, 21, checksum1);
    SHA256(checksum1, 32, checksum2);
    
    // Buat binary address
    unsigned char address_bin[25];
    memcpy(address_bin, version_hash160, 21);
    memcpy(address_bin + 21, checksum2, 4);
    
    // Encode ke base58
    return base58_encode(address_bin, 25);
}

// Generate WIF dari private key
string generate_wif(const unsigned char* private_key) {
    // WIF format: 0x80 + private_key + 0x01 + checksum(4 bytes)
    unsigned char wif_bytes[38];
    wif_bytes[0] = 0x80;
    memcpy(wif_bytes + 1, private_key, 32);
    wif_bytes[33] = 0x01; // compression flag
    
    // Double SHA256 untuk checksum
    unsigned char checksum1[32], checksum2[32];
    SHA256(wif_bytes, 34, checksum1);
    SHA256(checksum1, 32, checksum2);
    
    memcpy(wif_bytes + 34, checksum2, 4);
    
    return base58_encode(wif_bytes, 38);
}

// ==================== FUNGSI UTAMA ====================
int main(int argc, char* argv[]) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <address_list.txt>" << endl;
        cout << "Example: " << argv[0] << " list.txt" << endl;
        return 1;
    }
    
    // Baca file address
    ifstream file(argv[1]);
    if (!file.is_open()) {
        cout << "Error: Cannot open file " << argv[1] << endl;
        return 1;
    }
    
    vector<string> addresses;
    string line;
    while (getline(file, line)) {
        if (!line.empty()) {
            addresses.push_back(line);
        }
    }
    file.close();
    
    cout << "Loaded " << addresses.size() << " addresses from " << argv[1] << endl;
    
    // Decode addresses ke hash160
    vector<vector<unsigned char>> target_hashes;
    for (const string& addr : addresses) {
        unsigned char hash160[20];
        if (decode_base58(addr, hash160)) {
            target_hashes.push_back(vector<unsigned char>(hash160, hash160 + 20));
        } else {
            cout << "Warning: Invalid address format - " << addr << endl;
        }
    }
    
    if (target_hashes.empty()) {
        cout << "Error: No valid addresses found" << endl;
        return 1;
    }
    
    cout << "Valid targets: " << target_hashes.size() << endl;
    
    // Dapatkan info GPU
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    cout << "\n=== GPU Information ===" << endl;
    cout << "GPU: " << prop.name << endl;
    cout << "Compute Capability: " << prop.major << "." << prop.minor << endl;
    cout << "Total Memory: " << prop.totalGlobalMem / (1024*1024*1024.0) << " GB" << endl;
    cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << endl;
    cout << "MultiProcessors: " << prop.multiProcessorCount << endl;
    
    // Setup target hashes di GPU
    unsigned char* flat_hashes = new unsigned char[target_hashes.size() * 20];
    for (size_t i = 0; i < target_hashes.size(); i++) {
        memcpy(flat_hashes + i * 20, target_hashes[i].data(), 20);
    }
    
    cudaMemcpyToSymbol(d_target_hashes, flat_hashes, target_hashes.size() * 20);
    int num_targets = target_hashes.size();
    cudaMemcpyToSymbol(d_num_targets, &num_targets, sizeof(int));
    
    delete[] flat_hashes;
    
    // Alokasi memori GPU
    SearchResult* d_results;
    int* d_found_count;
    
    cudaMalloc(&d_results, MAX_TARGETS * sizeof(SearchResult));
    cudaMalloc(&d_found_count, sizeof(int));
    
    cudaMemset(d_found_count, 0, sizeof(int));
    cudaMemset(d_results, 0, MAX_TARGETS * sizeof(SearchResult));
    
    // Alokasi memori CPU untuk hasil
    SearchResult* h_results = new SearchResult[MAX_TARGETS];
    int h_found_count = 0;
    
    // Konfigurasi kernel
    int threads = THREADS_PER_BLOCK;
    int blocks = prop.multiProcessorCount * 8; // Optimasi
    int total_threads = blocks * threads;
    
    cout << "\n=== Kernel Configuration ===" << endl;
    cout << "Blocks: " << blocks << endl;
    cout << "Threads per Block: " << threads << endl;
    cout << "Total Threads: " << total_threads << endl;
    cout << "Keys per Thread: " << BATCH_SIZE << endl;
    cout << "Total Keys per Iteration: " << total_threads * BATCH_SIZE << endl;
    
    // Seed random
    unsigned long long seed = duration_cast<milliseconds>(
        system_clock::now().time_since_epoch()).count();
    
    cout << "\n=== Starting Bruteforce ===" << endl;
    cout << "Press Ctrl+C to stop" << endl;
    cout << "=============================" << endl;
    
    // Variabel statistik
    long long total_tested = 0;
    auto start_time = high_resolution_clock::now();
    auto last_print_time = start_time;
    
    // Main loop
    try {
        for (int iteration = 0; iteration < 1000000; iteration++) {
            // Jalankan kernel
            bruteforce_kernel<<<blocks, threads>>>(
                d_results, d_found_count, seed + iteration, total_threads
            );
            
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                cout << "CUDA Error: " << cudaGetErrorString(err) << endl;
                break;
            }
            
            cudaDeviceSynchronize();
            
            // Copy hasil
            cudaMemcpy(&h_found_count, d_found_count, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_results, d_results, MAX_TARGETS * sizeof(SearchResult), cudaMemcpyDeviceToHost);
            
            total_tested += total_threads * BATCH_SIZE;
            
            // Print progress
            auto current_time = high_resolution_clock::now();
            auto elapsed = duration_cast<milliseconds>(current_time - last_print_time).count() / 1000.0;
            
            if (elapsed >= 2.0) {
                auto total_elapsed = duration_cast<milliseconds>(current_time - start_time).count() / 1000.0;
                double keys_per_sec = total_tested / total_elapsed;
                
                cout << fixed << setprecision(2);
                cout << "\r[Progress] Keys: " << total_tested 
                     << " | Speed: " << keys_per_sec / 1000000 << " Mkeys/sec"
                     << " | Found: " << h_found_count << "      " << flush;
                
                last_print_time = current_time;
            }
            
            // Jika ditemukan match
            if (h_found_count > 0) {
                cout << "\n\n=== FOUND " << h_found_count << " MATCHES ===" << endl;
                
                // Simpan ke file
                ofstream outfile("found_keys.txt", ios::app);
                
                for (int i = 0; i < h_found_count; i++) {
                    cout << "\n--- Match " << (i + 1) << " ---" << endl;
                    
                    // Private key hex
                    cout << "Private Key: ";
                    for (int j = 0; j < 32; j++) {
                        printf("%02x", h_results[i].private_key[j]);
                    }
                    cout << endl;
                    
                    // Generate address untuk verifikasi
                    string address = generate_address(h_results[i].public_key, true);
                    cout << "Bitcoin Address: " << address << endl;
                    
                    // WIF format
                    string wif = generate_wif(h_results[i].private_key);
                    cout << "WIF: " << wif << endl;
                    
                    // Target address
                    if (h_results[i].target_index < addresses.size()) {
                        cout << "Target: " << addresses[h_results[i].target_index] << endl;
                    }
                    
                    // Simpan ke file
                    outfile << "Private Key: ";
                    for (int j = 0; j < 32; j++) {
                        outfile << hex << setw(2) << setfill('0') 
                               << (int)h_results[i].private_key[j];
                    }
                    outfile << endl;
                    outfile << "Address: " << address << endl;
                    outfile << "WIF: " << wif << endl;
                    outfile << "Target: " << addresses[h_results[i].target_index] << endl;
                    outfile << "-------------------" << endl;
                }
                
                outfile.close();
                cout << "\nResults saved to found_keys.txt" << endl;
                
                // Reset counter
                h_found_count = 0;
                cudaMemset(d_found_count, 0, sizeof(int));
            }
            
            // Cek waktu (maksimal 1 jam)
            auto total_elapsed = duration_cast<seconds>(current_time - start_time).count();
            if (total_elapsed > 3600) { // 1 jam
                cout << "\n\nTime limit reached (1 hour). Stopping." << endl;
                break;
            }
        }
    } catch (const exception& e) {
        cout << "\nError: " << e.what() << endl;
    }
    
    // Statistik akhir
    auto end_time = high_resolution_clock::now();
    auto total_elapsed = duration_cast<milliseconds>(end_time - start_time).count() / 1000.0;
    
    cout << "\n\n=== FINAL STATISTICS ===" << endl;
    cout << "Total keys tested: " << total_tested << endl;
    cout << "Total time: " << total_elapsed << " seconds" << endl;
    cout << "Average speed: " << (total_tested / total_elapsed) / 1000000 << " Mkeys/second" << endl;
    cout << "GPU: " << prop.name << endl;
    
    // Cleanup
    delete[] h_results;
    cudaFree(d_results);
    cudaFree(d_found_count);
    
    cout << "\nProgram finished." << endl;
    return 0;
}
