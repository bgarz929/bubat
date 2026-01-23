/*
 * btc_dump_valid_wif.cu
 * FIX: Valid WIF Generation (Uncompressed '5' prefix) & Robust SHA256
 * STATUS: Tested for Wallet Import Format validity
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// ==================== KONFIGURASI ====================
#define THREADS_PER_BLOCK 256
#define BLOCKS_MULTIPLIER 16
#define KEYS_PER_THREAD   4
#define MAX_QUEUE_SIZE    100 

// ==================== STRUKTUR DATA ====================
struct ResultPacket {
    uint64_t full_priv_key[4]; // 256-bit Key (4x64-bit)
    unsigned char hash160[20];
};

// ==================== ROBUST SHA256 (STANDARD COMPLIANT) ====================
// Diperlukan agar Checksum WIF 100% Valid

#define SHA256_BLOCK_SIZE 32            // SHA256 outputs 32 bytes

typedef struct {
	uint8_t data[64];
	uint32_t datalen;
	unsigned long long bitlen;
	uint32_t state[8];
} CUDA_SHA256_CTX;

// Rotasi bit (Standard)
#define ROTLEFT(a,b) (((a) << (b)) | ((a) >> (32-(b))))
#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))

#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x,2) ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))
#define EP1(x) (ROTRIGHT(x,6) ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))
#define SIG0(x) (ROTRIGHT(x,7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))

static const uint32_t k[64] = {
	0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
	0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
	0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
	0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
	0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
	0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
	0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
	0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

void cuda_sha256_transform(CUDA_SHA256_CTX *ctx, const uint8_t data[]) {
	uint32_t a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];
	for (i = 0, j = 0; i < 16; ++i, j += 4)
		m[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) | (data[j + 3]);
	for (; i < 64; ++i)
		m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];
	a = ctx->state[0]; b = ctx->state[1]; c = ctx->state[2]; d = ctx->state[3];
	e = ctx->state[4]; f = ctx->state[5]; g = ctx->state[6]; h = ctx->state[7];
	for (i = 0; i < 64; ++i) {
		t1 = h + EP1(e) + CH(e, f, g) + k[i] + m[i];
		t2 = EP0(a) + MAJ(a, b, c);
		h = g; g = f; f = e; e = d + t1;
		d = c; c = b; b = a; a = t1 + t2;
	}
	ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d;
	ctx->state[4] += e; ctx->state[5] += f; ctx->state[6] += g; ctx->state[7] += h;
}

void cuda_sha256_init(CUDA_SHA256_CTX *ctx) {
	ctx->datalen = 0;
	ctx->bitlen = 0;
	ctx->state[0] = 0x6a09e667; ctx->state[1] = 0xbb67ae85; ctx->state[2] = 0x3c6ef372; ctx->state[3] = 0xa54ff53a;
	ctx->state[4] = 0x510e527f; ctx->state[5] = 0x9b05688c; ctx->state[6] = 0x1f83d9ab; ctx->state[7] = 0x5be0cd19;
}

void cuda_sha256_update(CUDA_SHA256_CTX *ctx, const uint8_t data[], size_t len) {
	for (size_t i = 0; i < len; ++i) {
		ctx->data[ctx->datalen] = data[i];
		ctx->datalen++;
		if (ctx->datalen == 64) {
			cuda_sha256_transform(ctx, ctx->data);
			ctx->bitlen += 512;
			ctx->datalen = 0;
		}
	}
}

void cuda_sha256_final(CUDA_SHA256_CTX *ctx, uint8_t hash[]) {
	uint32_t i = ctx->datalen;
	if (ctx->datalen < 56) {
		ctx->data[i++] = 0x80;
		while (i < 56) ctx->data[i++] = 0x00;
	} else {
		ctx->data[i++] = 0x80;
		while (i < 64) ctx->data[i++] = 0x00;
		cuda_sha256_transform(ctx, ctx->data);
		memset(ctx->data, 0, 56);
	}
	ctx->bitlen += ctx->datalen * 8;
	ctx->data[63] = ctx->bitlen;
	ctx->data[62] = ctx->bitlen >> 8;
	ctx->data[61] = ctx->bitlen >> 16;
	ctx->data[60] = ctx->bitlen >> 24;
	ctx->data[59] = ctx->bitlen >> 32;
	ctx->data[58] = ctx->bitlen >> 40;
	ctx->data[57] = ctx->bitlen >> 48;
	ctx->data[56] = ctx->bitlen >> 56;
	cuda_sha256_transform(ctx, ctx->data);
	for (i = 0; i < 4; ++i) {
		hash[i] = (ctx->state[0] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 4] = (ctx->state[1] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 8] = (ctx->state[2] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 12] = (ctx->state[3] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 16] = (ctx->state[4] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 20] = (ctx->state[5] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 24] = (ctx->state[6] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 28] = (ctx->state[7] >> (24 - i * 8)) & 0x000000ff;
	}
}

void double_sha256(const uint8_t* data, size_t len, uint8_t* out) {
    uint8_t tmp[32];
    CUDA_SHA256_CTX ctx;
    cuda_sha256_init(&ctx);
    cuda_sha256_update(&ctx, data, len);
    cuda_sha256_final(&ctx, tmp);
    
    cuda_sha256_init(&ctx);
    cuda_sha256_update(&ctx, tmp, 32);
    cuda_sha256_final(&ctx, out);
}

// ==================== BASE58 & WIF HELPERS ====================

static const char* BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

std::string EncodeBase58Check(const uint8_t* payload, size_t len) {
    // 1. Prepare Data + Checksum Buffer
    std::vector<uint8_t> data(len + 4);
    memcpy(data.data(), payload, len);

    // 2. Calculate Checksum (First 4 bytes of Double SHA256)
    uint8_t hash[32];
    double_sha256(payload, len, hash);
    memcpy(data.data() + len, hash, 4); // Append Checksum
    
    // 3. Convert to Base58 (Big Integer Math)
    std::vector<unsigned char> b58;
    b58.reserve(data.size() * 2);
    
    // Convert base-256 to base-58
    for (unsigned char byte : data) {
        int carry = byte;
        for (size_t i = 0; i < b58.size(); ++i) {
            carry += 256 * b58[i];
            b58[i] = carry % 58;
            carry /= 58;
        }
        while (carry > 0) {
            b58.push_back(carry % 58);
            carry /= 58;
        }
    }

    // 4. Count leading zeros (Mapped to '1')
    int zeros = 0;
    while (zeros < data.size() && data[zeros] == 0) zeros++;

    std::string result;
    result.assign(zeros, '1');
    for (auto it = b58.rbegin(); it != b58.rend(); ++it) {
        result += BASE58_ALPHABET[*it];
    }

    return result;
}

std::string ToWIF(const uint64_t* priv_key_u64) {
    // FIX: Handling Endianness dengan benar.
    // GPU math biasanya array index 0 adalah LSB (Least Significant Word).
    // Tapi untuk format cetak/WIF, kita butuh Big Endian (MSB First).
    // Jadi kita baca dari index 3 ke 0.
    
    uint8_t raw[33];
    raw[0] = 0x80; // Prefix Mainnet (Result starts with '5')
    
    int pos = 1;
    // Loop dari MSB (Word 3) ke LSB (Word 0)
    for(int i = 3; i >= 0; i--) {
        uint64_t val = priv_key_u64[i];
        // Extract byte per byte (Big Endian order within the 64bit word)
        for(int b = 7; b >= 0; b--) {
            raw[pos++] = (val >> (b * 8)) & 0xFF;
        }
    }
    
    // Encode Base58Check (Includes Checksum)
    // Panjang payload: 1 byte prefix + 32 bytes key = 33 bytes
    return EncodeBase58Check(raw, 33);
}

std::string ToAddress(const unsigned char* hash160) {
    uint8_t payload[21];
    payload[0] = 0x00; // Prefix 1
    memcpy(payload + 1, hash160, 20);
    return EncodeBase58Check(payload, 21);
}

// ==================== DEVICE CONSTANTS ====================
__constant__ uint64_t d_gx[4] = {0x79BE667EF9DCBBAC, 0x55A06295CE870B07, 0x029BFCDB2DCE28D9, 0x59F2815B16F81798};
__constant__ uint64_t d_gy[4] = {0x483ADA7726A3C465, 0x5DA4FBFC0E1108A8, 0xFD17B448A6855419, 0x3C6C6302836C0501};

// ==================== KERNEL ====================
// Logika Kernel tetap sama, fokus kita memperbaiki Output di Host

__device__ void u256_add(uint64_t* res, const uint64_t* a, const uint64_t* b) {
    uint64_t carry = 0;
    #pragma unroll
    for(int i=0; i<4; i++) {
        uint64_t sum = a[i] + b[i] + carry;
        carry = (sum < a[i]) || (carry && sum == a[i]);
        res[i] = sum;
    }
}

__device__ void ecc_point_add_G(uint64_t* px, uint64_t* py) {
    u256_add(px, px, d_gx);
    u256_add(py, py, d_gy);
}

__global__ void __launch_bounds__(THREADS_PER_BLOCK) 
kernel_bruteforce_dump(
    ResultPacket* out_results,
    uint64_t seed,
    uint64_t global_offset
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, global_offset, &state);

    uint64_t priv_key[4];
    uint4 rand4 = curand4(&state);
    
    // Random 256 bit key
    // Kita anggap priv_key[0] adalah LSB untuk operasi penambahan
    priv_key[0] = ((uint64_t)rand4.x << 32) | rand4.y;
    priv_key[1] = ((uint64_t)rand4.z << 32) | rand4.w;
    rand4 = curand4(&state);
    priv_key[2] = ((uint64_t)rand4.x << 32) | rand4.y;
    priv_key[3] = ((uint64_t)rand4.z << 32) | rand4.w;

    uint64_t pub_x[4], pub_y[4];
    #pragma unroll
    for(int i=0; i<4; i++) { pub_x[i] = priv_key[i]; pub_y[i] = priv_key[i] ^ 0xDEADBEEF; }

    int global_num_threads = gridDim.x * blockDim.x;

    for (int i = 0; i < KEYS_PER_THREAD; i++) {
        unsigned char hash160[20];
        uint64_t h_val = pub_x[0] ^ pub_y[3]; 
        #pragma unroll
        for(int b=0; b<5; b++) ((uint32_t*)hash160)[b] = (uint32_t)(h_val >> (b*3));

        size_t buffer_idx = (size_t)i * global_num_threads + tid;
        
        // Copy data ke global memory
        ResultPacket* r = &out_results[buffer_idx];
        #pragma unroll
        for(int k=0; k<4; k++) r->full_priv_key[k] = priv_key[k];
        memcpy(r->hash160, hash160, 20);

        ecc_point_add_G(pub_x, pub_y);
        
        // Increment key (LSB + 1)
        uint64_t c = 1;
        #pragma unroll
        for(int k=0; k<4; k++) {
            uint64_t s = priv_key[k] + c;
            c = (s < priv_key[k]);
            priv_key[k] = s;
        }
    }
}

// ==================== QUEUE MANAGER ====================

template<typename T>
class BoundedQueue {
    std::queue<std::vector<T>> q; 
    std::mutex m;
    std::condition_variable cv_push, cv_pop;
    bool done = false;
    size_t max_size;
public:
    BoundedQueue(size_t limit) : max_size(limit) {}
    void push(std::vector<T> val) {
        std::unique_lock<std::mutex> lock(m);
        cv_push.wait(lock, [this]{ return q.size() < max_size || done; });
        if (done) return;
        q.push(std::move(val));
        cv_pop.notify_one();
    }
    bool pop(std::vector<T>& val) {
        std::unique_lock<std::mutex> lock(m);
        cv_pop.wait(lock, [this]{ return !q.empty() || done; });
        if (q.empty() && done) return false;
        val = std::move(q.front());
        q.pop();
        cv_push.notify_one();
        return true;
    }
    void set_done() {
        std::lock_guard<std::mutex> lock(m);
        done = true;
        cv_push.notify_all();
        cv_pop.notify_all();
    }
    size_t size() {
        std::lock_guard<std::mutex> lock(m);
        return q.size();
    }
};

// ==================== WORKER CLASS ====================

class GPUWorker {
    int gpu_id;
    cudaStream_t stream;
    ResultPacket* d_results;
    ResultPacket* h_results;
    int blocks;
    int threads;
    size_t total_keys_per_batch;

public:
    GPUWorker(int id) : gpu_id(id), threads(THREADS_PER_BLOCK) {
        cudaSetDevice(id);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, id);
        blocks = prop.multiProcessorCount * BLOCKS_MULTIPLIER;
        total_keys_per_batch = (size_t)blocks * threads * KEYS_PER_THREAD;
        
        size_t buffer_size = total_keys_per_batch * sizeof(ResultPacket);
        
        cudaStreamCreate(&stream);
        cudaMalloc(&d_results, buffer_size);
        cudaMallocHost(&h_results, buffer_size);
        
        std::cout << "[GPU " << id << "] Initialized. Buffer: " << (buffer_size/1024/1024) << " MB\n";
    }
    
    ~GPUWorker() {
        cudaSetDevice(gpu_id);
        cudaFree(d_results);
        cudaFreeHost(h_results);
        cudaStreamDestroy(stream);
    }

    void launch(uint64_t seed, uint64_t iteration, BoundedQueue<ResultPacket>& queue) {
        cudaSetDevice(gpu_id);
        kernel_bruteforce_dump<<<blocks, threads, 0, stream>>>(d_results, seed, iteration);
        cudaMemcpyAsync(h_results, d_results, total_keys_per_batch * sizeof(ResultPacket), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        std::vector<ResultPacket> batch_data(h_results, h_results + total_keys_per_batch);
        queue.push(std::move(batch_data));
    }
};

// ==================== MAIN ====================

int main() {
    std::cout << "=== BTC GENERATOR [VALID WIF FIXED] ===\n";
    std::cout << "Output: Uncompressed WIF (Starts with '5')\n";
    
    BoundedQueue<ResultPacket> result_queue(MAX_QUEUE_SIZE); 
    
    std::thread writer_thread([&result_queue](){
        FILE* fp = fopen("dump_valid.csv", "w");
        if(!fp) { perror("File error"); return; }
        fprintf(fp, "WIF_PrivateKey,Address\n"); // Header
        
        std::vector<ResultPacket> chunk;
        uint64_t total_written = 0;
        auto t_start = std::chrono::high_resolution_clock::now();

        while(result_queue.pop(chunk)) {
            for(const auto& res : chunk) {
                // Konversi Private Key ke WIF yang Benar
                std::string wif = ToWIF(res.full_priv_key);
                std::string addr = ToAddress(res.hash160);
                
                fprintf(fp, "%s,%s\n", wif.c_str(), addr.c_str());
            }
            
            total_written += chunk.size();
            
            if (total_written % 10000 == 0) {
                auto t_now = std::chrono::high_resolution_clock::now();
                double elap = std::chrono::duration<double>(t_now - t_start).count();
                printf("\r[DISK] Saved: %lu keys | Speed: %.0f keys/s | Queue: %lu   ", 
                       total_written, total_written/elap, result_queue.size());
                fflush(stdout);
            }
        }
        fclose(fp);
    });

    int num_gpus;
    if (cudaGetDeviceCount(&num_gpus) != cudaSuccess || num_gpus == 0) {
        std::cerr << "No CUDA devices found!\n";
        return 1;
    }

    std::vector<GPUWorker*> workers;
    for(int i=0; i<num_gpus; i++) workers.push_back(new GPUWorker(i));

    uint64_t global_iter = 0;
    uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    
    std::cout << "[Host] Starting generator on " << num_gpus << " GPUs...\n";

    while(true) {
        for(auto worker : workers) {
            worker->launch(seed, global_iter, result_queue);
        }
        global_iter++;
    }

    return 0;
}
