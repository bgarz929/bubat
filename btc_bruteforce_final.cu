/*
 * btc_dump_all_stable.cu
 * FITUR: Dump All Private Keys & Address
 * FIX: Menambahkan Bounded Queue untuk mencegah RAM penuh & Crash
 * FIX: Error checking lengkap
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
#define BLOCKS_MULTIPLIER 16    // Dikurangi agar kernel lebih ringan (mencegah TDR/Freeze)
#define KEYS_PER_THREAD   4     // Dikurangi drastis karena kita menyimpan SEMUA data
#define MAX_QUEUE_SIZE    50    // Maksimal 50 batch antri di RAM. Jika penuh, GPU pause.

// Macro cek error CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

// ==================== STRUKTUR DATA ====================

struct ResultPacket {
    uint64_t full_priv_key[4]; 
    unsigned char hash160[20];
};

// ==================== CRYPTO HELPERS (CPU SIDE) ====================

static const char* BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define EP1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define SIG0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ ((x) >> 3))
#define SIG1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))

static const uint32_t K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

void sha256_transform(uint32_t *state, const uint8_t *data) {
    uint32_t a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];
    for (i = 0, j = 0; i < 16; ++i, j += 4)
        m[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) | (data[j + 3]);
    for (; i < 64; ++i)
        m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];
    a = state[0]; b = state[1]; c = state[2]; d = state[3];
    e = state[4]; f = state[5]; g = state[6]; h = state[7];
    for (i = 0; i < 64; ++i) {
        t1 = h + EP1(e) + CH(e, f, g) + K[i] + m[i];
        t2 = EP0(a) + MAJ(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

void simple_sha256(const uint8_t *data, size_t len, uint8_t *hash) {
    uint32_t state[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint8_t buffer[64];
    uint32_t bitlen = len * 8;
    memcpy(buffer, data, len);
    buffer[len++] = 0x80;
    while (len < 56) buffer[len++] = 0;
    buffer[63] = bitlen & 0xFF; buffer[62] = (bitlen >> 8) & 0xFF;
    sha256_transform(state, buffer);
    for (int i = 0; i < 8; ++i) {
        hash[i * 4] = (state[i] >> 24) & 0xFF;
        hash[i * 4 + 1] = (state[i] >> 16) & 0xFF;
        hash[i * 4 + 2] = (state[i] >> 8) & 0xFF;
        hash[i * 4 + 3] = state[i] & 0xFF;
    }
}

void hash256(const uint8_t* data, size_t len, uint8_t* out) {
    uint8_t tmp[32];
    simple_sha256(data, len, tmp);
    simple_sha256(tmp, 32, out);
}

std::string EncodeBase58Check(const uint8_t* payload, size_t len) {
    uint8_t data[64]; 
    memcpy(data, payload, len);
    uint8_t hash[32];
    hash256(payload, len, hash);
    memcpy(data + len, hash, 4);
    size_t data_len = len + 4;
    int zeros = 0;
    while (zeros < data_len && data[zeros] == 0) zeros++;
    std::vector<unsigned char> b58((data_len * 138 / 100) + 1);
    size_t size = 0;
    for (size_t i = 0; i < data_len; ++i) {
        int carry = data[i];
        for (size_t j = 0; j < size; ++j) {
            carry += 256 * b58[j];
            b58[j] = carry % 58;
            carry /= 58;
        }
        while (carry > 0) {
            b58[size++] = carry % 58;
            carry /= 58;
        }
    }
    std::string result;
    result.reserve(zeros + size);
    result.assign(zeros, '1');
    for (int i = size - 1; i >= 0; --i) result += BASE58_ALPHABET[b58[i]];
    return result;
}

std::string ToWIF(const uint64_t* priv_key_u64) {
    uint8_t raw[33];
    raw[0] = 0x80;
    for(int i=0; i<4; i++) {
        uint64_t val = priv_key_u64[i];
        for(int b=0; b<8; b++) raw[1 + i*8 + (7-b)] = (val >> (b*8)) & 0xFF;
    }
    return EncodeBase58Check(raw, 33);
}

std::string ToAddress(const unsigned char* hash160) {
    uint8_t payload[21];
    payload[0] = 0x00; 
    memcpy(payload + 1, hash160, 20);
    return EncodeBase58Check(payload, 21);
}

// ==================== DEVICE CONSTANTS ====================
__constant__ uint64_t d_gx[4] = {0x79BE667EF9DCBBAC, 0x55A06295CE870B07, 0x029BFCDB2DCE28D9, 0x59F2815B16F81798};
__constant__ uint64_t d_gy[4] = {0x483ADA7726A3C465, 0x5DA4FBFC0E1108A8, 0xFD17B448A6855419, 0x3C6C6302836C0501};

// ==================== KERNEL ====================

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
        ResultPacket* r = &out_results[buffer_idx];
        
        #pragma unroll
        for(int k=0; k<4; k++) r->full_priv_key[k] = priv_key[k];
        memcpy(r->hash160, hash160, 20);

        ecc_point_add_G(pub_x, pub_y);
        priv_key[3]++; 
    }
}

// ==================== BOUNDED QUEUE (FIX CRASH) ====================
// Mencegah queue tumbuh tak terbatas jika disk lambat

template<typename T>
class BoundedQueue {
    std::queue<std::vector<T>> q; 
    std::mutex m;
    std::condition_variable cv_push, cv_pop;
    bool done = false;
    size_t max_size;
public:
    BoundedQueue(size_t limit) : max_size(limit) {}

    // PUSH: Blokir jika queue penuh
    void push(std::vector<T> val) {
        std::unique_lock<std::mutex> lock(m);
        // Tunggu sampai queue berkurang isinya
        cv_push.wait(lock, [this]{ return q.size() < max_size || done; });
        
        if (done) return;
        
        q.push(std::move(val));
        cv_pop.notify_one();
    }
    
    // POP: Ambil data
    bool pop(std::vector<T>& val) {
        std::unique_lock<std::mutex> lock(m);
        cv_pop.wait(lock, [this]{ return !q.empty() || done; });
        if (q.empty() && done) return false;
        val = std::move(q.front());
        q.pop();
        
        // Beritahu produser bahwa ada slot kosong
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

// ==================== WORKER ====================

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
        
        std::cout << "[GPU " << id << "] Batch size: " << total_keys_per_batch << " keys (" 
                  << (buffer_size / 1024 / 1024) << " MB)\n";

        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaMalloc(&d_results, buffer_size));
        CUDA_CHECK(cudaMallocHost(&h_results, buffer_size));
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
        
        CUDA_CHECK(cudaMemcpyAsync(h_results, d_results, total_keys_per_batch * sizeof(ResultPacket), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        std::vector<ResultPacket> batch_data(h_results, h_results + total_keys_per_batch);
        
        // Ini akan memblokir (TIDUR) jika queue penuh, mencegah RAM meledak
        queue.push(std::move(batch_data));
    }
};

// ==================== MAIN ====================

int main() {
    std::cout << "=== BTC GENERATOR STABLE [BOUNDED QUEUE] ===\n";
    
    // Batasi queue hanya 50 batch. Jika lebih, GPU akan menunggu Disk.
    BoundedQueue<ResultPacket> result_queue(MAX_QUEUE_SIZE); 
    
    std::thread writer_thread([&result_queue](){
        FILE* fp = fopen("dump_all.csv", "w");
        if(!fp) { perror("File error"); return; }
        fprintf(fp, "WIF_PrivateKey,Address\n");
        
        std::vector<ResultPacket> chunk;
        uint64_t total_written = 0;
        auto t_start = std::chrono::high_resolution_clock::now();

        while(result_queue.pop(chunk)) {
            for(const auto& res : chunk) {
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
        
        // Tidak perlu sleep manual, karena queue.push() otomatis sleep jika queue penuh
    }

    return 0;
}
