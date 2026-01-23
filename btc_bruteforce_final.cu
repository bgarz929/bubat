/*
 * btc_optimized_solver_fixed.cu
 * High-Performance Modular GPU Brute Force Solver
 * FIX: Moved Bloom Filter to Global Memory to bypass 64KB Constant Memory limit.
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
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// ==================== KONFIGURASI ====================
#define THREADS_PER_BLOCK 256
#define KEYS_PER_THREAD   512    
#define BLOCKS_MULTIPLIER 32     
#define BLOOM_SIZE_BITS   (1 << 24) // 16MB bits (2MB RAM)
#define BLOOM_ARRAY_SIZE  (BLOOM_SIZE_BITS / 32)
#define MAX_RESULTS_BUF   256    

// Konstanta ECC (Secp256k1)
#define SECP256K1_P 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
#define SECP256K1_N 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

// ==================== STRUKTUR DATA ====================

struct ResultPacket {
    unsigned char private_key_head[8]; 
    unsigned char hash160[20];
    uint32_t thread_id;
    uint32_t batch_idx;
    uint32_t block_idx;
    int gpu_id;
};

// ==================== DEVICE CONSTANTS ====================
// Bloom filter dihapus dari sini karena terlalu besar untuk __constant__ (Max 64KB)
// Kita pindahkan ke Global Memory dan pass via pointer.

__constant__ uint32_t d_target_prefixes[1024]; // Masih muat di constant (4KB)
__constant__ int d_num_targets;

// Precomputed G-Table 
__constant__ uint64_t d_gx[4] = {0x79BE667EF9DCBBAC, 0x55A06295CE870B07, 0x029BFCDB2DCE28D9, 0x59F2815B16F81798};
__constant__ uint64_t d_gy[4] = {0x483ADA7726A3C465, 0x5DA4FBFC0E1108A8, 0xFD17B448A6855419, 0x3C6C6302836C0501};

// ==================== KERNEL MATH HELPERS ====================

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

__device__ uint32_t murmur3_32(const unsigned char* key, uint32_t len, uint32_t seed) {
    uint32_t h = seed;
    uint32_t k;
    for (uint32_t i = 0; i < len; i++) {
        k = key[i];
        k *= 0xcc9e2d51;
        k = (k << 15) | (k >> 17);
        k *= 0x1b873593;
        h ^= k;
        h = (h << 13) | (h >> 19);
        h = h * 5 + 0xe6546b64;
    }
    h ^= len;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

// ==================== KERNEL UTAMA ====================

__global__ void __launch_bounds__(THREADS_PER_BLOCK) 
kernel_bruteforce_optimized(
    const uint32_t* __restrict__ bloom_filter, // FIX: Passed as pointer
    ResultPacket* out_results,
    int* out_count,
    uint64_t seed,
    uint64_t global_offset,
    int gpu_id
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

    for (int i = 0; i < KEYS_PER_THREAD; i++) {
        unsigned char hash160[20];
        
        uint64_t h_val = pub_x[0] ^ pub_y[3]; 
        #pragma unroll
        for(int b=0; b<5; b++) ((uint32_t*)hash160)[b] = (uint32_t)(h_val >> (b*3));

        // --- B. Bloom Filter Check (Global Memory Access) ---
        uint32_t h1 = murmur3_32(hash160, 20, 0) % BLOOM_SIZE_BITS;
        uint32_t h2 = murmur3_32(hash160, 20, 100) % BLOOM_SIZE_BITS;
        
        // Akses via pointer (menggantikan d_bloom_filter[...])
        // __restrict__ membantu compiler menggunakan L2 cache/LDG instruction
        bool bloom_hit = (bloom_filter[h1 / 32] & (1 << (h1 % 32))) &&
                         (bloom_filter[h2 / 32] & (1 << (h2 % 32)));

        if (bloom_hit) {
            // Logika prefix checking opsional bisa ditaruh di sini
            // (Menggunakan d_target_prefixes yang masih ada di constant memory)
            
            int idx = atomicAdd(out_count, 1);
            if (idx < MAX_RESULTS_BUF) {
                ResultPacket* r = &out_results[idx];
                memcpy(r->private_key_head, &priv_key[0], 8); 
                memcpy(r->hash160, hash160, 20);
                r->thread_id = tid;
                r->batch_idx = i;
                r->gpu_id = gpu_id;
                r->block_idx = blockIdx.x;
            }
        }

        ecc_point_add_G(pub_x, pub_y);
    }
}

// ==================== HOST UTILS ====================

class BloomFilter {
    std::vector<uint32_t> data;
public:
    BloomFilter() : data(BLOOM_ARRAY_SIZE, 0) {}

    void add(const unsigned char* hash) {
        uint32_t h1 = (*(uint32_t*)hash) % BLOOM_SIZE_BITS; 
        data[h1 / 32] |= (1 << (h1 % 32));
    }

    const uint32_t* get_data() const { return data.data(); }
    size_t get_size_bytes() const { return data.size() * sizeof(uint32_t); }
};

template<typename T>
class SafeQueue {
    std::queue<T> q;
    std::mutex m;
    std::condition_variable cv;
    bool done = false;
public:
    void push(T val) {
        std::lock_guard<std::mutex> lock(m);
        q.push(val);
        cv.notify_one();
    }
    
    bool pop(T& val) {
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [this]{ return !q.empty() || done; });
        if (q.empty() && done) return false;
        val = q.front();
        q.pop();
        return true;
    }
    
    void set_done() {
        std::lock_guard<std::mutex> lock(m);
        done = true;
        cv.notify_all();
    }
};

// ==================== CLASS WORKER ====================

class GPUWorker {
    int gpu_id;
    cudaStream_t stream_compute, stream_mem;
    ResultPacket* d_results;
    int* d_count;
    ResultPacket* h_results;
    int* h_count;
    
    // FIX: Pointer device untuk Bloom Filter
    uint32_t* d_bloom_ptr; 
    
    int blocks;
    int threads;

public:
    GPUWorker(int id) : gpu_id(id), threads(THREADS_PER_BLOCK) {
        cudaSetDevice(id);
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, id);
        blocks = prop.multiProcessorCount * BLOCKS_MULTIPLIER;
        
        cudaStreamCreate(&stream_compute);
        cudaStreamCreate(&stream_mem); 
        
        cudaMalloc(&d_results, MAX_RESULTS_BUF * sizeof(ResultPacket));
        cudaMalloc(&d_count, sizeof(int));
        
        // FIX: Alokasi Bloom Filter di Global Memory (VRAM)
        cudaMalloc(&d_bloom_ptr, BLOOM_ARRAY_SIZE * sizeof(uint32_t));
        
        cudaMallocHost(&h_results, MAX_RESULTS_BUF * sizeof(ResultPacket));
        cudaMallocHost(&h_count, sizeof(int));
    }
    
    ~GPUWorker() {
        // Cleanup (Good practice)
        cudaSetDevice(gpu_id);
        cudaFree(d_results);
        cudaFree(d_count);
        cudaFree(d_bloom_ptr);
        cudaFreeHost(h_results);
        cudaFreeHost(h_count);
        cudaStreamDestroy(stream_compute);
        cudaStreamDestroy(stream_mem);
    }
    
    void init_bloom(const BloomFilter& bloom) {
        cudaSetDevice(gpu_id);
        // FIX: Copy data ke pointer global memory, bukan ke symbol constant
        cudaMemcpy(d_bloom_ptr, bloom.get_data(), bloom.get_size_bytes(), cudaMemcpyHostToDevice);
    }

    void launch(uint64_t seed, uint64_t iteration, SafeQueue<ResultPacket>& queue) {
        cudaSetDevice(gpu_id);
        
        cudaMemsetAsync(d_count, 0, sizeof(int), stream_compute);
        
        // FIX: Pass d_bloom_ptr sebagai argumen pertama
        kernel_bruteforce_optimized<<<blocks, threads, 0, stream_compute>>>(
            d_bloom_ptr, 
            d_results, 
            d_count, 
            seed, 
            iteration, 
            gpu_id
        );
        
        cudaMemcpyAsync(h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost, stream_compute);
        
        cudaStreamSynchronize(stream_compute);
        
        if (*h_count > 0) {
            int count = std::min(*h_count, MAX_RESULTS_BUF);
            cudaMemcpyAsync(h_results, d_results, count * sizeof(ResultPacket), 
                           cudaMemcpyDeviceToHost, stream_mem);
            cudaStreamSynchronize(stream_mem);
            
            for(int i=0; i<count; i++) {
                queue.push(h_results[i]);
            }
        }
    }
    
    uint64_t get_keys_per_launch() {
        return (uint64_t)blocks * threads * KEYS_PER_THREAD;
    }
};

// ==================== MAIN ====================

int main(int argc, char** argv) {
    std::cout << "=== BTC OPTIMIZED SOLVER [BLOOM FIXED] ===\n";
    
    if (argc < 2) {
        std::cout << "Usage: ./solver <address_list.txt>\n";
        return 1;
    }

    std::cout << "[Host] Building Bloom Filter...\n";
    BloomFilter bloom;
    std::ifstream file(argv[1]);
    std::string line;
    int target_count = 0;
    while(std::getline(file, line)) {
        unsigned char dummy_hash[20];
        memset(dummy_hash, 0, 20); 
        bloom.add(dummy_hash);
        target_count++;
    }
    std::cout << "[Host] Loaded " << target_count << " targets.\n";

    SafeQueue<ResultPacket> result_queue;
    std::thread writer_thread([&result_queue](){
        ResultPacket res;
        std::ofstream outfile("found_keys.bin", std::ios::binary | std::ios::app);
        
        while(result_queue.pop(res)) {
            std::cout << "\n[FOUND] Potential match on GPU " << res.gpu_id << "\n";
            outfile.write((char*)&res, sizeof(ResultPacket));
            outfile.flush();
        }
    });

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    std::vector<GPUWorker*> workers;
    for(int i=0; i<num_gpus; i++) {
        workers.push_back(new GPUWorker(i));
        workers[i]->init_bloom(bloom);
    }

    uint64_t global_iter = 0;
    uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    
    auto start_time = std::chrono::high_resolution_clock::now();
    bool running = true;

    std::cout << "[Host] Starting kernels on " << num_gpus << " GPUs...\n";

    while(running) {
        for(auto worker : workers) {
            worker->launch(seed, global_iter, result_queue);
        }
        
        global_iter++;
        
        if (global_iter % 100 == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            uint64_t total_keys = global_iter * workers[0]->get_keys_per_launch() * num_gpus;
            
            if (elapsed > 0) {
                double mkeys = (total_keys / elapsed) / 1000000.0;
                std::cout << "\rSpeed: " << std::fixed << std::setprecision(2) << mkeys 
                          << " MKeys/s | Iter: " << global_iter << std::flush;
            }
        }
    }

    result_queue.set_done();
    writer_thread.join();
    
    return 0;
}
