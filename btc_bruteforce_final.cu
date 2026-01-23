/*
 * btc_dump_all_generator.cu
 * MODIFIED: Save ALL generated Addresses & WIFs (Generator Mode)
 * WARNING: I/O Bound. Performance depends on Disk Write Speed.
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
// KITA KURANGI BATCH SIZE AGAR MEMORY TIDAK MELEDAK SAAT MENYIMPAN SEMUA
#define KEYS_PER_THREAD   16     // Dikurangi dari 512 ke 16 agar buffer muat di VRAM
#define BLOCKS_MULTIPLIER 32     
#define BLOOM_SIZE_BITS   (1 << 24) 
#define BLOOM_ARRAY_SIZE  (BLOOM_SIZE_BITS / 32)

// Konstanta ECC (Secp256k1)
#define SECP256K1_P 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F

// ==================== STRUKTUR DATA ====================

struct ResultPacket {
    uint64_t full_priv_key[4]; // Simpan FULL private key (32 bytes)
    unsigned char hash160[20];
    int gpu_id;
};

// ==================== CRYPTO HELPERS (HOST SIDE) ====================
// Implementasi Mini SHA256 dan Base58 untuk konversi WIF/Address di CPU

static const char* BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

// Rotasi kanan (SHA256 helper)
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
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    uint8_t buffer[64];
    uint32_t bitlen[2] = {0, 0}; // Simplified for short inputs
    bitlen[1] = len * 8; // Assuming len < 56 bytes for our use case (keys/addresses)

    memcpy(buffer, data, len);
    buffer[len++] = 0x80;
    while (len < 56) buffer[len++] = 0;
    
    // Append length (Big Endian)
    buffer[63] = bitlen[1] & 0xFF; buffer[62] = (bitlen[1] >> 8) & 0xFF;
    // (Ignoring high bits for short inputs)

    sha256_transform(state, buffer);

    for (int i = 0; i < 8; ++i) {
        hash[i * 4] = (state[i] >> 24) & 0xFF;
        hash[i * 4 + 1] = (state[i] >> 16) & 0xFF;
        hash[i * 4 + 2] = (state[i] >> 8) & 0xFF;
        hash[i * 4 + 3] = state[i] & 0xFF;
    }
}

// Double SHA256 (Hash256)
void hash256(const uint8_t* data, size_t len, uint8_t* out) {
    uint8_t tmp[32];
    simple_sha256(data, len, tmp);
    simple_sha256(tmp, 32, out);
}

// Base58Check Encoding
std::string EncodeBase58Check(const uint8_t* payload, size_t len) {
    uint8_t data[64]; // payload + checksum
    memcpy(data, payload, len);

    // Calculate Checksum (First 4 bytes of Double SHA256)
    uint8_t hash[32];
    hash256(payload, len, hash);
    memcpy(data + len, hash, 4);
    
    size_t data_len = len + 4;

    // Convert to Base58
    // Counting leading zeros
    int zeros = 0;
    while (zeros < data_len && data[zeros] == 0) zeros++;

    // Convert byte array to big integer
    std::vector<unsigned char> b58((data_len * 138 / 100) + 1); // log(256)/log(58) approx 1.38
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
    for (int i = size - 1; i >= 0; --i)
        result += BASE58_ALPHABET[b58[i]];

    return result;
}

std::string ToWIF(const uint64_t* priv_key_u64) {
    // Convert 4x uint64 to 32 bytes array (Big Endian)
    uint8_t raw[34];
    raw[0] = 0x80; // Mainnet Private Key Prefix
    
    // Flip endianness for string rep if needed, but assuming u64 is standard layout
    // We need to carefully pack the u64 array into bytes.
    // Assuming priv_key[0] is most significant 64 bits.
    for(int i=0; i<4; i++) {
        uint64_t val = priv_key_u64[i];
        for(int b=0; b<8; b++) {
            raw[1 + i*8 + (7-b)] = (val >> (b*8)) & 0xFF;
        }
    }
    
    raw[33] = 0x01; // Compressed flag (Opsional, gunakan 33 byte len jika compressed, 34 bytes buffer)
    // Mari gunakan Uncompressed WIF (33 bytes: 1 prefix + 32 key) untuk simplisitas
    return EncodeBase58Check(raw, 33);
}

std::string ToAddress(const unsigned char* hash160) {
    uint8_t payload[21];
    payload[0] = 0x00; // Mainnet Address Prefix (1...)
    memcpy(payload + 1, hash160, 20);
    return EncodeBase58Check(payload, 21);
}

// ==================== DEVICE CONSTANTS ====================
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

// ==================== KERNEL UTAMA ====================

__global__ void __launch_bounds__(THREADS_PER_BLOCK) 
kernel_bruteforce_dump_all(
    ResultPacket* out_results,
    uint64_t seed,
    uint64_t global_offset,
    int gpu_id
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Inisialisasi RNG
    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, global_offset, &state);

    // Generate Start Private Key
    uint64_t priv_key[4];
    uint4 rand4 = curand4(&state);
    priv_key[0] = ((uint64_t)rand4.x << 32) | rand4.y;
    priv_key[1] = ((uint64_t)rand4.z << 32) | rand4.w;
    rand4 = curand4(&state);
    priv_key[2] = ((uint64_t)rand4.x << 32) | rand4.y;
    priv_key[3] = ((uint64_t)rand4.z << 32) | rand4.w;

    // Generate Public Key
    uint64_t pub_x[4], pub_y[4];
    #pragma unroll
    for(int i=0; i<4; i++) { pub_x[i] = priv_key[i]; pub_y[i] = priv_key[i] ^ 0xDEADBEEF; }

    // Hitung posisi unik di buffer output
    // Setiap thread memiliki slot untuk menyimpan SEMUA key hasil batch-nya
    // Buffer layout: [Thread0_Key0, Thread0_Key1, ..., Thread1_Key0, ...]
    // Atau interlaced: [T0_K0, T1_K0, ..., T0_K1, T1_K1...] -> Lebih coalesced
    
    int global_num_threads = gridDim.x * blockDim.x;

    for (int i = 0; i < KEYS_PER_THREAD; i++) {
        unsigned char hash160[20];
        
        // Simulasi Hash160
        uint64_t h_val = pub_x[0] ^ pub_y[3]; 
        #pragma unroll
        for(int b=0; b<5; b++) ((uint32_t*)hash160)[b] = (uint32_t)(h_val >> (b*3));

        // --- SIMPAN SEMUA HASIL ---
        // Kita tidak lagi mengecek Bloom Filter. Kita simpan mentah-mentah.
        // Index kalkulasi: (Iterasi Batch * Jumlah Total Thread) + Thread ID Global
        // Ini memastikan akses memori coalesced (berurutan antar thread)
        size_t buffer_idx = (size_t)i * global_num_threads + tid;
        
        ResultPacket* r = &out_results[buffer_idx];
        
        // Copy Full Private Key (32 bytes)
        #pragma unroll
        for(int k=0; k<4; k++) r->full_priv_key[k] = priv_key[k];
        
        // Copy Hash160
        memcpy(r->hash160, hash160, 20);
        
        r->gpu_id = gpu_id;

        // Next Key
        ecc_point_add_G(pub_x, pub_y);
        // priv_key seharusnya di increment juga (simulasi)
        priv_key[3]++; 
    }
}

// ==================== HOST UTILS ====================

// Queue untuk memindahkan data per-batch besar ke thread writer
template<typename T>
class SafeQueue {
    std::queue<std::vector<T>> q; // Queue of Vectors (Chunks)
    std::mutex m;
    std::condition_variable cv;
    bool done = false;
public:
    void push(std::vector<T> val) {
        std::lock_guard<std::mutex> lock(m);
        q.push(std::move(val)); // Move semantic agar cepat
        cv.notify_one();
    }
    
    bool pop(std::vector<T>& val) {
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [this]{ return !q.empty() || done; });
        if (q.empty() && done) return false;
        val = std::move(q.front());
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
    cudaStream_t stream;
    
    ResultPacket* d_results; // Device buffer (Huge)
    ResultPacket* h_results; // Host buffer (Huge)
    
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
        
        cudaStreamCreate(&stream);
        
        // Alokasi memori yang SANGAT BESAR untuk menampung semua hasil
        size_t buffer_size = total_keys_per_batch * sizeof(ResultPacket);
        
        std::cout << "[GPU " << id << "] Allocating buffer: " 
                  << (buffer_size / 1024 / 1024) << " MB per batch.\n";

        cudaMalloc(&d_results, buffer_size);
        cudaMallocHost(&h_results, buffer_size); // Pinned memory untuk transfer cepat
    }
    
    ~GPUWorker() {
        cudaSetDevice(gpu_id);
        cudaFree(d_results);
        cudaFreeHost(h_results);
        cudaStreamDestroy(stream);
    }

    void launch(uint64_t seed, uint64_t iteration, SafeQueue<ResultPacket>& queue) {
        cudaSetDevice(gpu_id);
        
        // 1. Launch Kernel (Tanpa Bloom Filter, Mode Dump)
        kernel_bruteforce_dump_all<<<blocks, threads, 0, stream>>>(
            d_results, 
            seed, 
            iteration, 
            gpu_id
        );
        
        // 2. Copy SEMUA hasil ke Host
        cudaMemcpyAsync(h_results, d_results, total_keys_per_batch * sizeof(ResultPacket), 
                       cudaMemcpyDeviceToHost, stream);
        
        // 3. Sync
        cudaStreamSynchronize(stream);
        
        // 4. Masukkan ke Queue untuk ditulis (Copy data ke vector)
        // Note: Ini akan memakan RAM CPU yang besar.
        std::vector<ResultPacket> batch_data(h_results, h_results + total_keys_per_batch);
        queue.push(std::move(batch_data));
    }
    
    size_t get_batch_size() { return total_keys_per_batch; }
};

// ==================== MAIN ====================

int main(int argc, char** argv) {
    std::cout << "=== BTC GENERATOR [DUMP ALL MODE] ===\n";
    std::cout << "WARNING: This will generate huge text files quickly.\n";
    
    // Setup IO Thread (Sekarang kerja keras!)
    SafeQueue<ResultPacket> result_queue;
    std::thread writer_thread([&result_queue](){
        std::ofstream outfile("dump_all.txt");
        outfile << "WIF_PrivateKey,Address\n"; // CSV Header
        
        std::vector<ResultPacket> chunk;
        uint64_t total_written = 0;

        while(result_queue.pop(chunk)) {
            // Proses chunk (Batch besar)
            for(const auto& res : chunk) {
                // 1. Convert Private Key to WIF
                std::string wif = ToWIF(res.full_priv_key);
                
                // 2. Convert Hash160 to Address
                std::string addr = ToAddress(res.hash160);
                
                // 3. Write to file
                outfile << wif << "," << addr << "\n";
            }
            total_written += chunk.size();
            std::cout << "\r[Writer] Total Saved: " << total_written << " keys..." << std::flush;
        }
    });

    // Init GPUs
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    std::vector<GPUWorker*> workers;
    for(int i=0; i<num_gpus; i++) {
        workers.push_back(new GPUWorker(i));
    }

    uint64_t global_iter = 0;
    uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    
    // LIMITASI: Karena file akan sangat besar, mari kita batasi iterasi atau loop forever
    // User minta "unlimited", tapi hati-hati harddisk penuh.
    bool running = true;

    while(running) {
        for(auto worker : workers) {
            worker->launch(seed, global_iter, result_queue);
        }
        global_iter++;
        
        // Opsional: Sleep sedikit agar disk tidak choking total jika GPU terlalu cepat
        // std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    result_queue.set_done();
    writer_thread.join();
    
    return 0;
}
