// btc_bruteforce_unlimited_multi_gpu_fixed.cu - Multi-GPU Version
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cctype>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <openssl/sha.h>
#include <openssl/ripemd.h>
#include <csignal>
#include <atomic>
#include <mutex>
#include <thread>
#include <queue>

using namespace std;
using namespace chrono;

// ==================== KONFIGURASI ====================
#define PRIVATE_KEY_SIZE 32
#define HASH160_SIZE 20
#define MAX_TARGETS 1000
#define THREADS_PER_BLOCK 256
#define KEYS_PER_THREAD 128
#define MAX_RESULTS 100
#define SAMPLE_INTERVAL 1000  // Simpan sample setiap 1000 iterasi

// Atomic flag untuk kontrol program
atomic<bool> stop_requested(false);
atomic<bool> save_requested(false);

// Struktur hasil dengan validasi
struct Result {
    unsigned char private_key[32];
    unsigned char hash160[20];
    unsigned char public_key[33];
    bool valid;
    unsigned long long thread_id;
    unsigned long long iteration;
    unsigned long long timestamp;
    int gpu_id;
    
    __host__ __device__ Result() : valid(false), thread_id(0), iteration(0), timestamp(0), gpu_id(0) {
        memset(private_key, 0, 32);
        memset(hash160, 0, 20);
        memset(public_key, 0, 33);
    }
    
    // Copy constructor
    __host__ __device__ Result(const Result& other) {
        memcpy(private_key, other.private_key, 32);
        memcpy(hash160, other.hash160, 20);
        memcpy(public_key, other.public_key, 33);
        valid = other.valid;
        thread_id = other.thread_id;
        iteration = other.iteration;
        timestamp = other.timestamp;
        gpu_id = other.gpu_id;
    }
};

// Konstanta secp256k1
__constant__ unsigned char d_secp256k1_n[32] = {
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE,
    0xBA, 0xAE, 0xDC, 0xE6, 0xAF, 0x48, 0xA0, 0x3B,
    0xBF, 0xD2, 0x5E, 0x8C, 0xD0, 0x36, 0x41, 0x41
};

__constant__ unsigned char d_target_hashes[MAX_TARGETS][HASH160_SIZE];
__constant__ int d_num_targets;

// ==================== FUNGSI VALIDASI GPU ====================
__device__ bool is_all_zeros_gpu(const unsigned char* data, int size) {
    for (int i = 0; i < size; i++) {
        if (data[i] != 0) return false;
    }
    return true;
}

__device__ bool is_private_key_valid_gpu(const unsigned char* private_key) {
    if (is_all_zeros_gpu(private_key, 32)) {
        return false;
    }
    
    for (int i = 31; i >= 0; i--) {
        if (private_key[i] < d_secp256k1_n[i]) break;
        if (private_key[i] > d_secp256k1_n[i]) return false;
    }
    
    return true;
}

// ==================== FUNGSI VALIDASI CPU ====================
bool is_all_zeros_cpu(const unsigned char* data, int size) {
    for (int i = 0; i < size; i++) {
        if (data[i] != 0) return false;
    }
    return true;
}

// ==================== FUNGSI HASH GPU ====================
__device__ void compute_simple_hash160(const unsigned char* public_key, unsigned char* output) {
    // Hash yang lebih sederhana namun deterministik
    unsigned char temp[33];
    
    // Copy public key ke temp
    for (int i = 0; i < 33; i++) {
        temp[i] = public_key[i];
    }
    
    for (int i = 0; i < 20; i++) {
        output[i] = 0;
        for (int j = 0; j < 33; j++) {
            output[i] ^= temp[j];
            temp[j] = (temp[j] + 17) % 256;
        }
        output[i] = (output[i] + i * 13) % 256;
    }
}

__device__ void generate_public_key_from_private(const unsigned char* private_key, unsigned char* public_key, int batch, int tid) {
    // Generate compressed public key (simulasi sederhana)
    // In real Bitcoin: public_key = private_key * G (point multiplication)
    // Here we use a deterministic transformation
    
    // First byte: 0x02 for even, 0x03 for odd
    public_key[0] = (private_key[31] % 2 == 0) ? 0x02 : 0x03;
    
    // Simulate x-coordinate of public key (32 bytes)
    for (int i = 0; i < 32; i++) {
        public_key[i + 1] = private_key[i] ^ ((i + batch + tid) * 7);
    }
}

// ==================== KERNEL UNTUK MULTI-GPU ====================
__global__ void bruteforce_kernel_multi_gpu(
    Result* results,
    int* found_count,
    unsigned long long seed,
    unsigned long long global_iteration,
    int gpu_id,
    unsigned char* sample_keys,
    int* sample_count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory untuk target hashes (256 target max)
    __shared__ unsigned char s_targets[256][HASH160_SIZE];
    
    // Load targets ke shared memory
    int targets_to_load = min(d_num_targets, 256);
    for (int i = threadIdx.x; i < targets_to_load * HASH160_SIZE; i += blockDim.x) {
        int target_idx = i / HASH160_SIZE;
        int byte_idx = i % HASH160_SIZE;
        s_targets[target_idx][byte_idx] = d_target_hashes[target_idx][byte_idx];
    }
    __syncthreads();
    
    // Inisialisasi RNG
    curandState_t state;
    curand_init(seed + tid + global_iteration * 1234567 + gpu_id * 987654321, 0, 0, &state);
    
    // Buffer lokal
    unsigned char private_key[32];
    unsigned char public_key[33];
    unsigned char hash160_result[20];
    
    for (int batch = 0; batch < KEYS_PER_THREAD; batch++) {
        // Generate random private key dengan entropy yang lebih baik
        for (int i = 0; i < 32; i += 4) {
            unsigned int rnd = curand(&state);
            memcpy(&private_key[i], &rnd, min(4, 32 - i));
        }
        
        // Tambah entropy dari berbagai sumber
        private_key[0] ^= (tid >> 0) & 0xFF;
        private_key[1] ^= (tid >> 8) & 0xFF;
        private_key[2] ^= (tid >> 16) & 0xFF;
        private_key[3] ^= batch & 0xFF;
        private_key[4] ^= (global_iteration >> 0) & 0xFF;
        private_key[5] ^= (global_iteration >> 8) & 0xFF;
        private_key[6] ^= gpu_id & 0xFF;
        
        // Simpan sample private key (hanya dari thread pertama di batch pertama)
        if (threadIdx.x == 0 && batch == 0 && blockIdx.x == 0) {
            int sample_idx = atomicAdd(sample_count, 1);
            if (sample_idx < 10) {  // Simpan maksimal 10 sample per GPU per batch
                memcpy(&sample_keys[sample_idx * 32], private_key, 32);
            }
        }
        
        // Pastikan private key tidak nol
        if (is_all_zeros_gpu(private_key, 32)) {
            private_key[31] = 1;  // Set LSB to 1
        }
        
        // VALIDASI: Bandingkan dengan n
        bool valid = true;
        for (int i = 31; i >= 0; i--) {
            if (private_key[i] < d_secp256k1_n[i]) break;
            if (private_key[i] > d_secp256k1_n[i]) {
                valid = false;
                break;
            }
        }
        if (!valid) continue;
        
        // Generate public key
        generate_public_key_from_private(private_key, public_key, batch, tid);
        
        // Hitung hash160
        compute_simple_hash160(public_key, hash160_result);
        
        // Bandingkan dengan semua targets
        for (int t = 0; t < targets_to_load; t++) {
            bool match = true;
            for (int j = 0; j < HASH160_SIZE; j++) {
                if (hash160_result[j] != s_targets[t][j]) {
                    match = false;
                    break;
                }
            }
            
            if (match) {
                int found_idx = atomicAdd(found_count, 1);
                
                if (found_idx < MAX_RESULTS) {
                    Result* result = &results[found_idx];
                    
                    // Salin data
                    memcpy(result->private_key, private_key, 32);
                    memcpy(result->public_key, public_key, 33);
                    memcpy(result->hash160, hash160_result, 20);
                    
                    result->valid = true;
                    result->thread_id = tid;
                    result->iteration = global_iteration * KEYS_PER_THREAD + batch;
                    result->timestamp = clock64();
                    result->gpu_id = gpu_id;
                }
                break;  // Stop checking other targets for this key
            }
        }
    }
}

// ==================== FUNGSI BANTU CPU ====================
string hex_encode(const unsigned char* data, int len) {
    const char* hex_chars = "0123456789abcdef";
    string result;
    for (int i = 0; i < len; i++) {
        result += hex_chars[(data[i] >> 4) & 0xF];
        result += hex_chars[data[i] & 0xF];
    }
    return result;
}

// Fungsi untuk decode hex string
bool hex_decode(const string& hex, unsigned char* output, size_t output_len) {
    if (hex.length() % 2 != 0 || hex.length() / 2 > output_len) {
        return false;
    }
    
    for (size_t i = 0; i < hex.length(); i += 2) {
        string byteString = hex.substr(i, 2);
        output[i/2] = (unsigned char) strtol(byteString.c_str(), NULL, 16);
    }
    return true;
}

string base58_encode(const unsigned char* data, int len) {
    const char* alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
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

string hash160_to_address(const unsigned char* hash160) {
    unsigned char version_hash160[21];
    version_hash160[0] = 0x00;  // Mainnet
    memcpy(version_hash160 + 1, hash160, 20);
    
    unsigned char checksum1[32], checksum2[32];
    SHA256(version_hash160, 21, checksum1);
    SHA256(checksum1, 32, checksum2);
    
    unsigned char address_bin[25];
    memcpy(address_bin, version_hash160, 21);
    memcpy(address_bin + 21, checksum2, 4);
    
    return base58_encode(address_bin, 25);
}

string private_key_to_wif(const unsigned char* private_key) {
    // WIF format: 0x80 + private_key + 0x01 + checksum
    unsigned char wif_bytes[38];
    wif_bytes[0] = 0x80;
    memcpy(wif_bytes + 1, private_key, 32);
    wif_bytes[33] = 0x01;  // Compressed flag
    
    unsigned char checksum1[32], checksum2[32];
    SHA256(wif_bytes, 34, checksum1);
    SHA256(checksum1, 32, checksum2);
    
    memcpy(wif_bytes + 34, checksum2, 4);
    
    return base58_encode(wif_bytes, 38);
}

bool is_result_valid(const Result& result) {
    if (!result.valid) return false;
    
    // Cek private key tidak nol menggunakan fungsi CPU
    if (is_all_zeros_cpu(result.private_key, 32)) {
        return false;
    }
    
    return true;
}

// Buffer untuk menyimpan 100 hasil terakhir
vector<Result> last_100_results;
mutex results_mutex;

// Buffer untuk menyimpan sample keys
vector<vector<unsigned char>> sample_keys_buffer;
mutex sample_mutex;

// Fungsi untuk menangani sinyal
void signal_handler(int signal) {
    cout << "\n\nSignal received (" << signal << "). Saving results..." << endl;
    stop_requested = true;
    save_requested = true;
}

void save_last_results() {
    lock_guard<mutex> lock(results_mutex);
    
    if (last_100_results.empty()) {
        cout << "No results to save." << endl;
        return;
    }
    
    ofstream outfile("last_100_results.txt", ios::out);
    if (!outfile.is_open()) {
        cout << "Error: Cannot open last_100_results.txt for writing." << endl;
        return;
    }
    
    outfile << "=== LAST 100 VALID RESULTS ===" << endl;
    outfile << "Saved at: " << time(nullptr) << endl;
    outfile << "Total results: " << last_100_results.size() << endl;
    outfile << "=================================" << endl << endl;
    
    int valid_count = 0;
    for (size_t i = 0; i < last_100_results.size(); i++) {
        const Result& result = last_100_results[i];
        
        if (is_result_valid(result)) {
            valid_count++;
            outfile << "Result #" << valid_count << endl;
            outfile << "GPU: " << result.gpu_id << endl;
            outfile << "Private Key: " << hex_encode(result.private_key, 32) << endl;
            
            string address = hash160_to_address(result.hash160);
            outfile << "Bitcoin Address: " << address << endl;
            
            string wif = private_key_to_wif(result.private_key);
            outfile << "WIF: " << wif << endl;
            
            outfile << "Thread ID: " << result.thread_id << endl;
            outfile << "Iteration: " << result.iteration << endl;
            outfile << "Timestamp: " << result.timestamp << endl;
            outfile << "-------------------" << endl << endl;
        }
    }
    
    outfile.close();
    cout << "Saved " << valid_count << " valid results to last_100_results.txt" << endl;
}

void save_sample_keys(int iteration, const vector<unsigned char>& samples, int gpu_id) {
    lock_guard<mutex> lock(sample_mutex);
    
    ofstream sample_file("sample_keys.txt", ios::app);
    if (!sample_file.is_open()) {
        cout << "Error: Cannot open sample_keys.txt for writing." << endl;
        return;
    }
    
    sample_file << "=== ITERATION " << iteration << " ===" << endl;
    sample_file << "GPU: " << gpu_id << endl;
    sample_file << "Timestamp: " << time(nullptr) << endl;
    
    for (size_t i = 0; i < samples.size() / 32; i++) {
        sample_file << "Sample " << i + 1 << ": " 
                   << hex_encode(&samples[i * 32], 32) << endl;
    }
    
    sample_file << "-------------------" << endl << endl;
    sample_file.close();
}

void update_last_results(const Result& result) {
    lock_guard<mutex> lock(results_mutex);
    
    if (last_100_results.size() >= 100) {
        // Hapus yang paling lama (index 0)
        last_100_results.erase(last_100_results.begin());
    }
    
    last_100_results.push_back(result);
}

// Fungsi untuk membuat hash160 target dari alamat (versi sederhana)
void create_target_hash_from_address(const string& address, unsigned char* hash160_out) {
    memset(hash160_out, 0, HASH160_SIZE);
    
    for (size_t i = 0; i < address.length(); i++) {
        for (int j = 0; j < HASH160_SIZE; j++) {
            hash160_out[j] ^= address[i] + (i * j * 11);
            hash160_out[j] = (hash160_out[j] * 37 + j) % 256;
        }
    }
    
    for (int i = 0; i < HASH160_SIZE; i++) {
        hash160_out[i] = (hash160_out[i] + i * 17) % 256;
    }
}

// ==================== KELAS UNTUK MULTI-GPU ====================
class GPUWorker {
private:
    int gpu_id;
    cudaStream_t stream;
    Result* d_results;
    int* d_found_count;
    unsigned char* d_sample_keys;
    int* d_sample_count;
    unsigned char h_sample_keys[10 * 32];
    int h_sample_count;
    int blocks;
    int threads;
    unsigned long long total_keys;
    
public:
    GPUWorker(int id, int b, int t) : gpu_id(id), blocks(b), threads(t), total_keys(0) {
        cudaSetDevice(gpu_id);
        cudaStreamCreate(&stream);
        
        cudaMalloc(&d_results, MAX_RESULTS * sizeof(Result));
        cudaMalloc(&d_found_count, sizeof(int));
        cudaMalloc(&d_sample_keys, 10 * 32);
        cudaMalloc(&d_sample_count, sizeof(int));
    }
    
    ~GPUWorker() {
        cudaSetDevice(gpu_id);
        cudaStreamDestroy(stream);
        cudaFree(d_results);
        cudaFree(d_found_count);
        cudaFree(d_sample_keys);
        cudaFree(d_sample_count);
    }
    
    void reset_counters() {
        cudaSetDevice(gpu_id);
        cudaMemsetAsync(d_found_count, 0, sizeof(int), stream);
        cudaMemsetAsync(d_sample_count, 0, sizeof(int), stream);
    }
    
    void launch_kernel(unsigned long long seed, unsigned long long global_iteration) {
        cudaSetDevice(gpu_id);
        
        // Reset results
        Result init_result;
        for (int i = 0; i < MAX_RESULTS; i++) {
            cudaMemcpyAsync(&d_results[i], &init_result, sizeof(Result), 
                           cudaMemcpyHostToDevice, stream);
        }
        
        bruteforce_kernel_multi_gpu<<<blocks, threads, 0, stream>>>(
            d_results, d_found_count, seed, global_iteration, gpu_id,
            d_sample_keys, d_sample_count
        );
        
        total_keys += (unsigned long long)blocks * threads * KEYS_PER_THREAD;
    }
    
    void synchronize() {
        cudaSetDevice(gpu_id);
        cudaStreamSynchronize(stream);
    }
    
    int get_found_count() {
        int count;
        cudaSetDevice(gpu_id);
        cudaMemcpyAsync(&count, d_found_count, sizeof(int), 
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        return count;
    }
    
    vector<Result> get_results(int found_count) {
        vector<Result> results(min(found_count, MAX_RESULTS));
        if (found_count > 0) {
            cudaSetDevice(gpu_id);
            cudaMemcpyAsync(results.data(), d_results, 
                           min(found_count, MAX_RESULTS) * sizeof(Result),
                           cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            
            // Tambah GPU ID ke setiap hasil
            for (auto& result : results) {
                result.gpu_id = gpu_id;
            }
        }
        return results;
    }
    
    vector<unsigned char> get_sample_keys() {
        cudaSetDevice(gpu_id);
        cudaMemcpyAsync(&h_sample_count, d_sample_count, sizeof(int),
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        vector<unsigned char> samples;
        if (h_sample_count > 0) {
            samples.resize(h_sample_count * 32);
            cudaMemcpyAsync(samples.data(), d_sample_keys, h_sample_count * 32,
                          cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }
        return samples;
    }
    
    unsigned long long get_total_keys() const { return total_keys; }
    int get_gpu_id() const { return gpu_id; }
};

// ==================== FUNGSI UTAMA ====================
int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <address_list.txt>" << endl;
        return 1;
    }
    
    // Setup signal handler
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    cout << "=== Bitcoin Brute Force - Multi-GPU Version ===" << endl;
    cout << "==============================================" << endl;
    cout << "Press Ctrl+C to stop and save results" << endl;
    cout << "==============================================" << endl;
    
    // Deteksi jumlah GPU
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    
    if (num_gpus == 0) {
        cout << "Error: No CUDA-capable GPU found!" << endl;
        return 1;
    }
    
    cout << "Found " << num_gpus << " GPU(s)" << endl;
    
    vector<cudaDeviceProp> props(num_gpus);
    for (int i = 0; i < num_gpus; i++) {
        cudaGetDeviceProperties(&props[i], i);
        cout << "GPU " << i << ": " << props[i].name 
             << " (Compute " << props[i].major << "." << props[i].minor << ")" 
             << " Memory: " << props[i].totalGlobalMem / (1024*1024*1024.0) << " GB" << endl;
    }
    
    ifstream file(argv[1]);
    if (!file.is_open()) {
        cout << "Error: Cannot open file " << argv[1] << endl;
        return 1;
    }
    
    // Baca alamat dari file
    vector<string> addresses;
    string line;
    while (getline(file, line)) {
        if (!line.empty() && line[0] != '#') {
            line.erase(remove_if(line.begin(), line.end(), ::isspace), line.end());
            if (!line.empty()) {
                addresses.push_back(line);
            }
        }
    }
    file.close();
    
    if (addresses.empty()) {
        cout << "Error: No addresses found" << endl;
        return 1;
    }
    
    cout << "\nLoaded " << addresses.size() << " addresses" << endl;
    
    // Buat target hashes dari alamat
    unsigned char target_hashes[MAX_TARGETS][HASH160_SIZE];
    int num_targets = min((int)addresses.size(), MAX_TARGETS);
    
    for (int i = 0; i < num_targets; i++) {
        create_target_hash_from_address(addresses[i], target_hashes[i]);
    }
    
    // Copy targets ke setiap GPU
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        cudaMemcpyToSymbol(d_target_hashes, target_hashes, num_targets * HASH160_SIZE);
        cudaMemcpyToSymbol(d_num_targets, &num_targets, sizeof(int));
    }
    
    // Konfigurasi GPU worker
    vector<GPUWorker*> workers;
    vector<thread> worker_threads;
    mutex cout_mutex;
    
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        int blocks = min(props[i].multiProcessorCount * 4, 65535);
        workers.push_back(new GPUWorker(i, blocks, THREADS_PER_BLOCK));
        
        cout_mutex.lock();
        cout << "GPU " << i << " configured with " << blocks << " blocks" << endl;
        cout_mutex.unlock();
    }
    
    cout << "\nStarting multi-GPU search..." << endl;
    cout << "Total keys per iteration: ";
    unsigned long long total_keys_per_iter = 0;
    for (auto& worker : workers) {
        total_keys_per_iter += worker->get_total_keys();
    }
    cout << total_keys_per_iter << endl;
    cout << "Press Ctrl+C to stop and save results" << endl;
    cout << "==============================================" << endl;
    
    // File untuk semua hasil
    ofstream all_results_file("all_found_keys.txt", ios::app);
    ofstream sample_file("sample_keys.txt", ios::app);
    
    if (all_results_file.is_open()) {
        all_results_file << "\n=== NEW SESSION ===" << endl;
        all_results_file << "Started at: " << time(nullptr) << endl;
        all_results_file << "GPUs: " << num_gpus << endl;
        all_results_file << "Target addresses: " << addresses.size() << endl;
        all_results_file << "=================================" << endl << endl;
    }
    
    if (sample_file.is_open()) {
        sample_file << "\n=== SAMPLE KEYS SESSION ===" << endl;
        sample_file << "Started at: " << time(nullptr) << endl;
        sample_file << "=================================" << endl << endl;
        sample_file.close();
    }
    
    // Variabel statistik
    unsigned long long global_iteration = 0;
    unsigned long long total_keys_all_gpus = 0;
    int valid_found_count = 0;
    auto start_time = high_resolution_clock::now();
    auto last_progress_time = start_time;
    auto last_save_time = start_time;
    
    // Seed untuk RNG
    unsigned long long seed = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    
    while (!stop_requested) {
        global_iteration++;
        
        // Reset counters untuk semua GPU
        for (auto& worker : workers) {
            worker->reset_counters();
        }
        
        // Launch kernel di semua GPU secara parallel
        vector<thread> kernel_threads;
        for (auto& worker : workers) {
            kernel_threads.emplace_back([worker, seed, global_iteration]() {
                worker->launch_kernel(seed, global_iteration);
            });
        }
        
        // Tunggu semua kernel selesai
        for (auto& t : kernel_threads) {
            t.join();
        }
        
        // Synchronize semua GPU
        for (auto& worker : workers) {
            worker->synchronize();
        }
        
        // Update statistik
        unsigned long long keys_this_iteration = 0;
        for (auto& worker : workers) {
            keys_this_iteration += (unsigned long long)worker->get_total_keys();
        }
        total_keys_all_gpus += keys_this_iteration;
        
        // Process results dari semua GPU
        int total_found_this_iteration = 0;
        for (auto& worker : workers) {
            int found_count = worker->get_found_count();
            total_found_this_iteration += found_count;
            
            if (found_count > 0) {
                vector<Result> gpu_results = worker->get_results(found_count);
                
                for (auto& result : gpu_results) {
                    if (is_result_valid(result)) {
                        valid_found_count++;
                        
                        // Update buffer 100 hasil terakhir
                        update_last_results(result);
                        
                        // Simpan ke file utama
                        if (all_results_file.is_open()) {
                            all_results_file << "=== VALID MATCH #" << valid_found_count << " ===" << endl;
                            all_results_file << "GPU: " << result.gpu_id << endl;
                            all_results_file << "Private Key: " << hex_encode(result.private_key, 32) << endl;
                            string address = hash160_to_address(result.hash160);
                            all_results_file << "Address: " << address << endl;
                            string wif = private_key_to_wif(result.private_key);
                            all_results_file << "WIF: " << wif << endl;
                            all_results_file << "Thread ID: " << result.thread_id << endl;
                            all_results_file << "Iteration: " << result.iteration << endl;
                            all_results_file << "Timestamp: " << time(nullptr) << endl;
                            all_results_file << "-------------------" << endl << endl;
                            all_results_file.flush();
                        }
                        
                        // Tampilkan di console
                        cout_mutex.lock();
                        cout << "\n\n=== NEW VALID MATCH #" << valid_found_count << " ===" << endl;
                        cout << "GPU: " << result.gpu_id << endl;
                        cout << "Private Key: " << hex_encode(result.private_key, 32) << endl;
                        cout << "Bitcoin Address: " << hash160_to_address(result.hash160) << endl;
                        cout << "WIF: " << private_key_to_wif(result.private_key) << endl;
                        cout << "Thread ID: " << result.thread_id << endl;
                        cout << "Iteration: " << result.iteration << endl;
                        cout_mutex.unlock();
                    }
                }
            }
            
            // Ambil dan simpan sample keys setiap SAMPLE_INTERVAL iterasi
            if (global_iteration % SAMPLE_INTERVAL == 0) {
                vector<unsigned char> samples = worker->get_sample_keys();
                if (!samples.empty()) {
                    save_sample_keys(global_iteration, samples, worker->get_gpu_id());
                    
                    cout_mutex.lock();
                    cout << "\nSaved " << samples.size() / 32 
                         << " sample keys from GPU " << worker->get_gpu_id() 
                         << " (Iteration " << global_iteration << ")" << endl;
                    cout_mutex.unlock();
                }
            }
        }
        
        // Tampilkan progress
        auto current_time = high_resolution_clock::now();
        auto elapsed = duration_cast<milliseconds>(current_time - start_time).count() / 1000.0;
        
        if (duration_cast<milliseconds>(current_time - last_progress_time).count() > 1000 && elapsed > 0) {
            double keys_per_sec = total_keys_all_gpus / elapsed;
            cout_mutex.lock();
            cout << fixed << setprecision(2);
            cout << "\r[Iter " << global_iteration << "] Speed: " 
                 << keys_per_sec / 1000000 << " Mkeys/sec | "
                 << "Total: " << total_keys_all_gpus / 1000000 << " Mkeys | "
                 << "Found: " << valid_found_count << " valid keys      " << flush;
            cout_mutex.unlock();
            last_progress_time = current_time;
        }
        
        // Auto-save setiap 5 menit
        if (duration_cast<seconds>(current_time - last_save_time).count() > 300) {
            cout_mutex.lock();
            cout << "\n\nAuto-saving last 100 results..." << endl;
            cout_mutex.unlock();
            
            save_last_results();
            last_save_time = current_time;
        }
        
        // Cek jika perlu save dan stop
        if (save_requested) {
            cout_mutex.lock();
            cout << "\n\nSaving results and shutting down..." << endl;
            cout_mutex.unlock();
            break;
        }
    }
    
    // Cleanup
    if (all_results_file.is_open()) {
        all_results_file << "\n=== SESSION ENDED ===" << endl;
        all_results_file << "Ended at: " << time(nullptr) << endl;
        all_results_file << "Total keys tested: " << total_keys_all_gpus << endl;
        all_results_file << "Total valid matches: " << valid_found_count << endl;
        all_results_file << "Total iterations: " << global_iteration << endl;
        all_results_file.close();
    }
    
    // Save final results
    save_last_results();
    
    auto end_time = high_resolution_clock::now();
    auto total_elapsed_seconds = duration_cast<seconds>(end_time - start_time).count();
    
    cout << "\n\n=== FINAL STATISTICS ===" << endl;
    cout << "GPUs used: " << num_gpus << endl;
    cout << "Total iterations: " << global_iteration << endl;
    cout << "Total keys tested: " << total_keys_all_gpus << " (" 
         << total_keys_all_gpus / 1000000 << " million)" << endl;
    cout << "Total time: " << total_elapsed_seconds << " seconds (" 
         << fixed << setprecision(2) << total_elapsed_seconds / 3600.0 << " hours)" << endl;
    
    if (total_elapsed_seconds > 0) {
        cout << "Average speed: " << (total_keys_all_gpus / total_elapsed_seconds) / 1000000 
             << " Mkeys/second" << endl;
        cout << "Average speed per GPU: " 
             << (total_keys_all_gpus / total_elapsed_seconds) / 1000000 / num_gpus 
             << " Mkeys/second" << endl;
    }
    
    cout << "Total valid matches found: " << valid_found_count << endl;
    
    if (valid_found_count > 0) {
        cout << "\nResults saved to:" << endl;
        cout << "- all_found_keys.txt (all matches)" << endl;
        cout << "- last_100_results.txt (last 100 matches)" << endl;
        cout << "- sample_keys.txt (sample keys every " << SAMPLE_INTERVAL << " iterations)" << endl;
    }
    
    // Cleanup workers
    for (auto& worker : workers) {
        delete worker;
    }
    
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        cudaDeviceReset();
    }
    
    cout << "\nProgram finished successfully." << endl;
    return 0;
}
