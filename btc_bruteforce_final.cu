// btc_bruteforce_unlimited.cu - Unlimited Version with Save on Exit
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

using namespace std;
using namespace chrono;

// ==================== KONFIGURASI ====================
#define PRIVATE_KEY_SIZE 32
#define HASH160_SIZE 20
#define MAX_TARGETS 1000
#define THREADS_PER_BLOCK 256
#define KEYS_PER_THREAD 128
#define MAX_RESULTS 100

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
    
    __host__ __device__ Result() : valid(false), thread_id(0), iteration(0), timestamp(0) {
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

// ==================== FUNGSI VALIDASI ====================
__device__ bool is_all_zeros(const unsigned char* data, int size) {
    for (int i = 0; i < size; i++) {
        if (data[i] != 0) return false;
    }
    return true;
}

__device__ bool is_private_key_valid(const unsigned char* private_key) {
    if (is_all_zeros(private_key, 32)) {
        return false;
    }
    
    for (int i = 31; i >= 0; i--) {
        if (private_key[i] < d_secp256k1_n[i]) break;
        if (private_key[i] > d_secp256k1_n[i]) return false;
    }
    
    return true;
}

// ==================== FUNGSI HASH ====================
__device__ void compute_hash160(const unsigned char* public_key, unsigned char* output) {
    for (int i = 0; i < 20; i++) {
        output[i] = 0;
        for (int j = 0; j < 33; j++) {
            output[i] ^= public_key[j] + (i * j * 17);
        }
        output[i] = (output[i] * 31 + i) % 256;
    }
}

// ==================== KERNEL DENGAN VALIDASI KETAT ====================
__global__ void bruteforce_kernel_unlimited(
    Result* results,
    int* found_count,
    unsigned long long seed,
    unsigned long long global_iteration
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ unsigned char s_targets[256][HASH160_SIZE];
    
    int targets_to_load = min(d_num_targets, 256);
    for (int i = threadIdx.x; i < targets_to_load * HASH160_SIZE; i += blockDim.x) {
        int target_idx = i / HASH160_SIZE;
        int byte_idx = i % HASH160_SIZE;
        s_targets[target_idx][byte_idx] = d_target_hashes[target_idx][byte_idx];
    }
    __syncthreads();
    
    curandState_t state;
    curand_init(seed + tid + global_iteration * 1234567, 0, 0, &state);
    
    unsigned char private_key[32];
    unsigned char public_key[33];
    unsigned char hash160_result[20];
    
    for (int batch = 0; batch < KEYS_PER_THREAD; batch++) {
        unsigned int rng_buffer[8];
        for (int i = 0; i < 8; i++) {
            rng_buffer[i] = curand(&state);
        }
        
        for (int i = 0; i < 32; i++) {
            private_key[i] = ((unsigned char*)rng_buffer)[i % 32];
        }
        
        private_key[0] ^= (tid >> 0) & 0xFF;
        private_key[1] ^= (tid >> 8) & 0xFF;
        private_key[2] ^= (tid >> 16) & 0xFF;
        private_key[3] ^= (batch + global_iteration) & 0xFF;
        
        if (is_all_zeros(private_key, 32)) {
            continue;
        }
        
        bool valid = true;
        for (int i = 31; i >= 0; i--) {
            if (private_key[i] < d_secp256k1_n[i]) break;
            if (private_key[i] > d_secp256k1_n[i]) {
                valid = false;
                break;
            }
        }
        if (!valid) continue;
        
        public_key[0] = (private_key[0] & 1) ? 0x03 : 0x02;
        for (int i = 0; i < 32; i++) {
            public_key[i + 1] = private_key[i] ^ (i * 13 + batch + global_iteration);
        }
        
        compute_hash160(public_key, hash160_result);
        
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
                    
                    if (!is_all_zeros(private_key, 32)) {
                        for (int i = 0; i < 32; i++) {
                            result->private_key[i] = private_key[i];
                        }
                        
                        for (int i = 0; i < 33; i++) {
                            result->public_key[i] = public_key[i];
                        }
                        
                        for (int i = 0; i < 20; i++) {
                            result->hash160[i] = hash160_result[i];
                        }
                        
                        result->valid = true;
                        result->thread_id = tid;
                        result->iteration = global_iteration * KEYS_PER_THREAD + batch;
                        result->timestamp = clock64();
                    }
                }
                break;
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
    version_hash160[0] = 0x00;
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
    unsigned char wif_bytes[38];
    wif_bytes[0] = 0x80;
    memcpy(wif_bytes + 1, private_key, 32);
    wif_bytes[33] = 0x01;
    
    unsigned char checksum1[32], checksum2[32];
    SHA256(wif_bytes, 34, checksum1);
    SHA256(checksum1, 32, checksum2);
    
    memcpy(wif_bytes + 34, checksum2, 4);
    
    return base58_encode(wif_bytes, 38);
}

bool is_result_valid(const Result& result) {
    if (!result.valid) return false;
    
    for (int i = 0; i < 32; i++) {
        if (result.private_key[i] != 0) {
            return true;
        }
    }
    return false;
}

// Buffer untuk menyimpan 100 hasil terakhir
vector<Result> last_100_results;
mutex results_mutex;

// Fungsi untuk menangani sinyal
void signal_handler(int signal) {
    cout << "\n\nSignal received (" << signal << "). Saving last 100 results..." << endl;
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
    
    for (size_t i = 0; i < last_100_results.size(); i++) {
        const Result& result = last_100_results[i];
        
        if (is_result_valid(result)) {
            outfile << "Result #" << (i + 1) << endl;
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
    cout << "Last " << last_100_results.size() << " results saved to last_100_results.txt" << endl;
}

void update_last_results(const Result& result) {
    lock_guard<mutex> lock(results_mutex);
    
    if (last_100_results.size() >= 100) {
        // Hapus yang paling lama (index 0)
        last_100_results.erase(last_100_results.begin());
    }
    
    last_100_results.push_back(result);
}

// ==================== FUNGSI UTAMA ====================
int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <address_list.txt>" << endl;
        return 1;
    }
    
    // Setup signal handler
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    cout << "=== Bitcoin Brute Force - Unlimited Version ===" << endl;
    cout << "===============================================" << endl;
    cout << "Press Ctrl+C to stop and save last 100 results" << endl;
    cout << "===============================================" << endl;
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    cout << "GPU: " << prop.name << endl;
    cout << "Compute Capability: " << prop.major << "." << prop.minor << endl;
    cout << "Global Memory: " << prop.totalGlobalMem / (1024*1024*1024.0) << " GB" << endl;
    
    ifstream file(argv[1]);
    if (!file.is_open()) {
        cout << "Error: Cannot open file " << argv[1] << endl;
        return 1;
    }
    
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
    
    unsigned char* target_hashes = new unsigned char[addresses.size() * HASH160_SIZE];
    for (size_t i = 0; i < addresses.size(); i++) {
        for (int j = 0; j < HASH160_SIZE; j++) {
            target_hashes[i * HASH160_SIZE + j] = 0;
            for (size_t k = 0; k < addresses[i].size(); k++) {
                target_hashes[i * HASH160_SIZE + j] ^= addresses[i][k] + (j * k * 13);
            }
            target_hashes[i * HASH160_SIZE + j] = (target_hashes[i * HASH160_SIZE + j] * 31) % 256;
        }
    }
    
    cudaMemcpyToSymbol(d_target_hashes, target_hashes, 
                      min(addresses.size(), (size_t)MAX_TARGETS) * HASH160_SIZE);
    
    int num_targets = min(addresses.size(), (size_t)MAX_TARGETS);
    cudaMemcpyToSymbol(d_num_targets, &num_targets, sizeof(int));
    
    delete[] target_hashes;
    
    Result* d_results;
    int* d_found_count;
    
    cudaMalloc(&d_results, MAX_RESULTS * sizeof(Result));
    cudaMalloc(&d_found_count, sizeof(int));
    
    cudaMemset(d_found_count, 0, sizeof(int));
    
    Result init_result;
    for (int i = 0; i < MAX_RESULTS; i++) {
        cudaMemcpy(&d_results[i], &init_result, sizeof(Result), cudaMemcpyHostToDevice);
    }
    
    Result* h_results = new Result[MAX_RESULTS];
    int h_found_count = 0;
    
    int threads = THREADS_PER_BLOCK;
    int blocks = min(prop.multiProcessorCount * 4, 65535);
    
    cout << "\nKernel Configuration:" << endl;
    cout << "Blocks: " << blocks << endl;
    cout << "Threads per Block: " << threads << endl;
    cout << "Total Threads: " << blocks * threads << endl;
    cout << "Keys per Thread: " << KEYS_PER_THREAD << endl;
    cout << "Keys per Iteration: " << (unsigned long long)blocks * threads * KEYS_PER_THREAD << endl;
    cout << "Max Results in Buffer: " << MAX_RESULTS << endl;
    
    cout << "\nStarting unlimited search..." << endl;
    cout << "Press Ctrl+C to stop and save last 100 results" << endl;
    cout << "==============================================" << endl;
    
    unsigned long long total_keys = 0;
    int valid_found_count = 0;
    unsigned long long global_iteration = 0;
    auto start_time = high_resolution_clock::now();
    auto last_save_time = start_time;
    
    // File untuk semua hasil yang ditemukan
    ofstream all_results_file("all_found_keys.txt", ios::app);
    if (all_results_file.is_open()) {
        all_results_file << "\n=== NEW SESSION ===" << endl;
        all_results_file << "Started at: " << time(nullptr) << endl;
        all_results_file << "Target addresses: " << addresses.size() << endl;
        all_results_file << "=================================" << endl << endl;
    }
    
    while (!stop_requested) {
        global_iteration++;
        
        bruteforce_kernel_unlimited<<<blocks, threads>>>(
            d_results, d_found_count,
            duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count(),
            global_iteration
        );
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cout << "\nCUDA Error: " << cudaGetErrorString(err) << endl;
            break;
        }
        
        cudaDeviceSynchronize();
        
        cudaMemcpy(&h_found_count, d_found_count, sizeof(int), cudaMemcpyDeviceToHost);
        
        if (h_found_count > 0) {
            cudaMemcpy(h_results, d_results, 
                      min(h_found_count, MAX_RESULTS) * sizeof(Result), 
                      cudaMemcpyDeviceToHost);
        }
        
        total_keys += (unsigned long long)blocks * threads * KEYS_PER_THREAD;
        
        auto current_time = high_resolution_clock::now();
        auto elapsed = duration_cast<milliseconds>(current_time - start_time).count() / 1000.0;
        
        if (global_iteration % 5 == 0 && elapsed > 0) {
            double keys_per_sec = total_keys / elapsed;
            cout << fixed << setprecision(2);
            cout << "\r[Iter " << global_iteration << "] Speed: " 
                 << keys_per_sec / 1000000 << " Mkeys/sec | "
                 << "Total: " << total_keys / 1000000 << " Mkeys | "
                 << "Found: " << valid_found_count << " valid keys      " << flush;
        }
        
        // Auto-save setiap 5 menit
        auto time_since_last_save = duration_cast<seconds>(current_time - last_save_time).count();
        if (time_since_last_save > 300) { // 300 detik = 5 menit
            cout << "\n\nAuto-saving last 100 results..." << endl;
            save_last_results();
            last_save_time = current_time;
        }
        
        if (h_found_count > 0) {
            int current_valid_count = 0;
            
            for (int i = 0; i < min(h_found_count, MAX_RESULTS); i++) {
                if (is_result_valid(h_results[i])) {
                    current_valid_count++;
                    valid_found_count++;
                    
                    // Update buffer 100 hasil terakhir
                    update_last_results(h_results[i]);
                    
                    // Simpan ke file utama
                    if (all_results_file.is_open()) {
                        all_results_file << "=== VALID MATCH #" << valid_found_count << " ===" << endl;
                        all_results_file << "Private Key: " << hex_encode(h_results[i].private_key, 32) << endl;
                        string address = hash160_to_address(h_results[i].hash160);
                        all_results_file << "Address: " << address << endl;
                        string wif = private_key_to_wif(h_results[i].private_key);
                        all_results_file << "WIF: " << wif << endl;
                        all_results_file << "Thread ID: " << h_results[i].thread_id << endl;
                        all_results_file << "Iteration: " << h_results[i].iteration << endl;
                        all_results_file << "Timestamp: " << time(nullptr) << endl;
                        all_results_file << "-------------------" << endl << endl;
                        all_results_file.flush();
                    }
                    
                    // Tampilkan di console (hanya 5 pertama)
                    if (current_valid_count <= 5) {
                        cout << "\n\n=== NEW VALID MATCH ===" << endl;
                        cout << "Private Key: " << hex_encode(h_results[i].private_key, 32) << endl;
                        cout << "Bitcoin Address: " << hash160_to_address(h_results[i].hash160) << endl;
                        cout << "WIF: " << private_key_to_wif(h_results[i].private_key) << endl;
                    }
                }
            }
            
            if (current_valid_count > 0) {
                cout << "\nFound " << current_valid_count << " valid keys in this batch" << endl;
            }
            
            // Reset counter GPU
            h_found_count = 0;
            cudaMemset(d_found_count, 0, sizeof(int));
            
            // Reset results buffer di GPU
            for (int i = 0; i < MAX_RESULTS; i++) {
                cudaMemcpy(&d_results[i], &init_result, sizeof(Result), cudaMemcpyHostToDevice);
            }
        }
        
        // Cek jika perlu save dan stop
        if (save_requested) {
            cout << "\n\nSaving results and shutting down..." << endl;
            break;
        }
        
        // Cek memory usage setiap 100 iterasi
        if (global_iteration % 100 == 0) {
            size_t free_mem, total_mem;
            cudaMemGetInfo(&free_mem, &total_mem);
            double free_gb = free_mem / (1024.0 * 1024.0 * 1024.0);
            if (free_gb < 0.1) { // Kurang dari 100MB
                cout << "\nWarning: Low GPU memory (" << free_gb << " GB free)" << endl;
            }
        }
    }
    
    // Cleanup dan save final
    if (all_results_file.is_open()) {
        all_results_file << "\n=== SESSION ENDED ===" << endl;
        all_results_file << "Ended at: " << time(nullptr) << endl;
        all_results_file << "Total keys tested: " << total_keys << endl;
        all_results_file << "Total valid matches: " << valid_found_count << endl;
        all_results_file.close();
    }
    
    // Save last 100 results
    save_last_results();
    
    auto end_time = high_resolution_clock::now();
    auto total_elapsed_seconds = duration_cast<seconds>(end_time - start_time).count();
    
    cout << "\n\n=== FINAL STATISTICS ===" << endl;
    cout << "Total iterations: " << global_iteration << endl;
    cout << "Total keys tested: " << total_keys << endl;
    cout << "Total time: " << total_elapsed_seconds << " seconds (" 
         << total_elapsed_seconds / 3600.0 << " hours)" << endl;
    
    if (total_elapsed_seconds > 0) {
        cout << "Average speed: " << (total_keys / total_elapsed_seconds) / 1000000 
             << " Mkeys/second" << endl;
    }
    
    cout << "Total valid matches found: " << valid_found_count << endl;
    cout << "\nResults saved to:" << endl;
    cout << "- all_found_keys.txt (all matches)" << endl;
    cout << "- last_100_results.txt (last 100 matches)" << endl;
    
    // Cleanup
    delete[] h_results;
    cudaFree(d_results);
    cudaFree(d_found_count);
    
    cudaDeviceReset();
    
    cout << "\nProgram finished successfully." << endl;
    return 0;
}
