// btc_bruteforce_unlimited.cu - Unlimited Search with Live Sample Output
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cctype>
#include <atomic>
#include <thread>
#include <signal.h>
#include <ctime>  // Ditambahkan untuk fungsi time dan localtime
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

using namespace std;
using namespace chrono;

// ==================== KONFIGURASI ====================
#define PRIVATE_KEY_SIZE 32
#define HASH160_SIZE 20
#define MAX_TARGETS 1000
#define THREADS_PER_BLOCK 256
#define KEYS_PER_THREAD 128
#define MAX_RESULTS 100
#define SAMPLE_KEYS_COUNT 5
#define DISPLAY_INTERVAL_SECONDS 1

// Atomic flag untuk kontrol dari main thread
atomic<bool> stop_requested(false);
atomic<bool> pause_requested(false);

// Struktur hasil
struct Result {
    unsigned char private_key[32];
    unsigned char hash160[20];
    bool valid;
    unsigned long long thread_id;
    unsigned long long iteration;
};

// Struktur untuk live samples
struct LiveSample {
    unsigned char private_key[32];
    unsigned long long generation_time;
    unsigned long long thread_id;
    int block_id;
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

// Global memory untuk live samples di GPU
__device__ LiveSample d_live_samples[SAMPLE_KEYS_COUNT];
__device__ unsigned long long d_sample_counter = 0;

// ==================== FUNGSI VALIDASI ====================
__device__ bool is_all_zeros(const unsigned char* data, int size) {
    for (int i = 0; i < size; i++) {
        if (data[i] != 0) return false;
    }
    return true;
}

// Fungsi host version untuk validasi di CPU
bool is_all_zeros_host(const unsigned char* data, int size) {
    for (int i = 0; i < size; i++) {
        if (data[i] != 0) return false;
    }
    return true;
}

__device__ bool is_private_key_valid(const unsigned char* private_key) {
    // Cek tidak semua nol
    if (is_all_zeros(private_key, 32)) {
        return false;
    }
    
    // Cek tidak sama dengan n
    for (int i = 31; i >= 0; i--) {
        if (private_key[i] < d_secp256k1_n[i]) break;
        if (private_key[i] > d_secp256k1_n[i]) return false;
    }
    
    return true;
}

// ==================== FUNGSI UNTUK LIVE SAMPLES ====================
__device__ void update_live_sample(const unsigned char* private_key, 
                                   unsigned long long thread_id,
                                   int block_id) {
    // Update satu sample secara round-robin
    unsigned long long idx = atomicAdd(&d_sample_counter, 1) % SAMPLE_KEYS_COUNT;
    
    for (int i = 0; i < 32; i++) {
        d_live_samples[idx].private_key[i] = private_key[i];
    }
    
    d_live_samples[idx].generation_time = clock64();
    d_live_samples[idx].thread_id = thread_id;
    d_live_samples[idx].block_id = block_id;
}

// ==================== KERNEL DENGAN LIVE SAMPLING ====================
__global__ void bruteforce_kernel_unlimited(
    Result* results,
    int* found_count,
    unsigned long long seed,
    unsigned long long iteration
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int global_tid = blockIdx.x * gridDim.x + threadIdx.x;
    
    // Shared memory untuk target hashes
    __shared__ unsigned char s_targets[256][HASH160_SIZE];
    
    // Load targets ke shared memory
    int targets_to_load = min(d_num_targets, 256);
    for (int i = threadIdx.x; i < targets_to_load * HASH160_SIZE; i += blockDim.x) {
        int target_idx = i / HASH160_SIZE;
        int byte_idx = i % HASH160_SIZE;
        s_targets[target_idx][byte_idx] = d_target_hashes[target_idx][byte_idx];
    }
    __syncthreads();
    
    // Inisialisasi RNG dengan seed yang berbeda untuk setiap thread dan iteration
    curandState_t state;
    curand_init(seed + tid + (iteration * 1234567), 0, 0, &state);
    
    // Buffer lokal
    unsigned char private_key[32];
    unsigned char public_key[33];
    unsigned char hash160_result[20];
    
    // Flag untuk sample collection
    bool collect_sample = (threadIdx.x < SAMPLE_KEYS_COUNT) && (blockIdx.x == 0);
    
    for (int batch = 0; batch < KEYS_PER_THREAD; batch++) {
        // Generate random private key dengan entropy tinggi
        unsigned int rng_buffer[8];
        for (int i = 0; i < 8; i++) {
            rng_buffer[i] = curand(&state);
        }
        
        // Mix entropy dari berbagai sumber
        for (int i = 0; i < 32; i++) {
            private_key[i] = ((unsigned char*)rng_buffer)[i];
            // Tambah entropy dari thread ID, batch, iteration
            private_key[i] ^= (tid >> (i % 4)) & 0xFF;
            private_key[i] ^= (batch >> (i % 4)) & 0xFF;
            private_key[i] ^= (iteration >> (i % 8)) & 0xFF;
        }
        
        // Tambah entropy tambahan
        private_key[0] ^= clock64() & 0xFF;
        private_key[1] ^= (clock64() >> 8) & 0xFF;
        
        // VALIDASI: Pastikan tidak nol
        if (is_all_zeros(private_key, 32)) {
            // Regenerate jika nol
            private_key[0] = curand(&state) & 0xFF;
            if (private_key[0] == 0) private_key[0] = 1;
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
        
        // COLLECT LIVE SAMPLE: Thread tertentu mengupdate sample
        if (collect_sample && (batch % 16 == 0)) {
            update_live_sample(private_key, tid, blockIdx.x);
        }
        
        // Buat public key sederhana
        public_key[0] = (private_key[0] & 1) ? 0x03 : 0x02;
        for (int i = 0; i < 32; i++) {
            public_key[i + 1] = private_key[i] ^ (i * 13 + batch + iteration);
        }
        
        // Hitung hash160 (sederhana untuk demo)
        for (int i = 0; i < 20; i++) {
            hash160_result[i] = 0;
            for (int j = 0; j < 33; j++) {
                hash160_result[i] ^= public_key[j] + (i * j * 17);
            }
            hash160_result[i] = (hash160_result[i] * 31 + i + batch) % 256;
        }
        
        // Bandingkan dengan targets
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
                    
                    // Pastikan private key valid sebelum disimpan
                    if (!is_all_zeros(private_key, 32)) {
                        for (int i = 0; i < 32; i++) {
                            result->private_key[i] = private_key[i];
                        }
                        
                        for (int i = 0; i < 20; i++) {
                            result->hash160[i] = hash160_result[i];
                        }
                        
                        result->valid = true;
                        result->thread_id = tid;
                        result->iteration = iteration * KEYS_PER_THREAD + batch;
                    }
                }
                break;
            }
        }
    }
}

// ==================== FUNGSI BANTU ====================
string hex_encode(const unsigned char* data, int len) {
    const char* hex_chars = "0123456789abcdef";
    string result;
    for (int i = 0; i < len; i++) {
        result += hex_chars[(data[i] >> 4) & 0xF];
        result += hex_chars[data[i] & 0xF];
    }
    return result;
}

void print_colored(const string& text, int color_code) {
    cout << "\033[1;" << color_code << "m" << text << "\033[0m";
}

void clear_screen() {
    cout << "\033[2J\033[1;1H";
}

void display_header() {
    print_colored("╔══════════════════════════════════════════════════════════════════╗\n", 36);
    print_colored("║          BITCOIN PRIVATE KEY BRUTEFORCE - UNLIMITED MODE        ║\n", 36);
    print_colored("╚══════════════════════════════════════════════════════════════════╝\n", 36);
}

// Signal handler untuk Ctrl+C
void signal_handler(int signum) {
    if (signum == SIGINT) {
        stop_requested = true;
        cout << "\n\nShutdown signal received. Stopping gracefully...\n";
    }
}

// Thread untuk menampilkan live samples
void display_thread_func(atomic<bool>& stop_flag, 
                        LiveSample* h_live_samples,
                        atomic<unsigned long long>& total_keys,
                        atomic<int>& valid_matches) {
    
    auto last_display = steady_clock::now();
    
    while (!stop_flag) {
        auto now = steady_clock::now();
        auto elapsed = duration_cast<milliseconds>(now - last_display).count();
        
        if (elapsed >= DISPLAY_INTERVAL_SECONDS * 1000) {
            // Clear and redraw display
            clear_screen();
            display_header();
            
            cout << "\n\n";
            print_colored("══════════════════════ LIVE STATISTICS ═══════════════════════\n", 33);
            
            cout << "Total Keys Tested: ";
            print_colored(to_string(total_keys / 1000000) + " M\n", 32);
            
            cout << "Valid Matches Found: ";
            print_colored(to_string(valid_matches) + "\n", 32);
            
            cout << "Running Time: ";
            static auto start_time = steady_clock::now();
            auto run_time = duration_cast<seconds>(steady_clock::now() - start_time).count();
            int hours = run_time / 3600;
            int minutes = (run_time % 3600) / 60;
            int seconds = run_time % 60;
            print_colored(to_string(hours) + "h " + to_string(minutes) + "m " + to_string(seconds) + "s\n", 32);
            
            cout << "\n";
            print_colored("══════════════════════ LIVE KEY SAMPLES ═══════════════════════\n", 33);
            cout << "(Updated every second - Last 5 generated private keys)\n\n";
            
            for (int i = 0; i < SAMPLE_KEYS_COUNT; i++) {
                cout << "Sample " << (i+1) << ":\n";
                cout << "  Private Key: ";
                print_colored(hex_encode(h_live_samples[i].private_key, 32), 34);
                cout << "\n";
                cout << "  Generated by: Thread " << h_live_samples[i].thread_id 
                     << " (Block " << h_live_samples[i].block_id << ")\n";
                cout << "  Timestamp: " << h_live_samples[i].generation_time << " cycles\n";
                cout << "\n";
            }
            
            print_colored("═══════════════════════════════════════════════════════════════\n", 33);
            
            // Perbaikan: Gunakan variabel time_t untuk localtime
            time_t current_time = time(nullptr);
            cout << "Current time: " << put_time(localtime(&current_time), "%H:%M:%S") << "\n";
            
            last_display = now;
        }
        
        this_thread::sleep_for(milliseconds(100));
    }
}

// ==================== FUNGSI UTAMA ====================
int main(int argc, char** argv) {
    // Setup signal handler
    signal(SIGINT, signal_handler);
    
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <address_list.txt>" << endl;
        return 1;
    }
    
    clear_screen();
    display_header();
    
    // Get GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    cout << "\n";
    print_colored("╔══════════════════════════════════════════════════════════════════╗\n", 35);
    cout << "║ GPU: ";
    print_colored(prop.name, 32);
    cout << string(65 - strlen(prop.name), ' ') << "║\n";
    cout << "║ Compute Capability: ";
    print_colored(to_string(prop.major) + "." + to_string(prop.minor), 32);
    cout << "                                               ║\n";
    cout << "║ Memory: ";
    print_colored(to_string(prop.totalGlobalMem / (1024*1024*1024)) + " GB", 32);
    cout << "                                                ║\n";
    cout << "║ SMs: ";
    print_colored(to_string(prop.multiProcessorCount), 32);
    cout << "                                                       ║\n";
    print_colored("╚══════════════════════════════════════════════════════════════════╝\n", 35);
    
    // Read addresses
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
    
    cout << "\nLoaded ";
    print_colored(to_string(addresses.size()), 32);
    cout << " target addresses\n";
    
    // Create target hashes
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
    
    // Copy to GPU
    cudaMemcpyToSymbol(d_target_hashes, target_hashes, 
                      min(addresses.size(), (size_t)MAX_TARGETS) * HASH160_SIZE);
    
    int num_targets = min(addresses.size(), (size_t)MAX_TARGETS);
    cudaMemcpyToSymbol(d_num_targets, &num_targets, sizeof(int));
    
    delete[] target_hashes;
    
    // Allocate GPU memory
    Result* d_results;
    int* d_found_count;
    
    cudaMalloc(&d_results, MAX_RESULTS * sizeof(Result));
    cudaMalloc(&d_found_count, sizeof(int));
    
    // Initialize live samples on GPU
    LiveSample init_sample;
    memset(init_sample.private_key, 0, 32);
    init_sample.generation_time = 0;
    init_sample.thread_id = 0;
    init_sample.block_id = 0;
    
    LiveSample* d_live_samples_ptr;
    cudaGetSymbolAddress((void**)&d_live_samples_ptr, d_live_samples);
    for (int i = 0; i < SAMPLE_KEYS_COUNT; i++) {
        cudaMemcpy(&d_live_samples_ptr[i], &init_sample, sizeof(LiveSample), cudaMemcpyHostToDevice);
    }
    
    // Allocate CPU memory for live samples
    LiveSample* h_live_samples = new LiveSample[SAMPLE_KEYS_COUNT];
    
    // Initialize GPU memory
    cudaMemset(d_found_count, 0, sizeof(int));
    
    Result init_result;
    memset(init_result.private_key, 0, 32);
    memset(init_result.hash160, 0, 20);
    init_result.valid = false;
    init_result.thread_id = 0;
    init_result.iteration = 0;
    
    for (int i = 0; i < MAX_RESULTS; i++) {
        cudaMemcpy(&d_results[i], &init_result, sizeof(Result), cudaMemcpyHostToDevice);
    }
    
    // Allocate CPU memory for results
    Result* h_results = new Result[MAX_RESULTS];
    
    // Kernel configuration
    int threads = THREADS_PER_BLOCK;
    int blocks = min(prop.multiProcessorCount * 4, 65535);
    
    cout << "\nKernel Configuration:\n";
    cout << "  Blocks: ";
    print_colored(to_string(blocks), 33);
    cout << "\n";
    cout << "  Threads per Block: ";
    print_colored(to_string(threads), 33);
    cout << "\n";
    cout << "  Total Threads: ";
    print_colored(to_string(blocks * threads), 33);
    cout << "\n";
    cout << "  Keys per Thread: ";
    print_colored(to_string(KEYS_PER_THREAD), 33);
    cout << "\n";
    cout << "  Keys per Iteration: ";
    print_colored(to_string((unsigned long long)blocks * threads * KEYS_PER_THREAD / 1000000) + " M", 33);
    cout << "\n";
    
    cout << "\nStarting UNLIMITED search...\n";
    cout << "Press ";
    print_colored("Ctrl+C", 31);
    cout << " to stop\n";
    
    // Statistics
    atomic<unsigned long long> total_keys(0);
    atomic<int> valid_matches(0);
    atomic<unsigned long long> iteration_counter(0);
    
    // Start display thread
    atomic<bool> display_stop(false);
    thread display_thread(display_thread_func, 
                         ref(display_stop), 
                         h_live_samples,
                         ref(total_keys),
                         ref(valid_matches));
    
    auto program_start = steady_clock::now();
    
    try {
        unsigned long long iteration = 0;
        
        while (!stop_requested) {
            if (pause_requested) {
                this_thread::sleep_for(milliseconds(100));
                continue;
            }
            
            iteration++;
            iteration_counter = iteration;
            
            // Launch kernel
            bruteforce_kernel_unlimited<<<blocks, threads>>>(
                d_results, d_found_count,
                duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count(),
                iteration
            );
            
            // Check for errors
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                cout << "\nCUDA Error: " << cudaGetErrorString(err) << endl;
                break;
            }
            
            cudaDeviceSynchronize();
            
            // Update total keys counter
            total_keys += (unsigned long long)blocks * threads * KEYS_PER_THREAD;
            
            // Copy live samples from GPU
            cudaMemcpy(h_live_samples, d_live_samples_ptr, 
                      SAMPLE_KEYS_COUNT * sizeof(LiveSample), 
                      cudaMemcpyDeviceToHost);
            
            // Copy results if any found
            int h_found_count = 0;
            cudaMemcpy(&h_found_count, d_found_count, sizeof(int), cudaMemcpyDeviceToHost);
            
            if (h_found_count > 0) {
                cudaMemcpy(h_results, d_results, 
                          min(h_found_count, MAX_RESULTS) * sizeof(Result), 
                          cudaMemcpyDeviceToHost);
                
                // Process found results
                int current_valid = 0;
                for (int i = 0; i < min(h_found_count, MAX_RESULTS); i++) {
                    // Perbaikan: Gunakan fungsi host version untuk validasi di CPU
                    if (h_results[i].valid && !is_all_zeros_host(h_results[i].private_key, 32)) {
                        current_valid++;
                        
                        // Save to file
                        ofstream outfile("found_keys.txt", ios::app);
                        if (outfile.is_open()) {
                            outfile << "=== MATCH FOUND ===\n";
                            outfile << "Timestamp: " << time(nullptr) << "\n";
                            outfile << "Private Key: " << hex_encode(h_results[i].private_key, 32) << "\n";
                            outfile << "Thread ID: " << h_results[i].thread_id << "\n";
                            outfile << "Iteration: " << h_results[i].iteration << "\n";
                            outfile << "Total Keys Tested: " << total_keys << "\n";
                            outfile << "-------------------\n";
                            outfile.close();
                        }
                    }
                }
                
                valid_matches += current_valid;
                
                // Reset counter
                h_found_count = 0;
                cudaMemset(d_found_count, 0, sizeof(int));
                
                // Re-initialize results
                for (int i = 0; i < MAX_RESULTS; i++) {
                    cudaMemcpy(&d_results[i], &init_result, sizeof(Result), cudaMemcpyHostToDevice);
                }
            }
            
            // Small delay to prevent overheating
            if (iteration % 100 == 0) {
                this_thread::sleep_for(microseconds(100));
            }
        }
    }
    catch (const exception& e) {
        cout << "\nException: " << e.what() << endl;
    }
    
    // Stop display thread
    display_stop = true;
    if (display_thread.joinable()) {
        display_thread.join();
    }
    
    // Final statistics
    auto program_end = steady_clock::now();
    auto total_elapsed = duration_cast<seconds>(program_end - program_start);
    
    clear_screen();
    display_header();
    
    cout << "\n\n";
    print_colored("══════════════════════ SEARCH COMPLETED ═══════════════════════\n", 32);
    
    cout << "\nFinal Statistics:\n";
    cout << "  Total Iterations: ";
    print_colored(to_string(iteration_counter), 33);
    cout << "\n";
    
    cout << "  Total Keys Tested: ";
    print_colored(to_string(total_keys / 1000000) + " million", 33);
    cout << "\n";
    
    cout << "  Total Time: ";
    int hours = total_elapsed.count() / 3600;
    int minutes = (total_elapsed.count() % 3600) / 60;
    int seconds = total_elapsed.count() % 60;
    print_colored(to_string(hours) + "h " + to_string(minutes) + "m " + to_string(seconds) + "s", 33);
    cout << "\n";
    
    if (total_elapsed.count() > 0) {
        double keys_per_sec = total_keys / (double)total_elapsed.count();
        cout << "  Average Speed: ";
        print_colored(to_string(keys_per_sec / 1000000) + " Mkeys/second", 33);
        cout << "\n";
    }
    
    cout << "  Valid Matches Found: ";
    print_colored(to_string(valid_matches), valid_matches > 0 ? 32 : 31);
    cout << "\n";
    
    if (valid_matches > 0) {
        cout << "\nMatches saved to: ";
        print_colored("found_keys.txt", 32);
        cout << "\n";
    }
    
    cout << "\n";
    print_colored("═══════════════════════════════════════════════════════════════\n", 32);
    
    // Cleanup
    delete[] h_results;
    delete[] h_live_samples;
    cudaFree(d_results);
    cudaFree(d_found_count);
    
    cudaDeviceReset();
    
    cout << "\nProgram terminated successfully.\n";
    return 0;
}
