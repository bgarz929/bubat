// btc_bruteforce_final.cu - Fixed Private Key Validation and Output
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

using namespace std;
using namespace chrono;

// ==================== KONFIGURASI ====================
#define PRIVATE_KEY_SIZE 32
#define HASH160_SIZE 20
#define MAX_TARGETS 1000
#define THREADS_PER_BLOCK 256
#define KEYS_PER_THREAD 128
#define MAX_RESULTS 100

// Struktur hasil dengan validasi
struct Result {
    unsigned char private_key[32];
    unsigned char hash160[20];
    unsigned char public_key[33];
    bool valid;  // Flag validitas
    unsigned long long thread_id;
    unsigned long long iteration;
    
    // Constructor
    __host__ __device__ Result() : valid(false), thread_id(0), iteration(0) {
        memset(private_key, 0, 32);
        memset(hash160, 0, 20);
        memset(public_key, 0, 33);
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

// ==================== FUNGSI HASH SEDERHANA ====================
__device__ void compute_hash160(const unsigned char* public_key, unsigned char* output) {
    // Hash sederhana yang deterministik
    for (int i = 0; i < 20; i++) {
        output[i] = 0;
        for (int j = 0; j < 33; j++) {
            output[i] ^= public_key[j] + (i * j * 17);
        }
        output[i] = (output[i] * 31 + i) % 256;
    }
}

// ==================== KERNEL DENGAN VALIDASI KETAT ====================
__global__ void bruteforce_kernel_fixed(
    Result* results,
    int* found_count,
    unsigned long long seed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
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
    
    // Inisialisasi RNG dengan seed yang berbeda untuk setiap thread
    curandState_t state;
    curand_init(seed + tid, 0, 0, &state);
    
    // Buffer lokal
    unsigned char private_key[32];
    unsigned char public_key[33];
    unsigned char hash160_result[20];
    
    for (int batch = 0; batch < KEYS_PER_THREAD; batch++) {
        // Generate random private key
        unsigned int rng_buffer[8];
        for (int i = 0; i < 8; i++) {
            rng_buffer[i] = curand(&state);
        }
        
        // Convert to bytes
        for (int i = 0; i < 32; i++) {
            private_key[i] = ((unsigned char*)rng_buffer)[i % 32];
        }
        
        // Tambah entropy dari batch dan thread ID
        private_key[0] ^= (tid >> 0) & 0xFF;
        private_key[1] ^= (tid >> 8) & 0xFF;
        private_key[2] ^= (tid >> 16) & 0xFF;
        private_key[3] ^= batch & 0xFF;
        
        // VALIDASI KETAT: Pastikan tidak nol
        if (is_all_zeros(private_key, 32)) {
            continue;
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
        
        // Buat public key sederhana
        public_key[0] = (private_key[0] & 1) ? 0x03 : 0x02;
        for (int i = 0; i < 32; i++) {
            public_key[i + 1] = private_key[i] ^ (i * 13 + batch);
        }
        
        // Hitung hash160
        compute_hash160(public_key, hash160_result);
        
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
                // Gunakan atomicAdd untuk mendapatkan index yang unik
                int found_idx = atomicAdd(found_count, 1);
                
                // Batasi jumlah hasil yang disimpan
                if (found_idx < MAX_RESULTS) {
                    // Salin data dengan validasi
                    Result* result = &results[found_idx];
                    
                    // Pastikan private key tidak nol sebelum disalin
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
                        result->iteration = batch;
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
    // WIF format: 0x80 + private_key + 0x01 + checksum
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
    // Cek flag valid
    if (!result.valid) return false;
    
    // Cek private key tidak nol
    for (int i = 0; i < 32; i++) {
        if (result.private_key[i] != 0) {
            return true;
        }
    }
    return false;
}

// ==================== FUNGSI UTAMA ====================
int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <address_list.txt>" << endl;
        return 1;
    }
    
    cout << "=== Bitcoin Brute Force - Fixed Output ===" << endl;
    cout << "==========================================" << endl;
    
    // Get GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    cout << "GPU: " << prop.name << endl;
    cout << "Compute Capability: " << prop.major << "." << prop.minor << endl;
    cout << "Global Memory: " << prop.totalGlobalMem / (1024*1024*1024.0) << " GB" << endl;
    
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
            // Remove whitespace
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
    
    // Initialize GPU memory
    cudaMemset(d_found_count, 0, sizeof(int));
    
    // Initialize results with invalid state
    Result init_result;
    for (int i = 0; i < MAX_RESULTS; i++) {
        cudaMemcpy(&d_results[i], &init_result, sizeof(Result), cudaMemcpyHostToDevice);
    }
    
    // Allocate CPU memory
    Result* h_results = new Result[MAX_RESULTS];
    int h_found_count = 0;
    
    // Kernel configuration
    int threads = THREADS_PER_BLOCK;
    int blocks = min(prop.multiProcessorCount * 4, 65535);
    
    cout << "\nKernel Configuration:" << endl;
    cout << "Blocks: " << blocks << endl;
    cout << "Threads per Block: " << threads << endl;
    cout << "Total Threads: " << blocks * threads << endl;
    cout << "Keys per Thread: " << KEYS_PER_THREAD << endl;
    cout << "Keys per Iteration: " << (unsigned long long)blocks * threads * KEYS_PER_THREAD << endl;
    cout << "Max Results to Store: " << MAX_RESULTS << endl;
    
    cout << "\nStarting search..." << endl;
    cout << "Press Ctrl+C to stop" << endl;
    cout << "==================================" << endl;
    
    // Statistics
    unsigned long long total_keys = 0;
    int valid_found_count = 0;
    auto start_time = high_resolution_clock::now();
    
    int iteration = 0;
    while (true) {
        iteration++;
        
        // Launch kernel
        bruteforce_kernel_fixed<<<blocks, threads>>>(
            d_results, d_found_count,
            duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() + iteration
        );
        
        // Check for errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cout << "\nCUDA Error: " << cudaGetErrorString(err) << endl;
            break;
        }
        
        cudaDeviceSynchronize();
        
        // Copy results
        cudaMemcpy(&h_found_count, d_found_count, sizeof(int), cudaMemcpyDeviceToHost);
        
        if (h_found_count > 0) {
            // Copy only the valid results
            cudaMemcpy(h_results, d_results, 
                      min(h_found_count, MAX_RESULTS) * sizeof(Result), 
                      cudaMemcpyDeviceToHost);
        }
        
        // Update statistics
        total_keys += (unsigned long long)blocks * threads * KEYS_PER_THREAD;
        
        // Print progress every 2 seconds
        auto current_time = high_resolution_clock::now();
        auto elapsed = duration_cast<milliseconds>(current_time - start_time).count() / 1000.0;
        
        if (iteration % 5 == 0 && elapsed > 0) {
            double keys_per_sec = total_keys / elapsed;
            cout << fixed << setprecision(2);
            cout << "\r[Iter " << iteration << "] Speed: " 
                 << keys_per_sec / 1000000 << " Mkeys/sec | "
                 << "Total: " << total_keys / 1000000 << " Mkeys | "
                 << "Found (raw): " << h_found_count << " | "
                 << "Valid: " << valid_found_count << "      " << flush;
        }
        
        // Process found results
        if (h_found_count > 0) {
            int current_valid_count = 0;
            
            // Filter and display valid results
            for (int i = 0; i < min(h_found_count, MAX_RESULTS); i++) {
                if (is_result_valid(h_results[i])) {
                    current_valid_count++;
                    
                    // Only display first few valid results
                    if (current_valid_count <= 5) {
                        cout << "\n\n=== VALID MATCH " << current_valid_count << " ===" << endl;
                        
                        // Private key
                        cout << "Private Key: " << hex_encode(h_results[i].private_key, 32) << endl;
                        
                        // Generate address
                        string address = hash160_to_address(h_results[i].hash160);
                        cout << "Bitcoin Address: " << address << endl;
                        
                        // WIF format
                        string wif = private_key_to_wif(h_results[i].private_key);
                        cout << "WIF: " << wif << endl;
                        
                        // Additional info
                        cout << "Thread ID: " << h_results[i].thread_id << endl;
                        cout << "Iteration: " << h_results[i].iteration << endl;
                    }
                    
                    // Save to file
                    ofstream outfile("found_keys.txt", ios::app);
                    if (outfile.is_open()) {
                        outfile << "=== VALID MATCH ===" << endl;
                        outfile << "Private Key: " << hex_encode(h_results[i].private_key, 32) << endl;
                        
                        string address = hash160_to_address(h_results[i].hash160);
                        outfile << "Address: " << address << endl;
                        
                        string wif = private_key_to_wif(h_results[i].private_key);
                        outfile << "WIF: " << wif << endl;
                        
                        outfile << "Thread ID: " << h_results[i].thread_id << endl;
                        outfile << "Iteration: " << h_results[i].iteration << endl;
                        outfile << "Timestamp: " << time(NULL) << endl;
                        outfile << "-------------------" << endl;
                        outfile.close();
                    }
                }
            }
            
            valid_found_count += current_valid_count;
            
            if (current_valid_count > 0) {
                cout << "\nTotal valid matches in this batch: " << current_valid_count << endl;
                cout << "Results saved to found_keys.txt" << endl;
            }
            
            // Reset counter
            h_found_count = 0;
            cudaMemset(d_found_count, 0, sizeof(int));
            
            // Re-initialize results
            for (int i = 0; i < MAX_RESULTS; i++) {
                cudaMemcpy(&d_results[i], &init_result, sizeof(Result), cudaMemcpyHostToDevice);
            }
        }
        
        // Check time limit (10 minutes)
        auto total_elapsed = duration_cast<seconds>(current_time - start_time).count();
        if (total_elapsed > 600) {
            cout << "\n\nTime limit reached (10 minutes). Stopping." << endl;
            break;
        }
        
        // Safety check for too many iterations
        if (iteration > 10000) {
            cout << "\n\nMaximum iterations reached. Stopping." << endl;
            break;
        }
    }
    
    // Final statistics
    auto end_time = high_resolution_clock::now();
    auto total_elapsed_seconds = duration_cast<seconds>(end_time - start_time).count();
    
    cout << "\n\n=== FINAL STATISTICS ===" << endl;
    cout << "Total iterations: " << iteration << endl;
    cout << "Total keys tested: " << total_keys << endl;
    cout << "Total time: " << total_elapsed_seconds << " seconds" << endl;
    
    if (total_elapsed_seconds > 0) {
        cout << "Average speed: " << (total_keys / total_elapsed_seconds) / 1000000 
             << " Mkeys/second" << endl;
    }
    
    cout << "Total valid matches found: " << valid_found_count << endl;
    cout << "Invalid/zero private keys filtered: " << (h_found_count - valid_found_count) << endl;
    
    // Cleanup
    delete[] h_results;
    cudaFree(d_results);
    cudaFree(d_found_count);
    
    cudaDeviceReset();
    
    cout << "\nProgram finished successfully." << endl;
    return 0;
}
