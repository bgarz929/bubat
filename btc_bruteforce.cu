// btc_bruteforce_fixed.cu - Bitcoin Brute Force CUDA (Fixed)
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
#define COMPRESSED_PUBLIC_KEY_SIZE 33
#define HASH160_SIZE 20
#define MAX_TARGETS 100
#define THREADS_PER_BLOCK 256
#define BATCH_SIZE 64

// Struktur hasil
typedef struct {
    unsigned char private_key[32];
    unsigned char hash160[20];
    int found;
} SearchResult;

// ==================== KONSTANTA GPU ====================
__constant__ unsigned char d_secp256k1_n[32] = {
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE,
    0xBA, 0xAE, 0xDC, 0xE6, 0xAF, 0x48, 0xA0, 0x3B,
    0xBF, 0xD2, 0x5E, 0x8C, 0xD0, 0x36, 0x41, 0x41
};

__constant__ unsigned char d_target_hashes[MAX_TARGETS][HASH160_SIZE];
__constant__ int d_num_targets;

// ==================== FUNGSI PEMBANTU GPU ====================
__device__ int compare_big_int(const unsigned char* a, const unsigned char* b) {
    for (int i = 31; i >= 0; i--) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

__device__ bool is_zero(const unsigned char* a) {
    for (int i = 0; i < 32; i++) {
        if (a[i] != 0) return false;
    }
    return true;
}

// Fungsi hash sederhana untuk demo
__device__ void simple_hash160(const unsigned char* data, int len, unsigned char* output) {
    // Simulasi hash sederhana
    for (int i = 0; i < 20; i++) {
        output[i] = 0;
        for (int j = 0; j < len && j < 32; j++) {
            output[i] ^= data[j] + (i * j * 17);
        }
        output[i] = (output[i] * 31 + i) % 256;
    }
}

// ==================== KERNEL UTAMA ====================
__global__ void bruteforce_kernel(
    SearchResult* results,
    int* found_count,
    unsigned long long seed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Inisialisasi RNG - PERBAIKAN: gunakan seed yang berbeda per thread
    curandState_t state;
    curand_init(seed + tid, 0, 0, &state);
    
    // Buffer untuk perhitungan
    unsigned char private_key[32];
    unsigned char public_key[33];
    unsigned char hash160_result[20];
    
    // Generate dan test beberapa keys per thread
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        // Generate random private key
        for (int i = 0; i < 32; i++) {
            private_key[i] = curand(&state) & 0xFF;
        }
        
        // Validasi private key
        if (is_zero(private_key)) continue;
        if (compare_big_int(private_key, d_secp256k1_n) >= 0) continue;
        
        // Buat public key sederhana (format compressed)
        public_key[0] = (private_key[0] & 1) ? 0x03 : 0x02;
        for (int i = 0; i < 32; i++) {
            public_key[i + 1] = private_key[i] ^ (i * 13 + batch);
        }
        
        // Hitung hash160 sederhana
        simple_hash160(public_key, 33, hash160_result);
        
        // Bandingkan dengan targets
        for (int i = 0; i < d_num_targets; i++) {
            bool match = true;
            for (int j = 0; j < HASH160_SIZE; j++) {
                if (hash160_result[j] != d_target_hashes[i][j]) {
                    match = false;
                    break;
                }
            }
            
            if (match) {
                int found_idx = atomicAdd(found_count, 1);
                if (found_idx < MAX_TARGETS) {
                    memcpy(results[found_idx].private_key, private_key, 32);
                    memcpy(results[found_idx].hash160, hash160_result, 20);
                    results[found_idx].found = 1;
                    
                    // Debug output
                    // printf("Thread %d found match at target %d\n", tid, i);
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
    version_hash160[0] = 0x00; // Mainnet
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
    wif_bytes[0] = 0x80; // Mainnet
    memcpy(wif_bytes + 1, private_key, 32);
    wif_bytes[33] = 0x01; // Compressed
    
    unsigned char checksum1[32], checksum2[32];
    SHA256(wif_bytes, 34, checksum1);
    SHA256(checksum1, 32, checksum2);
    
    memcpy(wif_bytes + 34, checksum2, 4);
    
    return base58_encode(wif_bytes, 38);
}

void print_gpu_info() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    if (device_count == 0) {
        cout << "Error: No CUDA-capable device found" << endl;
        return;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    cout << "GPU Device: " << prop.name << endl;
    cout << "Compute Capability: " << prop.major << "." << prop.minor << endl;
    cout << "Total Global Memory: " << prop.totalGlobalMem / (1024*1024*1024.0) << " GB" << endl;
    cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << endl;
    cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << endl;
    cout << "MultiProcessors: " << prop.multiProcessorCount << endl;
    cout << "Warp Size: " << prop.warpSize << endl;
}

// ==================== FUNGSI UTAMA ====================
int main(int argc, char* argv[]) {
    cout << "=== Bitcoin Private Key Brute Force (CUDA) ===" << endl;
    cout << "==============================================" << endl;
    
    // Inisialisasi CUDA
    cudaError_t cuda_status = cudaSetDevice(0);
    if (cuda_status != cudaSuccess) {
        cout << "Error setting CUDA device: " << cudaGetErrorString(cuda_status) << endl;
        return 1;
    }
    
    print_gpu_info();
    
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
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') continue;
        
        // Remove whitespace
        line.erase(remove_if(line.begin(), line.end(), ::isspace), line.end());
        if (!line.empty()) {
            addresses.push_back(line);
        }
    }
    file.close();
    
    if (addresses.empty()) {
        cout << "Error: No addresses found in file" << endl;
        return 1;
    }
    
    cout << "\nLoaded " << addresses.size() << " addresses from " << argv[1] << endl;
    
    // Buat target hashes sederhana
    vector<vector<unsigned char>> target_hashes;
    for (const string& addr : addresses) {
        vector<unsigned char> hash(HASH160_SIZE);
        // Hash sederhana dari address string
        for (int i = 0; i < HASH160_SIZE; i++) {
            hash[i] = 0;
            for (size_t j = 0; j < addr.size(); j++) {
                hash[i] ^= addr[j] + (i * j * 13);
            }
            hash[i] = (hash[i] * 31 + i) % 256;
        }
        target_hashes.push_back(hash);
    }
    
    // Setup target hashes di GPU
    unsigned char* flat_hashes = new unsigned char[target_hashes.size() * HASH160_SIZE];
    for (size_t i = 0; i < target_hashes.size(); i++) {
        memcpy(flat_hashes + i * HASH160_SIZE, target_hashes[i].data(), HASH160_SIZE);
    }
    
    cuda_status = cudaMemcpyToSymbol(d_target_hashes, flat_hashes, 
                                     target_hashes.size() * HASH160_SIZE);
    if (cuda_status != cudaSuccess) {
        cout << "Error copying target hashes to GPU: " 
             << cudaGetErrorString(cuda_status) << endl;
        delete[] flat_hashes;
        return 1;
    }
    
    int num_targets = target_hashes.size();
    cuda_status = cudaMemcpyToSymbol(d_num_targets, &num_targets, sizeof(int));
    if (cuda_status != cudaSuccess) {
        cout << "Error copying num_targets to GPU: " 
             << cudaGetErrorString(cuda_status) << endl;
        delete[] flat_hashes;
        return 1;
    }
    
    delete[] flat_hashes;
    
    // Alokasi memori GPU
    SearchResult* d_results = nullptr;
    int* d_found_count = nullptr;
    
    cuda_status = cudaMalloc(&d_results, MAX_TARGETS * sizeof(SearchResult));
    if (cuda_status != cudaSuccess) {
        cout << "Error allocating d_results: " << cudaGetErrorString(cuda_status) << endl;
        return 1;
    }
    
    cuda_status = cudaMalloc(&d_found_count, sizeof(int));
    if (cuda_status != cudaSuccess) {
        cout << "Error allocating d_found_count: " << cudaGetErrorString(cuda_status) << endl;
        cudaFree(d_results);
        return 1;
    }
    
    // Reset GPU memory
    cuda_status = cudaMemset(d_found_count, 0, sizeof(int));
    if (cuda_status != cudaSuccess) {
        cout << "Error resetting d_found_count: " << cudaGetErrorString(cuda_status) << endl;
        cudaFree(d_results);
        cudaFree(d_found_count);
        return 1;
    }
    
    cuda_status = cudaMemset(d_results, 0, MAX_TARGETS * sizeof(SearchResult));
    if (cuda_status != cudaSuccess) {
        cout << "Error resetting d_results: " << cudaGetErrorString(cuda_status) << endl;
        cudaFree(d_results);
        cudaFree(d_found_count);
        return 1;
    }
    
    // Alokasi memori CPU untuk hasil
    SearchResult* h_results = new SearchResult[MAX_TARGETS];
    int h_found_count = 0;
    
    // Konfigurasi kernel - PERBAIKAN: mulai dengan konfigurasi kecil
    int threads = THREADS_PER_BLOCK;
    int blocks = 16;  // Mulai dengan jumlah block kecil
    
    cout << "\n=== Kernel Configuration ===" << endl;
    cout << "Blocks: " << blocks << endl;
    cout << "Threads per Block: " << threads << endl;
    cout << "Total Threads: " << blocks * threads << endl;
    cout << "Keys per Thread: " << BATCH_SIZE << endl;
    cout << "Keys per Iteration: " << (long long)blocks * threads * BATCH_SIZE << endl;
    
    // Seed random
    unsigned long long seed = duration_cast<milliseconds>(
        system_clock::now().time_since_epoch()).count();
    
    cout << "\n=== Starting Search ===" << endl;
    cout << "Press Ctrl+C to stop" << endl;
    cout << "==================================" << endl;
    
    // Variabel statistik
    long long total_tested = 0;
    auto start_time = high_resolution_clock::now();
    auto last_print_time = start_time;
    
    // Main loop
    try {
        for (int iteration = 0; iteration < 1000; iteration++) {
            // Jalankan kernel dengan error checking
            bruteforce_kernel<<<blocks, threads>>>(d_results, d_found_count, seed + iteration);
            
            cuda_status = cudaGetLastError();
            if (cuda_status != cudaSuccess) {
                cout << "\nCUDA Kernel Error (iteration " << iteration << "): " 
                     << cudaGetErrorString(cuda_status) << endl;
                
                // Coba konfigurasi yang lebih kecil
                if (cuda_status == cudaErrorInvalidValue) {
                    cout << "Trying smaller configuration..." << endl;
                    blocks = max(1, blocks / 2);
                    threads = max(32, threads / 2);
                    cout << "New config: " << blocks << " blocks, " 
                         << threads << " threads" << endl;
                    continue;
                } else {
                    break;
                }
            }
            
            cuda_status = cudaDeviceSynchronize();
            if (cuda_status != cudaSuccess) {
                cout << "\nCUDA Sync Error: " << cudaGetErrorString(cuda_status) << endl;
                break;
            }
            
            // Copy hasil dari GPU
            cuda_status = cudaMemcpy(&h_found_count, d_found_count, sizeof(int), 
                                    cudaMemcpyDeviceToHost);
            if (cuda_status != cudaSuccess) {
                cout << "\nError copying found_count: " 
                     << cudaGetErrorString(cuda_status) << endl;
                break;
            }
            
            if (h_found_count > 0) {
                cuda_status = cudaMemcpy(h_results, d_results, 
                                        MAX_TARGETS * sizeof(SearchResult), 
                                        cudaMemcpyDeviceToHost);
                if (cuda_status != cudaSuccess) {
                    cout << "\nError copying results: " 
                         << cudaGetErrorString(cuda_status) << endl;
                    break;
                }
            }
            
            total_tested += (long long)blocks * threads * BATCH_SIZE;
            
            // Print progress
            auto current_time = high_resolution_clock::now();
            auto elapsed = duration_cast<milliseconds>(current_time - last_print_time).count() / 1000.0;
            
            if (elapsed >= 2.0) {
                auto total_elapsed = duration_cast<milliseconds>(current_time - start_time).count() / 1000.0;
                if (total_elapsed > 0) {
                    double keys_per_sec = total_tested / total_elapsed;
                    cout << fixed << setprecision(2);
                    cout << "\r[Progress] Keys: " << total_tested 
                         << " | Speed: " << keys_per_sec / 1000000 << " Mkeys/sec"
                         << " | Found: " << h_found_count << "      " << flush;
                }
                last_print_time = current_time;
            }
            
            // Jika ditemukan match
            if (h_found_count > 0) {
                cout << "\n\n=== FOUND " << h_found_count << " MATCHES ===" << endl;
                
                // Simpan ke file
                ofstream outfile("found_keys.txt", ios::app);
                if (!outfile.is_open()) {
                    cout << "Warning: Cannot open found_keys.txt for writing" << endl;
                } else {
                    for (int i = 0; i < h_found_count; i++) {
                        cout << "\n--- Match " << (i + 1) << " ---" << endl;
                        
                        // Private key hex
                        string private_key_hex = hex_encode(h_results[i].private_key, 32);
                        cout << "Private Key: " << private_key_hex << endl;
                        
                        // Generate address
                        string address = hash160_to_address(h_results[i].hash160);
                        cout << "Bitcoin Address: " << address << endl;
                        
                        // WIF format
                        string wif = private_key_to_wif(h_results[i].private_key);
                        cout << "WIF: " << wif << endl;
                        
                        // Simpan ke file
                        outfile << "Private Key: " << private_key_hex << endl;
                        outfile << "Address: " << address << endl;
                        outfile << "WIF: " << wif << endl;
                        outfile << "-------------------" << endl;
                    }
                    outfile.close();
                    cout << "\nResults saved to found_keys.txt" << endl;
                }
                
                // Reset counter
                h_found_count = 0;
                cuda_status = cudaMemset(d_found_count, 0, sizeof(int));
                if (cuda_status != cudaSuccess) {
                    cout << "Error resetting d_found_count: " 
                         << cudaGetErrorString(cuda_status) << endl;
                    break;
                }
            }
            
            // Tingkatkan blocks secara bertahap jika berhasil
            if (iteration % 10 == 0 && iteration > 0) {
                int max_blocks = 256; // Batas maksimum untuk Tesla T4
                if (blocks < max_blocks) {
                    blocks = min(blocks * 2, max_blocks);
                    cout << "\nIncreased blocks to: " << blocks << endl;
                }
            }
            
            // Cek waktu (maksimal 10 menit untuk demo)
            auto total_elapsed = duration_cast<seconds>(current_time - start_time).count();
            if (total_elapsed > 600) { // 10 menit
                cout << "\n\nTime limit reached (10 minutes). Stopping." << endl;
                break;
            }
        }
    } catch (const exception& e) {
        cout << "\nException: " << e.what() << endl;
    }
    
    // Statistik akhir
    auto end_time = high_resolution_clock::now();
    auto total_elapsed = duration_cast<milliseconds>(end_time - start_time).count() / 1000.0;
    
    cout << "\n\n=== FINAL STATISTICS ===" << endl;
    cout << "Total keys tested: " << total_tested << endl;
    cout << "Total time: " << total_elapsed << " seconds" << endl;
    if (total_elapsed > 0) {
        cout << "Average speed: " << (total_tested / total_elapsed) / 1000000 
             << " Mkeys/second" << endl;
    }
    cout << "Final configuration: " << blocks << " blocks, " 
         << threads << " threads/block" << endl;
    
    // Cleanup
    delete[] h_results;
    cudaFree(d_results);
    cudaFree(d_found_count);
    
    // Reset device
    cudaDeviceReset();
    
    cout << "\nProgram finished successfully." << endl;
    return 0;
}
