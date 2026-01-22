#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <openssl/sha.h>
#include <openssl/ripemd.h>
#include "secp256k1_cuda.h"
#include "base58_cuda.h"

using namespace std;
using namespace std::chrono;

// Fungsi untuk decode base58 address ke hash160
bool decode_base58_address(const string& address, unsigned char* hash160) {
    vector<unsigned char> digits;
    for (char c : address) {
        const char* pos = strchr(BASE58_ALPHABET, c);
        if (!pos) return false;
        digits.push_back(pos - BASE58_ALPHABET);
    }
    
    // Konversi dari base58 ke bytes
    vector<unsigned char> bytes((digits.size() * 733) / 1000 + 1);
    int zeros = 0;
    while (zeros < digits.size() && digits[zeros] == 0) zeros++;
    
    int length = 0;
    for (int i = 0; i < digits.size(); i++) {
        int carry = digits[i];
        for (int j = bytes.size() - 1; j >= 0; j--) {
            carry += 58 * bytes[j];
            bytes[j] = carry % 256;
            carry /= 256;
        }
        while (carry > 0) {
            bytes.insert(bytes.begin(), carry % 256);
            carry /= 256;
        }
    }
    
    // Skip leading zeros
    int start = zeros;
    while (start < bytes.size() && bytes[start] == 0) start++;
    
    // Hash160 adalah 20 byte setelah version byte (0x00)
    if (bytes.size() - start != 25) return false;
    if (bytes[start] != 0x00) return false; // Mainnet
    
    // Verifikasi checksum
    unsigned char checksum1[32], checksum2[32];
    SHA256(&bytes[start], 21, checksum1);
    SHA256(checksum1, 32, checksum2);
    
    if (memcmp(&bytes[start + 21], checksum2, 4) != 0) return false;
    
    // Copy hash160
    memcpy(hash160, &bytes[start + 1], 20);
    return true;
}

// Fungsi untuk generate address dari public key
string generate_address_from_pubkey(const unsigned char* pubkey, bool compressed) {
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

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <address_list.txt>" << endl;
        return 1;
    }
    
    // Baca file address
    ifstream file(argv[1]);
    if (!file) {
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
    
    cout << "Loaded " << addresses.size() << " target addresses" << endl;
    
    // Decode addresses ke hash160
    vector<vector<unsigned char>> target_hashes;
    for (const string& addr : addresses) {
        unsigned char hash160[20];
        if (decode_base58_address(addr, hash160)) {
            target_hashes.push_back(vector<unsigned char>(hash160, hash160 + 20));
        } else {
            cout << "Warning: Invalid address format: " << addr << endl;
        }
    }
    
    if (target_hashes.empty()) {
        cout << "Error: No valid addresses found" << endl;
        return 1;
    }
    
    cout << "Valid targets: " << target_hashes.size() << endl;
    
    // Inisialisasi CUDA
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    cout << "\nGPU Device: " << prop.name << endl;
    cout << "Compute Capability: " << prop.major << "." << prop.minor << endl;
    cout << "Total Global Memory: " << prop.totalGlobalMem / (1024*1024*1024.0) << " GB" << endl;
    cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << endl;
    cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << endl;
    cout << "Warp Size: " << prop.warpSize << endl;
    
    // Setup konstanta secp256k1
    init_secp256k1_constants();
    
    // Setup target hashes
    unsigned char* flat_hashes = new unsigned char[target_hashes.size() * 20];
    for (size_t i = 0; i < target_hashes.size(); i++) {
        memcpy(flat_hashes + i * 20, target_hashes[i].data(), 20);
    }
    
    cudaError_t err = setup_target_hashes(flat_hashes, target_hashes.size());
    if (err != cudaSuccess) {
        cout << "Error setting up target hashes: " << cudaGetErrorString(err) << endl;
        return 1;
    }
    delete[] flat_hashes;
    
    // Alokasi memori
    SearchResult* d_results;
    int* d_found_count;
    SearchResult* h_results = new SearchResult[MAX_TARGETS];
    int h_found_count = 0;
    
    cudaMalloc(&d_results, MAX_TARGETS * sizeof(SearchResult));
    cudaMalloc(&d_found_count, sizeof(int));
    
    cudaMemset(d_found_count, 0, sizeof(int));
    cudaMemset(d_results, 0, MAX_TARGETS * sizeof(SearchResult));
    
    // Konfigurasi kernel
    int threads = THREADS_PER_BLOCK;
    int blocks = prop.multiProcessorCount * 16; // Optimasi untuk occupancy
    int batch_size = threads * blocks;
    
    cout << "\nKernel Configuration:" << endl;
    cout << "Blocks: " << blocks << endl;
    cout << "Threads per Block: " << threads << endl;
    cout << "Total Threads: " << batch_size << endl;
    cout << "Batch Size: " << batch_size * 64 << " keys per iteration" << endl;
    
    cout << "\nStarting bruteforce..." << endl;
    cout << "==========================================" << endl;
    
    // Variables untuk statistik
    long long total_tested = 0;
    auto start_time = high_resolution_clock::now();
    auto last_print_time = start_time;
    
    // Seed random
    unsigned long long seed = duration_cast<milliseconds>(
        system_clock::now().time_since_epoch()).count();
    
    // Main loop
    for (int iteration = 0; iteration < 1000000; iteration++) {
        // Jalankan kernel
        err = launch_bruteforce_kernel(blocks, threads, d_results, 
                                       d_found_count, seed + iteration, 
                                       batch_size);
        
        if (err != cudaSuccess) {
            cout << "Kernel error: " << cudaGetErrorString(err) << endl;
            break;
        }
        
        // Copy hasil
        cudaMemcpy(&h_found_count, d_found_count, sizeof(int), 
                  cudaMemcpyDeviceToHost);
        cudaMemcpy(h_results, d_results, MAX_TARGETS * sizeof(SearchResult),
                  cudaMemcpyDeviceToHost);
        
        total_tested += batch_size * 64;
        
        // Print progress setiap 5 detik
        auto current_time = high_resolution_clock::now();
        auto elapsed = duration_cast<milliseconds>(current_time - last_print_time).count() / 1000.0;
        
        if (elapsed >= 5.0) {
            auto total_elapsed = duration_cast<milliseconds>(current_time - start_time).count() / 1000.0;
            double keys_per_sec = total_tested / total_elapsed;
            
            cout << fixed << setprecision(2);
            cout << "\rProgress: " << total_tested << " keys, "
                 << keys_per_sec / 1000000 << " Mkeys/sec, "
                 << "Found: " << h_found_count << "        " << flush;
            
            last_print_time = current_time;
        }
        
        // Jika ditemukan
        if (h_found_count > 0) {
            cout << "\n\n=== FOUND " << h_found_count << " MATCHES ===" << endl;
            
            for (int i = 0; i < h_found_count; i++) {
                cout << "\nMatch " << (i + 1) << ":" << endl;
                
                // Tampilkan private key
                cout << "Private Key (hex): ";
                for (int j = 0; j < 32; j++) {
                    printf("%02x", h_results[i].private_key[j]);
                }
                cout << endl;
                
                // Tampilkan public key
                cout << "Public Key (compressed): ";
                printf("%02x", h_results[i].public_key[0]);
                for (int j = 1; j < 33; j++) {
                    printf("%02x", h_results[i].public_key[j]);
                }
                cout << endl;
                
                // Generate address untuk verifikasi
                string address = generate_address_from_pubkey(h_results[i].public_key, true);
                cout << "Bitcoin Address: " << address << endl;
                
                // Tampilkan target address yang cocok
                int target_idx = h_results[i].target_index;
                if (target_idx < addresses.size()) {
                    cout << "Target Address: " << addresses[target_idx] << endl;
                }
                
                cout << "WIF Format: " << 
                    generate_wif_from_private_key(h_results[i].private_key) << endl;
            }
            
            // Simpan ke file
            ofstream outfile("found_keys.txt");
            for (int i = 0; i < h_found_count; i++) {
                outfile << "Private Key: ";
                for (int j = 0; j < 32; j++) {
                    outfile << hex << setw(2) << setfill('0') 
                           << (int)h_results[i].private_key[j];
                }
                outfile << endl;
                
                string address = generate_address_from_pubkey(h_results[i].public_key, true);
                outfile << "Address: " << address << endl;
                
                outfile << "WIF: " << 
                    generate_wif_from_private_key(h_results[i].private_key) << endl;
                outfile << "-------------------" << endl;
            }
            outfile.close();
            
            cout << "\nResults saved to found_keys.txt" << endl;
            
            // Reset counter untuk melanjutkan pencarian
            h_found_count = 0;
            cudaMemset(d_found_count, 0, sizeof(int));
        }
        
        // Check for exit condition
        if (duration_cast<seconds>(current_time - start_time).count() > 3600) { // 1 jam
            cout << "\n\nTime limit reached. Stopping." << endl;
            break;
        }
    }
    
    // Statistik akhir
    auto end_time = high_resolution_clock::now();
    auto total_elapsed = duration_cast<milliseconds>(end_time - start_time).count() / 1000.0;
    
    cout << "\n\n=== FINAL STATISTICS ===" << endl;
    cout << "Total keys tested: " << total_tested << endl;
    cout << "Total time: " << total_elapsed << " seconds" << endl;
    cout << "Average speed: " << (total_tested / total_elapsed) / 1000000 
         << " Mkeys/second" << endl;
    cout << "Total iterations: " << (total_tested / (batch_size * 64)) << endl;
    
    // Cleanup
    delete[] h_results;
    cudaFree(d_results);
    cudaFree(d_found_count);
    
    return 0;
}
