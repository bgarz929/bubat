#ifndef SECP256K1_CUDA_H
#define SECP256K1_CUDA_H

#include <cuda.h>
#include <cuda_runtime.h>

// Ukuran konstan
#define PRIVATE_KEY_SIZE 32
#define PUBLIC_KEY_SIZE 65
#define COMPRESSED_PUBLIC_KEY_SIZE 33
#define HASH160_SIZE 20
#define ADDRESS_SIZE 25
#define SHA256_DIGEST_SIZE 32
#define RIPEMD160_DIGEST_SIZE 20
#define MAX_TARGETS 10000
#define THREADS_PER_BLOCK 256
#define SHARED_MEM_SIZE 49152  // 48KB per block

// Struktur untuk titik elliptic
typedef struct {
    unsigned char x[32];
    unsigned char y[32];
    bool infinity;
} EC_Point;

// Struktur hasil
typedef struct {
    unsigned char private_key[32];
    unsigned char public_key[33];
    char address[36];
    int target_index;
    int found;
} SearchResult;

// Konstanta kurva secp256k1
extern __constant__ unsigned char d_secp256k1_p[32];
extern __constant__ unsigned char d_secp256k1_n[32];
extern __constant__ unsigned char d_secp256k1_gx[32];
extern __constant__ unsigned char d_secp256k1_gy[32];
extern __constant__ unsigned char d_target_hashes[MAX_TARGETS][HASH160_SIZE];
extern __constant__ int d_num_targets;
extern __constant__ unsigned int d_batch_start[3];

// Prototipe fungsi
void init_secp256k1_constants();
cudaError_t setup_target_hashes(const unsigned char* hashes, int count);
cudaError_t launch_bruteforce_kernel(int blocks, int threads, 
                                     SearchResult* d_results, int* d_found_count,
                                     unsigned long long seed, int batch_size);
void generate_random_start_point(unsigned int start[3]);

#endif
