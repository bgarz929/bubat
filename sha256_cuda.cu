#include "sha256_cuda.h"
#include <cuda.h>

// Konstanta SHA256
__constant__ unsigned int d_sha256_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// Fungsi rotasi SHA256
__device__ __forceinline__ unsigned int rotr32(unsigned int x, int n) {
    return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ unsigned int ch(unsigned int x, unsigned int y, unsigned int z) {
    return (x & y) ^ (~x & z);
}

__device__ __forceinline__ unsigned int maj(unsigned int x, unsigned int y, unsigned int z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __forceinline__ unsigned int sigma0(unsigned int x) {
    return rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22);
}

__device__ __forceinline__ unsigned int sigma1(unsigned int x) {
    return rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25);
}

__device__ __forceinline__ unsigned int gamma0(unsigned int x) {
    return rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3);
}

__device__ __forceinline__ unsigned int gamma1(unsigned int x) {
    return rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10);
}

// SHA256 di GPU dengan shared memory optimasi
__device__ void sha256_transform(unsigned int* state, const unsigned char* data) {
    unsigned int w[64];
    unsigned int a, b, c, d, e, f, g, h;
    int i;
    
    // Load data ke w[0..15]
    #pragma unroll
    for (i = 0; i < 16; i++) {
        w[i] = (data[i*4] << 24) | (data[i*4+1] << 16) | 
               (data[i*4+2] << 8) | data[i*4+3];
    }
    
    // Expand message
    #pragma unroll
    for (i = 16; i < 64; i++) {
        w[i] = gamma1(w[i-2]) + w[i-7] + gamma0(w[i-15]) + w[i-16];
    }
    
    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];
    f = state[5];
    g = state[6];
    h = state[7];
    
    // Compression function
    #pragma unroll
    for (i = 0; i < 64; i++) {
        unsigned int t1 = h + sigma1(e) + ch(e, f, g) + d_sha256_k[i] + w[i];
        unsigned int t2 = sigma0(a) + maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

__device__ void sha256(const unsigned char* data, int len, unsigned char* output) {
    unsigned int state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    
    // Process full 64-byte chunks
    int i;
    for (i = 0; i + 64 <= len; i += 64) {
        sha256_transform(state, data + i);
    }
    
    // Padding untuk data terakhir
    unsigned char pad[64];
    int rem = len - i;
    int pad_len = (rem < 56) ? 64 : 128;
    
    // Salin data terakhir
    if (rem > 0) {
        #pragma unroll
        for (int j = 0; j < rem; j++) {
            pad[j] = data[i + j];
        }
    }
    
    // Tambah bit '1'
    pad[rem] = 0x80;
    
    // Isi dengan 0
    #pragma unroll
    for (int j = rem + 1; j < (pad_len == 64 ? 56 : 120); j++) {
        pad[j] = 0;
    }
    
    // Tambah panjang (bits) sebagai 64-bit big-endian
    unsigned long long bit_len = (unsigned long long)len * 8;
    if (pad_len == 64) {
        for (int j = 0; j < 8; j++) {
            pad[56 + j] = (bit_len >> (56 - j * 8)) & 0xFF;
        }
        sha256_transform(state, pad);
    } else {
        for (int j = 0; j < 8; j++) {
            pad[120 + j] = (bit_len >> (56 - j * 8)) & 0xFF;
        }
        sha256_transform(state, pad);
        sha256_transform(state, pad + 64);
    }
    
    // Convert state ke output (big-endian)
    #pragma unroll
    for (i = 0; i < 8; i++) {
        output[i*4] = (state[i] >> 24) & 0xFF;
        output[i*4+1] = (state[i] >> 16) & 0xFF;
        output[i*4+2] = (state[i] >> 8) & 0xFF;
        output[i*4+3] = state[i] & 0xFF;
    }
}
