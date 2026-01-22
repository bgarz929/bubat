#include "ripemd160_cuda.h"

// Konstanta RIPEMD160
__constant__ unsigned int d_ripemd160_k[5] = {
    0x00000000, 0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xa953fd4e
};

__constant__ unsigned int d_ripemd160_kp[5] = {
    0x50a28be6, 0x5c4dd124, 0x6d703ef3, 0x7a6d76e9, 0x00000000
};

__constant__ int d_ripemd160_r[80] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
    3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12,
    1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
    4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13
};

__constant__ int d_ripemd160_rp[80] = {
    5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12,
    6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
    15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13,
    8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
    12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11
};

__constant__ int d_ripemd160_s[80] = {
    11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8,
    7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
    11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5,
    11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
    9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6
};

__constant__ int d_ripemd160_sp[80] = {
    8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6,
    9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
    9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5,
    15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
    8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11
};

// Fungsi helper RIPEMD160
__device__ __forceinline__ unsigned int rol(unsigned int x, int n) {
    return (x << n) | (x >> (32 - n));
}

__device__ __forceinline__ unsigned int f(unsigned int j, unsigned int x, unsigned int y, unsigned int z) {
    if (j < 16) return x ^ y ^ z;
    if (j < 32) return (x & y) | (~x & z);
    if (j < 48) return (x | ~y) ^ z;
    if (j < 64) return (x & z) | (y & ~z);
    return x ^ (y | ~z);
}

// RIPEMD160 di GPU
__device__ void ripemd160(const unsigned char* data, int len, unsigned char* output) {
    unsigned int h0 = 0x67452301;
    unsigned int h1 = 0xefcdab89;
    unsigned int h2 = 0x98badcfe;
    unsigned int h3 = 0x10325476;
    unsigned int h4 = 0xc3d2e1f0;
    
    unsigned int a, b, c, d, e, ap, bp, cp, dp, ep;
    unsigned int x[16];
    int i, j;
    
    // Process 64-byte chunks
    for (i = 0; i + 64 <= len; i += 64) {
        // Load chunk ke x
        #pragma unroll
        for (j = 0; j < 16; j++) {
            x[j] = (data[i + j*4] << 24) | (data[i + j*4+1] << 16) | 
                   (data[i + j*4+2] << 8) | data[i + j*4+3];
        }
        
        a = h0; b = h1; c = h2; d = h3; e = h4;
        ap = h0; bp = h1; cp = h2; dp = h3; ep = h4;
        
        // 80 round utama
        #pragma unroll 16
        for (j = 0; j < 80; j++) {
            unsigned int t = rol(a + f(j, b, c, d) + x[d_ripemd160_r[j]] + d_ripemd160_k[j/16], d_ripemd160_s[j]) + e;
            a = e; e = d; d = rol(c, 10); c = b; b = t;
            
            t = rol(ap + f(79-j, bp, cp, dp) + x[d_ripemd160_rp[j]] + d_ripemd160_kp[j/16], d_ripemd160_sp[j]) + ep;
            ap = ep; ep = dp; dp = rol(cp, 10); cp = bp; bp = t;
        }
        
        unsigned int t = h1 + c + dp;
        h1 = h2 + d + ep;
        h2 = h3 + e + ap;
        h3 = h4 + a + bp;
        h4 = h0 + b + cp;
        h0 = t;
    }
    
    // Output dalam little-endian
    output[0] = h0 & 0xFF; output[1] = (h0 >> 8) & 0xFF; output[2] = (h0 >> 16) & 0xFF; output[3] = (h0 >> 24) & 0xFF;
    output[4] = h1 & 0xFF; output[5] = (h1 >> 8) & 0xFF; output[6] = (h1 >> 16) & 0xFF; output[7] = (h1 >> 24) & 0xFF;
    output[8] = h2 & 0xFF; output[9] = (h2 >> 8) & 0xFF; output[10] = (h2 >> 16) & 0xFF; output[11] = (h2 >> 24) & 0xFF;
    output[12] = h3 & 0xFF; output[13] = (h3 >> 8) & 0xFF; output[14] = (h3 >> 16) & 0xFF; output[15] = (h3 >> 24) & 0xFF;
    output[16] = h4 & 0xFF; output[17] = (h4 >> 8) & 0xFF; output[18] = (h4 >> 16) & 0xFF; output[19] = (h4 >> 24) & 0xFF;
}
