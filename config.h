// config.h - Configuration file for Tesla T4 optimization

#ifndef CONFIG_H
#define CONFIG_H

// Tesla T4 specific optimizations
#define T4_SM_COUNT 40
#define T4_MAX_THREADS_PER_SM 1024
#define T4_MAX_REGISTERS_PER_SM 65536
#define T4_SHARED_MEM_PER_SM 65536

// Kernel optimization parameters
#define OPTIMAL_THREADS_PER_BLOCK 256      // 8 warps
#define OPTIMAL_BLOCKS_PER_SM 4           // Maximum occupancy
#define OPTIMAL_REGISTERS_PER_THREAD 64    // Balance occupancy vs performance

// Memory optimization
#define USE_TEXTURE_MEMORY 1
#define USE_CONSTANT_MEMORY 1
#define USE_SHARED_MEMORY 1
#define SHARED_MEM_SIZE 49152              // 48KB for Tesla T4

// Performance tuning
#define ENABLE_WARP_LEVEL_OPTIMIZATIONS 1
#define ENABLE_LOOP_UNROLLING 1
#define ENABLE_PRECOMPUTED_TABLES 1
#define ENABLE_BATCH_PROCESSING 1
#define BATCH_SIZE 256

// Debug and profiling
#define ENABLE_PROFILING 0
#define ENABLE_DEBUG_OUTPUT 0
#define ENABLE_PERFORMANCE_COUNTERS 1

// Target configuration
#define MAX_TARGET_ADDRESSES 2048
#define HASH_CACHE_SIZE 1024

#endif // CONFIG_H
