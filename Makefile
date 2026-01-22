# Makefile for Bitcoin CUDA Brute Force

# Compiler
NVCC = nvcc
CC = g++

# CUDA paths
CUDA_PATH = /usr/local/cuda
CUDA_INC = $(CUDA_PATH)/include
CUDA_LIB = $(CUDA_PATH)/lib64

# OpenSSL paths
OPENSSL_INC = /usr/include/openssl
OPENSSL_LIB = /usr/lib/x86_64-linux-gnu

# Compiler flags
NVCC_FLAGS = -O3 -arch=sm_70 --use_fast_math --ptxas-options=-v --maxrregcount=64
CC_FLAGS = -O3 -march=native -fopenmp

# Linker flags
LDFLAGS = -L$(CUDA_LIB) -L$(OPENSSL_LIB) -lcudart -lcuda -lcurand -lssl -lcrypto -fopenmp

# Target
TARGET = btc_bruteforce

# Source files
CUDA_SRCS = btc_bruteforce.cu secp256k1_cuda.cu sha256_cuda.cu ripemd160_cuda.cu base58_cuda.cu
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)

# Header files
HEADERS = secp256k1_cuda.h sha256_cuda.h ripemd160_cuda.h base58_cuda.h

# Default target
all: $(TARGET)

# Link target
$(TARGET): $(CUDA_OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

# Compile CUDA source files
%.o: %.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -I$(CUDA_INC) -I$(OPENSSL_INC) -c $< -o $@

# Clean
clean:
	rm -f $(TARGET) $(CUDA_OBJS) found_keys.txt

# Run with address list
run: $(TARGET)
	./$(TARGET) list.txt

# Performance test
benchmark: $(TARGET)
	nvprof ./$(TARGET) list.txt

# Install dependencies (Ubuntu/Debian)
install-deps:
	sudo apt-get update
	sudo apt-get install -y nvidia-cuda-toolkit nvidia-driver-525 libssl-dev build-essential

.PHONY: all clean run benchmark install-deps
