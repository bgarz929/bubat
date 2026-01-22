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
NVCC_FLAGS = -O3 -arch=sm_70 --use_fast_math --ptxas-options=-v --maxrregcount=64 -Xcompiler -fopenmp
CC_FLAGS = -O3 -march=native -fopenmp

# Linker flags
LDFLAGS = -L$(CUDA_LIB) -L$(OPENSSL_LIB) -lcudart -lcuda -lcurand -lssl -lcrypto -lpthread -fopenmp

# Target
TARGET = btc_bruteforce

# Source files (semua dalam satu file .cu untuk memudahkan)
SRC = btc_bruteforce.cu
EXEC = btc_bruteforce

# Default target
all: $(EXEC)

# Compile semua dalam satu langkah
$(EXEC): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -I$(CUDA_INC) -I$(OPENSSL_INC) -o $@ $< $(LDFLAGS)

# Clean
clean:
	rm -f $(EXEC) *.o found_keys.txt

# Run with address list
run: $(EXEC)
	./$(EXEC) list.txt

# Performance test
profile: $(EXEC)
	nvprof ./$(EXEC) list.txt

# Install dependencies (Ubuntu/Debian)
install-deps:
	sudo apt-get update
	sudo apt-get install -y nvidia-cuda-toolkit nvidia-driver-525 libssl-dev build-essential

.PHONY: all clean run profile install-deps
