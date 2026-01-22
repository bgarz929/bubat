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
# HAPUS -Xcompiler -fopenmp dari NVCC_FLAGS, pindah ke LDFLAGS
NVCC_FLAGS = -O3 -arch=sm_70 --use_fast_math --ptxas-options=-v --maxrregcount=64
CC_FLAGS = -O3 -march=native

# Linker flags
# Hanya -fopenmp di LDFLAGS untuk g++
LDFLAGS = -L$(CUDA_LIB) -L$(OPENSSL_LIB) -lcudart -lcuda -lcurand -lssl -lcrypto -lpthread

# Target
TARGET = btc_bruteforce

# Source files
SRC = btc_bruteforce.cu

# Default target
all: $(TARGET)

# Compile semua dalam satu langkah
$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -I$(CUDA_INC) -I$(OPENSSL_INC) -o $@ $< $(LDFLAGS)

# Clean
clean:
	rm -f $(TARGET) *.o found_keys.txt

# Run with address list
run: $(TARGET)
	./$(TARGET) list.txt

# Performance test
profile: $(TARGET)
	nvprof ./$(TARGET) list.txt

# Install dependencies (Ubuntu/Debian)
install-deps:
	sudo apt-get update
	sudo apt-get install -y nvidia-cuda-toolkit nvidia-driver-525 libssl-dev build-essential

.PHONY: all clean run profile install-deps
