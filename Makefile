# Makefile for Optimized Bitcoin Brute Force
CC = nvcc
CFLAGS = -O3 -arch=sm_75 --use_fast_math --ftz=true --prec-div=false --prec-sqrt=false \
         --fmad=true -Xptxas -O3,-v,-dlcm=cg -maxrregcount=64 -Xcompiler -O3,-march=native,-fopenmp
LDFLAGS = -lcurand -lssl -lcrypto -lpthread -fopenmp
TARGET = btc_bruteforce_opt

# Tesla T4 specific optimizations
T4_FLAGS = --generate-code arch=compute_75,code=sm_75 \
           --ptxas-options=-v \
           --default-stream per-thread

# Performance profiling flags
PROFILE_FLAGS = --profile-all-functions --export-profile

# Default target
all: $(TARGET)

# Compile with Tesla T4 optimizations
$(TARGET): btc_bruteforce_optimized.cu
	$(CC) $(CFLAGS) $(T4_FLAGS) -o $@ $< $(LDFLAGS)

# Debug build
debug:
	$(CC) -G -g -arch=sm_75 -o $(TARGET)_debug btc_bruteforce_optimized.cu $(LDFLAGS)

# Profile build
profile:
	$(CC) $(CFLAGS) $(T4_FLAGS) $(PROFILE_FLAGS) -o $(TARGET)_profile btc_bruteforce_optimized.cu $(LDFLAGS)

# Multi-GPU support (if available)
multi-gpu:
	$(CC) $(CFLAGS) -arch=sm_75 -DUSE_MULTI_GPU -o $(TARGET)_multi btc_bruteforce_optimized.cu $(LDFLAGS)

# Clean
clean:
	rm -f $(TARGET) $(TARGET)_debug $(TARGET)_profile $(TARGET)_multi *.o found_keys.txt

# Run with performance monitoring
run: $(TARGET)
	nvprof --metrics all ./$(TARGET) rich.txt

# Run continuously
continuous: $(TARGET)
	./continuous_run.sh

# Install dependencies
install-deps:
	sudo apt-get update
	sudo apt-get install -y nvidia-cuda-toolkit nvidia-driver-525 libssl-dev build-essential

.PHONY: all debug profile multi-gpu clean run continuous install-deps
