# Makefile for Optimized Bitcoin Brute Force
CC = nvcc
CFLAGS = -O3 -arch=sm_75 --use_fast_math --ftz=true --prec-div=false --prec-sqrt=false \
         --fmad=true -Xptxas -O3,-v,-dlcm=cg -maxrregcount=64
CXXFLAGS = -O3 -march=native
LDFLAGS = -lcurand -lssl -lcrypto -lpthread
TARGET = btc_bruteforce_opt

# Tesla T4 specific optimizations
T4_FLAGS = --generate-code arch=compute_75,code=sm_75 \
           --ptxas-options=-v \
           --default-stream per-thread

# Default target
all: $(TARGET)

# Compile with Tesla T4 optimizations
$(TARGET): btc_bruteforce_optimized.cu
	$(CC) $(CFLAGS) $(T4_FLAGS) -Xcompiler="$(CXXFLAGS)" -o $@ $< $(LDFLAGS)

# Debug build
debug:
	$(CC) -G -g -arch=sm_75 -o $(TARGET)_debug btc_bruteforce_optimized.cu $(LDFLAGS)

# Profile build
profile:
	$(CC) $(CFLAGS) $(T4_FLAGS) --profile-all-functions -Xcompiler="$(CXXFLAGS)" -o $(TARGET)_profile btc_bruteforce_optimized.cu $(LDFLAGS)

# Clean
clean:
	rm -f $(TARGET) $(TARGET)_debug $(TARGET)_profile *.o found_keys.txt

# Run with performance monitoring
run: $(TARGET)
	nvprof --metrics all ./$(TARGET) rich.txt

# Quick run
quick: $(TARGET)
	./$(TARGET) rich.txt

# Install dependencies
install-deps:
	sudo apt-get update
	sudo apt-get install -y nvidia-cuda-toolkit nvidia-driver-525 libssl-dev build-essential

.PHONY: all debug profile clean run quick install-deps
