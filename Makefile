# Makefile for Bitcoin CUDA Brute Force (Fixed)
CC = nvcc
CFLAGS = -O3 -arch=sm_70
LIBS = -lssl -lcrypto -lcurand
TARGET = btc_bruteforce

# Default target
all: $(TARGET)

# Compile
$(TARGET): btc_bruteforce_fixed.cu
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

# Clean
clean:
	rm -f $(TARGET) found_keys.txt

# Run
run: $(TARGET)
	./$(TARGET) list.txt

# Debug build
debug:
	$(CC) -G -g -arch=sm_70 -o $(TARGET)_debug btc_bruteforce_fixed.cu $(LIBS)

# Profile build
profile:
	$(CC) -pg -arch=sm_70 -o $(TARGET)_profile btc_bruteforce_fixed.cu $(LIBS)

.PHONY: all clean run debug profile
