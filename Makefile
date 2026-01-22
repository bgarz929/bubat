# Makefile for Unlimited Bitcoin Brute Force
CC = nvcc
CFLAGS = -O3 -arch=sm_75 --use_fast_math -maxrregcount=64 --std=c++11
LIBS = -lcurand -lssl -lcrypto -lpthread
TARGET = btc_unlimited

all: $(TARGET)

$(TARGET): btc_bruteforce_unlimited.cu
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGET) found_keys.txt

run: $(TARGET)
	./$(TARGET) rich.txt

profile: $(TARGET)
	nvprof ./$(TARGET) rich.txt

debug:
	$(CC) -G -g -arch=sm_75 -o $(TARGET)_debug btc_bruteforce_unlimited.cu $(LIBS)

.PHONY: all clean run profile debug
