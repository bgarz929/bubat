# Makefile for Bitcoin Brute Force (Fixed Output)
CC = nvcc
CFLAGS = -O3 -arch=sm_75 --use_fast_math -maxrregcount=64
LIBS = -lcurand -lssl -lcrypto
TARGET = btc_bruteforce_final

all: $(TARGET)

$(TARGET): btc_bruteforce_final.cu
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGET) found_keys.txt

run: $(TARGET)
	./$(TARGET) rich.txt

profile: $(TARGET)
	nvprof ./$(TARGET) rich.txt

debug:
	$(CC) -G -g -arch=sm_75 -o $(TARGET)_debug btc_bruteforce_final.cu $(LIBS)

.PHONY: all clean run profile debug
