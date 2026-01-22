#!/bin/bash

# Script untuk menjalankan Bitcoin Brute Force

echo "=== Bitcoin Private Key Brute Force with CUDA ==="
echo

# Check dependencies
if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA compiler (nvcc) not found!"
    echo "Please install CUDA toolkit first."
    exit 1
fi

if ! command -v g++ &> /dev/null; then
    echo "Error: g++ compiler not found!"
    exit 1
fi

# Check for GPU
if ! nvidia-smi &> /dev/null; then
    echo "Error: NVIDIA GPU not found or driver not installed!"
    exit 1
fi

# Compile
echo "Compiling..."
make clean
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Compilation successful!"
echo

# Check for address list
if [ ! -f "list.txt" ]; then
    echo "Creating sample address list..."
    cat > list.txt << EOF
1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa
1BitcoinEaterAddressDontSendf59kuE
1FeexV6bAHb8ybZjqQMjJrcCrHGW9sb6uF
13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so
1BY8GQbnueYofwSuFAT3USAhGjPrkxDdW9
EOF
    echo "Sample list.txt created with 5 addresses."
fi

# Display GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo
echo "Starting brute force..."
echo "Press Ctrl+C to stop."
echo

# Run the program
./btc_bruteforce list.txt

# Cleanup
make clean
