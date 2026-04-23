#!/bin/bash
# Linux build for all native targets. Invoked inside the Docker container.
set -e

mkdir -p bin

echo "[1/4] BSM CUDA..."
nvcc -o bin/options_engine src/options_engine.cu -O2 -arch=sm_89 --use_fast_math

echo "[2/4] BSM CPU..."
g++ -O2 -o bin/cpu_baseline src/cpu_baseline.cpp

echo "[3/4] LSM CUDA..."
nvcc -o bin/lsm_engine src/lsm_engine.cu -O2 -arch=sm_89 --use_fast_math

echo "[4/4] LSM CPU..."
g++ -O2 -o bin/lsm_cpu src/lsm_cpu.cpp

echo "Build complete."
