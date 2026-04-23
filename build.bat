@echo off
echo Building GPU engine...
nvcc -o options_engine src/options_engine.cu -O2 -arch=sm_89 --use_fast_math

echo Building CPU baseline...
cl /O2 /EHsc /Fe:cpu_baseline.exe src\cpu_baseline.cpp

echo Building LSM GPU engine...
nvcc -o lsm_engine src/lsm_engine.cu -O2 -arch=sm_89 --use_fast_math

echo Building LSM CPU baseline...
cl /O2 /EHsc /Fe:lsm_cpu.exe src\lsm_cpu.cpp

echo.
echo Build complete.
