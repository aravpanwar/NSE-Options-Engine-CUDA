@echo off
echo Building GPU engine...
nvcc -o options_engine src/options_engine.cu -O2 -arch=sm_89 --use_fast_math

echo Building CPU baseline...
cl /O2 /EHsc /Fe:cpu_baseline.exe src\cpu_baseline.cpp

echo.
echo Build complete.
