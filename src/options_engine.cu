#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

struct OptionContract {
    float S, K, T, r, sigma;
    int type;
};

struct Greeks {
    float price, delta, gamma, theta, vega;
};

__device__ float normcdf_device(float x) { return normcdff(x); }
__device__ float normpdf_device(float x) { return expf(-0.5f * x * x) * 0.3989422803f; }

__global__ void bsm_kernel(const OptionContract* contracts, Greeks* results, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    OptionContract opt = contracts[idx];
    float sqrtT = sqrtf(opt.T);
    float d1 = (logf(opt.S / opt.K) + (opt.r + 0.5f * opt.sigma * opt.sigma) * opt.T) / (opt.sigma * sqrtT);
    float d2 = d1 - opt.sigma * sqrtT;
    float Nd1 = normcdf_device(d1), Nd2 = normcdf_device(d2);
    float Nd1_ = normcdf_device(-d1), Nd2_ = normcdf_device(-d2);
    float nd1 = normpdf_device(d1);
    float disc = expf(-opt.r * opt.T);
    Greeks g;
    if (opt.type == 0) { g.price = opt.S * Nd1 - opt.K * disc * Nd2; g.delta = Nd1; }
    else               { g.price = opt.K * disc * Nd2_ - opt.S * Nd1_; g.delta = Nd1 - 1.0f; }
    g.gamma = nd1 / (opt.S * opt.sigma * sqrtT);
    g.theta = (-(opt.S * nd1 * opt.sigma) / (2.0f * sqrtT) - opt.r * opt.K * disc * (opt.type == 0 ? Nd2 : Nd2_)) / 365.0f;
    g.vega  = opt.S * nd1 * sqrtT * 0.01f;
    results[idx] = g;
}

void generate_nse_chain(OptionContract* contracts, int n) {
    float spot = 22500.0f, baseStrike = 21000.0f, strikeStep = 50.0f;
    float r = 0.065f, sigma = 0.15f, T = 7.0f / 365.0f;
    for (int i = 0; i < n; i++) {
        contracts[i].S     = spot;
        contracts[i].K     = baseStrike + (i % (n / 2)) * strikeStep;
        contracts[i].T     = T;
        contracts[i].r     = r;
        contracts[i].sigma = sigma + (i % 10) * 0.005f;
        contracts[i].type  = (i < n / 2) ? 0 : 1;
    }
}

int main() {
    const int N = 1000000;
    OptionContract* h_contracts = new OptionContract[N];
    Greeks*         h_results   = new Greeks[N];
    generate_nse_chain(h_contracts, N);

    OptionContract* d_contracts;
    Greeks*         d_results;
    cudaMalloc(&d_contracts, N * sizeof(OptionContract));
    cudaMalloc(&d_results,   N * sizeof(Greeks));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemcpy(d_contracts, h_contracts, N * sizeof(OptionContract), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    cudaEventRecord(start);
    bsm_kernel<<<blocks, threads>>>(d_contracts, d_results, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(h_results, d_results, N * sizeof(Greeks), cudaMemcpyDeviceToHost);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("\n=== NSE Options Pricing Engine (CUDA) ===\n");
    printf("Contracts priced : %d\n", N);
    printf("GPU kernel time  : %.4f ms\n", ms);
    printf("Throughput       : %.2f million contracts/sec\n\n", (N / ms) / 1000.0f);

    printf("%-6s %-8s %-8s %-8s %-8s %-8s %-8s\n", "Idx", "Strike", "Type", "Price", "Delta", "Gamma", "Vega");
    printf("-----------------------------------------------------\n");
    for (int i = 0; i < 5; i++) {
        printf("%-6d %-8.0f %-8s %-8.2f %-8.4f %-8.6f %-8.4f\n",
               i, h_contracts[i].K,
               h_contracts[i].type == 0 ? "CALL" : "PUT",
               h_results[i].price, h_results[i].delta,
               h_results[i].gamma, h_results[i].vega);
    }

    cudaFree(d_contracts);
    cudaFree(d_results);
    delete[] h_contracts;
    delete[] h_results;
    return 0;
}
