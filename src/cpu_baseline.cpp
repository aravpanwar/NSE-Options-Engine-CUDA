#include <stdio.h>
#include <math.h>
#include <chrono>

struct OptionContract { float S, K, T, r, sigma; int type; };
struct Greeks { float price, delta, gamma, theta, vega; };

float normcdf_cpu(float x) {
    return 0.5f * erfcf(-x * 0.7071067811865476f);
}
float normpdf_cpu(float x) {
    return expf(-0.5f * x * x) * 0.3989422803f;
}

void bsm_cpu(const OptionContract& opt, Greeks& g) {
    float sqrtT = sqrtf(opt.T);
    float d1 = (logf(opt.S / opt.K) + (opt.r + 0.5f * opt.sigma * opt.sigma) * opt.T) / (opt.sigma * sqrtT);
    float d2 = d1 - opt.sigma * sqrtT;
    float Nd1 = normcdf_cpu(d1), Nd2 = normcdf_cpu(d2);
    float Nd1_ = normcdf_cpu(-d1), Nd2_ = normcdf_cpu(-d2);
    float nd1 = normpdf_cpu(d1);
    float disc = expf(-opt.r * opt.T);
    if (opt.type == 0) { g.price = opt.S * Nd1 - opt.K * disc * Nd2; g.delta = Nd1; }
    else               { g.price = opt.K * disc * Nd2_ - opt.S * Nd1_; g.delta = Nd1 - 1.0f; }
    g.gamma = nd1 / (opt.S * opt.sigma * sqrtT);
    g.theta = (-(opt.S * nd1 * opt.sigma) / (2.0f * sqrtT) - opt.r * opt.K * disc * (opt.type == 0 ? Nd2 : Nd2_)) / 365.0f;
    g.vega  = opt.S * nd1 * sqrtT * 0.01f;
}

int main() {
    const int N = 1000000;
    OptionContract* contracts = new OptionContract[N];
    Greeks*         results   = new Greeks[N];

    float spot = 22500.0f, baseStrike = 21000.0f, strikeStep = 50.0f;
    float r = 0.065f, sigma = 0.15f, T = 7.0f / 365.0f;
    for (int i = 0; i < N; i++) {
        contracts[i] = { spot, baseStrike + (i % (N/2)) * strikeStep, T, r, sigma + (i%10)*0.005f, (i < N/2)?0:1 };
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) bsm_cpu(contracts[i], results[i]);
    auto t2 = std::chrono::high_resolution_clock::now();

    float ms = std::chrono::duration<float, std::milli>(t2 - t1).count();

    printf("\n=== NSE Options Pricing Engine (CPU) ===\n");
    printf("Contracts priced : %d\n", N);
    printf("CPU time         : %.4f ms\n", ms);
    printf("Throughput       : %.2f million contracts/sec\n", (N / ms) / 1000.0f);

    delete[] contracts;
    delete[] results;
    return 0;
}
