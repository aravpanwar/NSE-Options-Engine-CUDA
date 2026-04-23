// Sequential CPU baseline for the Longstaff-Schwartz MC pricer.
// Mirrors lsm_engine.cu step for step so the GPU speedup number is honest.
// Convention: type 0 = call, 1 = put.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string>
#include <vector>
#include <random>
#include <chrono>

struct Params {
    float S0, K, r, sigma, T;
    int   steps;
    int   paths;
    int   type;
    unsigned long long seed;
};

static inline float payoff(float S, float K, int type) {
    return (type == 0) ? fmaxf(S - K, 0.f) : fmaxf(K - S, 0.f);
}

static double bsm_european(double S, double K, double T, double r, double sigma, int type) {
    double sqrtT = sqrt(T);
    double d1 = (log(S/K) + (r + 0.5*sigma*sigma) * T) / (sigma * sqrtT);
    double d2 = d1 - sigma * sqrtT;
    double N1 = 0.5 * erfc(-d1 / sqrt(2.0));
    double N2 = 0.5 * erfc(-d2 / sqrt(2.0));
    if (type == 0) return S * N1 - K * exp(-r*T) * N2;
    return K * exp(-r*T) * (1.0 - N2) - S * (1.0 - N1);
}

static bool solve3(double a01, double a02, double a12, double a22,
                   double b0,  double b1,  double b2,  unsigned int n,
                   float beta[3])
{
    if (n < 3) return false;
    double A00 = (double)n;
    double c00 = a02*a22 - a12*a12;
    double c01 = a12*a02 - a01*a22;
    double c02 = a01*a12 - a02*a02;
    double c11 = A00*a22 - a02*a02;
    double c12 = a01*a02 - A00*a12;
    double c22 = A00*a02 - a01*a01;

    double det = A00*c00 + a01*c01 + a02*c02;
    if (fabs(det) < 1e-20) return false;
    double inv = 1.0 / det;
    beta[0] = (float)(inv * (c00*b0 + c01*b1 + c02*b2));
    beta[1] = (float)(inv * (c01*b0 + c11*b1 + c12*b2));
    beta[2] = (float)(inv * (c02*b0 + c12*b1 + c22*b2));
    return true;
}

static void parse_args(int argc, char** argv, Params& p) {
    p.S0 = 40.f; p.K = 40.f; p.r = 0.06f; p.sigma = 0.20f; p.T = 1.0f;
    p.steps = 50; p.paths = 100000; p.type = 1; p.seed = 42ULL;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--S0"    && i+1 < argc) p.S0    = (float)atof(argv[++i]);
        else if (a == "--K"     && i+1 < argc) p.K     = (float)atof(argv[++i]);
        else if (a == "--r"     && i+1 < argc) p.r     = (float)atof(argv[++i]);
        else if (a == "--sigma" && i+1 < argc) p.sigma = (float)atof(argv[++i]);
        else if (a == "--T"     && i+1 < argc) p.T     = (float)atof(argv[++i]);
        else if (a == "--steps" && i+1 < argc) p.steps = atoi(argv[++i]);
        else if (a == "--paths" && i+1 < argc) p.paths = atoi(argv[++i]);
        else if (a == "--seed"  && i+1 < argc) p.seed  = strtoull(argv[++i], nullptr, 10);
        else if (a == "--type"  && i+1 < argc) {
            std::string v = argv[++i];
            p.type = (v == "call" || v == "CALL") ? 0 : 1;
        }
    }
}

int main(int argc, char** argv) {
    Params p;
    parse_args(argc, argv, p);

    float dt = p.T / p.steps;
    float df = expf(-p.r * dt);

    printf("\n=== Longstaff-Schwartz American Option Pricer (CPU) ===\n");
    printf("  S0=%.4f  K=%.4f  r=%.4f  sigma=%.4f  T=%.4f\n",
           p.S0, p.K, p.r, p.sigma, p.T);
    printf("  steps=%d  paths=%d  type=%s  seed=%llu\n\n",
           p.steps, p.paths, p.type == 0 ? "CALL" : "PUT", p.seed);

    auto t_start = std::chrono::high_resolution_clock::now();

    std::vector<float> paths((size_t)p.paths * (p.steps + 1));
    std::vector<float> V(p.paths);

    std::mt19937_64 rng(p.seed);
    std::normal_distribution<float> norm(0.0f, 1.0f);
    float drift     = (p.r - 0.5f * p.sigma * p.sigma) * dt;
    float diffusion =  p.sigma * sqrtf(dt);

    for (int i = 0; i < p.paths; ++i) {
        float S = p.S0;
        paths[i] = S;
        for (int t = 1; t <= p.steps; ++t) {
            float z = norm(rng);
            S *= expf(drift + diffusion * z);
            paths[(size_t)t * p.paths + i] = S;
        }
    }

    for (int i = 0; i < p.paths; ++i) {
        float S_T = paths[(size_t)p.steps * p.paths + i];
        V[i] = payoff(S_T, p.K, p.type);
    }

    for (int t = p.steps - 1; t >= 1; --t) {
        double a01 = 0, a02 = 0, a12 = 0, a22 = 0;
        double b0  = 0, b1  = 0, b2  = 0;
        unsigned int nitm = 0;

        for (int i = 0; i < p.paths; ++i) {
            V[i] *= df;
            float S  = paths[(size_t)t * p.paths + i];
            float ex = payoff(S, p.K, p.type);
            if (ex <= 0.f) continue;
            double s  = (double)S / (double)p.K;
            double s2 = s * s;
            double y  = (double)V[i];
            a01 += s;
            a02 += s2;
            a12 += s2 * s;
            a22 += s2 * s2;
            b0  += y;
            b1  += s  * y;
            b2  += s2 * y;
            nitm++;
        }

        float beta[3];
        if (!solve3(a01, a02, a12, a22, b0, b1, b2, nitm, beta)) continue;

        for (int i = 0; i < p.paths; ++i) {
            float S  = paths[(size_t)t * p.paths + i];
            float ex = payoff(S, p.K, p.type);
            if (ex <= 0.f) continue;
            float s    = S / p.K;
            float cont = beta[0] + beta[1] * s + beta[2] * s * s;
            if (ex > cont) V[i] = ex;
        }
    }

    double sumV = 0.0;
    for (int i = 0; i < p.paths; ++i) {
        V[i] *= df;
        sumV += V[i];
    }
    double american = sumV / p.paths;
    double european = bsm_european(p.S0, p.K, p.T, p.r, p.sigma, p.type);

    auto t_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    printf("Results:\n");
    printf("  American price    : %.6f\n", american);
    printf("  European BSM      : %.6f\n", european);
    printf("  Early-ex premium  : %.6f\n", american - european);
    printf("  CPU time          : %.2f ms\n", ms);
    printf("  Throughput        : %.3f M paths/sec\n\n",
           (p.paths / ms) / 1000.0f);
    return 0;
}
