// Longstaff-Schwartz Monte Carlo pricer for American options, CUDA.
// Convention: type 0 = call, 1 = put.
//
// Two modes:
//   single-contract: legacy CLI flags (--S0, --K, ...); prints one result.
//   batch: --input <csv> [--output <csv>]; prices each row, writes a result row.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define CHK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ fprintf(stderr,"cuda error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

struct Params {
    float S0, K, r, sigma, T;
    int   steps;
    int   paths;
    int   type;              // 0 = call, 1 = put
    unsigned long long seed;
};

struct Buffers {
    float*        d_paths = nullptr;
    float*        d_V     = nullptr;   // doubles as OOS payoff buffer after Phase 1
    double*       d_sums  = nullptr;
    double*       d_sumV  = nullptr;
    unsigned int* d_nitm  = nullptr;
    float*        d_betas = nullptr;   // 3 * max_steps floats; policy coefficients per step
    int           max_paths = 0;
    int           max_steps = 0;
};

// -----------------------------------------------------------------------------
// Kernels
// -----------------------------------------------------------------------------

__global__ void sim_paths(float* paths, Params p, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.paths) return;

    curandStatePhilox4_32_10_t rng;
    curand_init(p.seed, i, 0, &rng);

    float drift     = (p.r - 0.5f * p.sigma * p.sigma) * dt;
    float diffusion =  p.sigma * sqrtf(dt);

    float S = p.S0;
    paths[i] = S;
    for (int t = 1; t <= p.steps; ++t) {
        float z = curand_normal(&rng);
        S *= expf(drift + diffusion * z);
        paths[t * p.paths + i] = S;
    }
}

__device__ __host__ inline float payoff(float S, float K, int type) {
    return (type == 0) ? fmaxf(S - K, 0.f) : fmaxf(K - S, 0.f);
}

__global__ void terminal(const float* paths, float* V, Params p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.paths) return;
    V[i] = payoff(paths[p.steps * p.paths + i], p.K, p.type);
}

__global__ void collect(const float* paths, float* V, Params p, int t,
                        float df, double* sums, unsigned int* nitm)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.paths) return;

    V[i] *= df;

    float S  = paths[t * p.paths + i];
    float ex = payoff(S, p.K, p.type);
    if (ex <= 0.f) return;

    double s  = (double)S / (double)p.K;
    double s2 = s  * s;
    double s3 = s2 * s;
    double s4 = s3 * s;
    double y  = (double)V[i];

    atomicAdd(&sums[0], s);
    atomicAdd(&sums[1], s2);
    atomicAdd(&sums[2], s3);
    atomicAdd(&sums[3], s4);
    atomicAdd(&sums[4], y);
    atomicAdd(&sums[5], s  * y);
    atomicAdd(&sums[6], s2 * y);
    atomicAdd(nitm, 1u);
}

__global__ void apply_exercise(const float* paths, float* V, Params p, int t,
                               float b0, float b1, float b2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.paths) return;

    float S  = paths[t * p.paths + i];
    float ex = payoff(S, p.K, p.type);
    if (ex <= 0.f) return;

    float s    = S / p.K;
    float cont = b0 + b1 * s + b2 * s * s;
    if (ex > cont) V[i] = ex;
}

__global__ void discount_all(float* V, int n, float df) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) V[i] *= df;
}

__global__ void reduce_sum(const float* V, int n, double* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(out, (double)V[i]);
}

// Phase 2 (Rasmussen 2005 out-of-sample valuation). Regenerates a fresh path
// set with a seed independent from Phase 1, walks forward applying the policy
// betas[t] learned during the backward sweep, and records the first profitable
// exercise payoff (or the terminal payoff if never exercised), discounted to
// t=0. One thread per path; no intermediate path buffer.
__global__ void value_oos(Params p, float dt,
                          const float* betas, float* payoffs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.paths) return;

    curandStatePhilox4_32_10_t rng;
    curand_init(p.seed ^ 0xDEADBEEFDEADBEEFULL, i, 0, &rng);

    float drift     = (p.r - 0.5f * p.sigma * p.sigma) * dt;
    float diffusion =  p.sigma * sqrtf(dt);

    float S = p.S0;
    for (int t = 1; t < p.steps; ++t) {
        float z = curand_normal(&rng);
        S *= expf(drift + diffusion * z);

        float ex = payoff(S, p.K, p.type);
        if (ex > 0.f) {
            float s  = S / p.K;
            float b0 = betas[t*3 + 0];
            float b1 = betas[t*3 + 1];
            float b2 = betas[t*3 + 2];
            float cont = b0 + b1 * s + b2 * s * s;
            if (ex > cont) {
                payoffs[i] = ex * expf(-p.r * (float)t * dt);
                return;
            }
        }
    }
    // Never exercised early; take one final step to expiry and collect terminal payoff.
    float z = curand_normal(&rng);
    S *= expf(drift + diffusion * z);
    float ex = payoff(S, p.K, p.type);
    payoffs[i] = ex * expf(-p.r * (float)p.steps * dt);
}

// -----------------------------------------------------------------------------
// Host helpers
// -----------------------------------------------------------------------------

static bool solve3(const double* s, unsigned int n, float beta[3]) {
    if (n < 3) return false;

    double A00 = (double)n, A01 = s[0], A02 = s[1];
    double                  A11 = s[1], A12 = s[2];
    double                              A22 = s[3];
    double b0  = s[4], b1 = s[5], b2 = s[6];

    double c00 = A11*A22 - A12*A12;
    double c01 = A12*A02 - A01*A22;
    double c02 = A01*A12 - A11*A02;
    double c11 = A00*A22 - A02*A02;
    double c12 = A01*A02 - A00*A12;
    double c22 = A00*A11 - A01*A01;

    double det = A00*c00 + A01*c01 + A02*c02;
    if (fabs(det) < 1e-20) return false;
    double inv = 1.0 / det;

    beta[0] = (float)(inv * (c00*b0 + c01*b1 + c02*b2));
    beta[1] = (float)(inv * (c01*b0 + c11*b1 + c12*b2));
    beta[2] = (float)(inv * (c02*b0 + c12*b1 + c22*b2));
    return true;
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

// -----------------------------------------------------------------------------
// Core pricing loop. Reuses pre-allocated buffers; no cudaMalloc per call.
// -----------------------------------------------------------------------------

struct PriceResult {
    double american;      // out-of-sample (Rasmussen 2005), primary
    double american_is;   // in-sample LSM estimate, kept for comparison
    double european;
    float  gpu_ms;
};

static PriceResult price_one(const Params& p, Buffers& b) {
    float dt = p.T / p.steps;
    float df = expf(-p.r * dt);

    int threads = 256;
    int blocks  = (p.paths + threads - 1) / threads;

    // Default policy: cont = 1e10 means we never exercise at that step. Overwritten
    // when solve3 succeeds. Size 3*steps so we can index by t directly.
    std::vector<float> host_betas((size_t)3 * p.steps, 0.f);
    for (int t = 0; t < p.steps; ++t) host_betas[(size_t)t*3 + 0] = 1e10f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Phase 1: backward sweep. Fits the exercise policy and produces the
    // in-sample LSM estimate as a byproduct.
    sim_paths<<<blocks, threads>>>(b.d_paths, p, dt);
    terminal <<<blocks, threads>>>(b.d_paths, b.d_V, p);

    double       h_sums[7];
    unsigned int h_nitm;
    float        beta[3];

    for (int t = p.steps - 1; t >= 1; --t) {
        CHK(cudaMemset(b.d_sums, 0, 7 * sizeof(double)));
        CHK(cudaMemset(b.d_nitm, 0, sizeof(unsigned int)));

        collect<<<blocks, threads>>>(b.d_paths, b.d_V, p, t, df, b.d_sums, b.d_nitm);

        CHK(cudaMemcpy(h_sums, b.d_sums, 7 * sizeof(double), cudaMemcpyDeviceToHost));
        CHK(cudaMemcpy(&h_nitm, b.d_nitm, sizeof(unsigned int), cudaMemcpyDeviceToHost));

        if (solve3(h_sums, h_nitm, beta)) {
            apply_exercise<<<blocks, threads>>>(b.d_paths, b.d_V, p, t,
                                                beta[0], beta[1], beta[2]);
            host_betas[(size_t)t*3 + 0] = beta[0];
            host_betas[(size_t)t*3 + 1] = beta[1];
            host_betas[(size_t)t*3 + 2] = beta[2];
        }
    }

    // In-sample: discount once more, reduce. This is V_IS (the Phase 1 estimate).
    discount_all<<<blocks, threads>>>(b.d_V, p.paths, df);
    CHK(cudaMemset(b.d_sumV, 0, sizeof(double)));
    reduce_sum<<<blocks, threads>>>(b.d_V, p.paths, b.d_sumV);
    double h_sumV_is = 0.0;
    CHK(cudaMemcpy(&h_sumV_is, b.d_sumV, sizeof(double), cudaMemcpyDeviceToHost));

    // Phase 2: out-of-sample valuation with the fitted policy on fresh paths.
    CHK(cudaMemcpy(b.d_betas, host_betas.data(),
                   3 * (size_t)p.steps * sizeof(float), cudaMemcpyHostToDevice));
    value_oos<<<blocks, threads>>>(p, dt, b.d_betas, b.d_V);
    CHK(cudaMemset(b.d_sumV, 0, sizeof(double)));
    reduce_sum<<<blocks, threads>>>(b.d_V, p.paths, b.d_sumV);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);

    double h_sumV_oos = 0.0;
    CHK(cudaMemcpy(&h_sumV_oos, b.d_sumV, sizeof(double), cudaMemcpyDeviceToHost));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    PriceResult r;
    r.american     = h_sumV_oos / p.paths;
    r.american_is  = h_sumV_is  / p.paths;
    r.european     = bsm_european(p.S0, p.K, p.T, p.r, p.sigma, p.type);
    r.gpu_ms       = ms;
    return r;
}

static void alloc_buffers(Buffers& b, int max_paths, int max_steps) {
    size_t paths_bytes = (size_t)max_paths * (max_steps + 1) * sizeof(float);
    CHK(cudaMalloc(&b.d_paths, paths_bytes));
    CHK(cudaMalloc(&b.d_V,     max_paths * sizeof(float)));
    CHK(cudaMalloc(&b.d_sums,  7 * sizeof(double)));
    CHK(cudaMalloc(&b.d_nitm,  sizeof(unsigned int)));
    CHK(cudaMalloc(&b.d_sumV,  sizeof(double)));
    CHK(cudaMalloc(&b.d_betas, 3 * (size_t)max_steps * sizeof(float)));
    b.max_paths = max_paths;
    b.max_steps = max_steps;
}

static void free_buffers(Buffers& b) {
    cudaFree(b.d_paths);
    cudaFree(b.d_V);
    cudaFree(b.d_sums);
    cudaFree(b.d_nitm);
    cudaFree(b.d_sumV);
    cudaFree(b.d_betas);
}

// -----------------------------------------------------------------------------
// CSV I/O for batch mode
// -----------------------------------------------------------------------------

static std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : line) {
        if (c == ',') { out.push_back(cur); cur.clear(); }
        else if (c != '\r') { cur.push_back(c); }
    }
    out.push_back(cur);
    return out;
}

struct Row {
    std::string symbol, expiry, type_str;
    float spot, strike, T, r, sigma;
};

static std::vector<Row> read_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    std::string line;
    if (!std::getline(f, line)) { fprintf(stderr, "empty CSV\n"); exit(1); }

    auto hdr = split_csv(line);
    std::unordered_map<std::string, int> col;
    for (int i = 0; i < (int)hdr.size(); ++i) col[hdr[i]] = i;

    auto req = [&](const char* k) -> int {
        auto it = col.find(k);
        if (it == col.end()) { fprintf(stderr, "missing column: %s\n", k); exit(1); }
        return it->second;
    };
    int c_sym = req("symbol"), c_spot = req("spot"), c_strike = req("strike"),
        c_exp = req("expiry"), c_T = req("T"), c_r = req("r"),
        c_sig = req("sigma"), c_type = req("type");

    std::vector<Row> rows;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto p = split_csv(line);
        Row r;
        r.symbol   = p[c_sym];
        r.spot     = std::stof(p[c_spot]);
        r.strike   = std::stof(p[c_strike]);
        r.expiry   = p[c_exp];
        r.T        = std::stof(p[c_T]);
        r.r        = std::stof(p[c_r]);
        r.sigma    = std::stof(p[c_sig]);
        r.type_str = p[c_type];
        rows.push_back(r);
    }
    return rows;
}

// -----------------------------------------------------------------------------
// CLI
// -----------------------------------------------------------------------------

struct Cfg {
    Params p;
    std::string input;
    std::string output;
};

static void parse_args(int argc, char** argv, Cfg& c) {
    c.p.S0 = 40.f; c.p.K = 40.f; c.p.r = 0.06f; c.p.sigma = 0.20f; c.p.T = 1.0f;
    c.p.steps = 50; c.p.paths = 100000; c.p.type = 1; c.p.seed = 42ULL;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--S0"     && i+1 < argc) c.p.S0    = (float)atof(argv[++i]);
        else if (a == "--K"      && i+1 < argc) c.p.K     = (float)atof(argv[++i]);
        else if (a == "--r"      && i+1 < argc) c.p.r     = (float)atof(argv[++i]);
        else if (a == "--sigma"  && i+1 < argc) c.p.sigma = (float)atof(argv[++i]);
        else if (a == "--T"      && i+1 < argc) c.p.T     = (float)atof(argv[++i]);
        else if (a == "--steps"  && i+1 < argc) c.p.steps = atoi(argv[++i]);
        else if (a == "--paths"  && i+1 < argc) c.p.paths = atoi(argv[++i]);
        else if (a == "--seed"   && i+1 < argc) c.p.seed  = strtoull(argv[++i], nullptr, 10);
        else if (a == "--type"   && i+1 < argc) {
            std::string v = argv[++i];
            c.p.type = (v == "call" || v == "CALL") ? 0 : 1;
        }
        else if (a == "--input"  && i+1 < argc) c.input   = argv[++i];
        else if (a == "--output" && i+1 < argc) c.output  = argv[++i];
    }
}

// -----------------------------------------------------------------------------
// Entry points
// -----------------------------------------------------------------------------

static int run_single(const Cfg& c) {
    printf("\n=== Longstaff-Schwartz American Option Pricer (CUDA) ===\n");
    printf("  S0=%.4f  K=%.4f  r=%.4f  sigma=%.4f  T=%.4f\n",
           c.p.S0, c.p.K, c.p.r, c.p.sigma, c.p.T);
    printf("  steps=%d  paths=%d  type=%s  seed=%llu\n\n",
           c.p.steps, c.p.paths, c.p.type == 0 ? "CALL" : "PUT", c.p.seed);

    Buffers b;
    alloc_buffers(b, c.p.paths, c.p.steps);
    PriceResult r = price_one(c.p, b);
    free_buffers(b);

    printf("Results:\n");
    printf("  American (OOS)    : %.6f   <- primary, Rasmussen 2005 out-of-sample\n", r.american);
    printf("  American (IS)     : %.6f   <- in-sample LSM, shown for comparison\n", r.american_is);
    printf("  European BSM      : %.6f\n", r.european);
    printf("  Early-ex premium  : %.6f   (OOS - European)\n", r.american - r.european);
    printf("  IS - OOS          : %.6f   (size of in-sample optimism bias)\n", r.american_is - r.american);
    printf("  GPU time          : %.2f ms  (both phases)\n", r.gpu_ms);
    printf("  Throughput        : %.2f M paths/sec  (2x %d paths total)\n\n",
           (2.0f * c.p.paths / r.gpu_ms) / 1000.0f, c.p.paths);
    return 0;
}

static int run_batch(const Cfg& c) {
    auto rows = read_csv(c.input);
    if (rows.empty()) { fprintf(stderr, "no rows in %s\n", c.input.c_str()); return 1; }

    std::string out = c.output.empty() ? "results.csv" : c.output;
    FILE* fo = fopen(out.c_str(), "w");
    if (!fo) { fprintf(stderr, "cannot open %s for write\n", out.c_str()); return 1; }
    fprintf(fo, "symbol,spot,strike,expiry,T,r,sigma,type,american,american_is,european,premium,gpu_ms\n");

    Buffers b;
    alloc_buffers(b, c.p.paths, c.p.steps);

    printf("\n=== LSM batch mode ===\n");
    printf("  input=%s  rows=%zu  paths=%d  steps=%d\n\n",
           c.input.c_str(), rows.size(), c.p.paths, c.p.steps);

    double total_ms = 0.0;
    for (size_t i = 0; i < rows.size(); ++i) {
        const Row& row = rows[i];
        Params p = c.p;
        p.S0    = row.spot;
        p.K     = row.strike;
        p.T     = row.T;
        p.r     = row.r;
        p.sigma = row.sigma;
        p.type  = (row.type_str == "call" || row.type_str == "CALL") ? 0 : 1;
        p.seed  = c.p.seed + (unsigned long long)i;

        PriceResult r = price_one(p, b);
        total_ms += r.gpu_ms;

        fprintf(fo, "%s,%.4f,%.4f,%s,%.6f,%.6f,%.6f,%s,%.6f,%.6f,%.6f,%.6f,%.2f\n",
                row.symbol.c_str(), row.spot, row.strike, row.expiry.c_str(),
                row.T, row.r, row.sigma, row.type_str.c_str(),
                r.american, r.american_is, r.european,
                r.american - r.european, r.gpu_ms);

        if ((i + 1) % 25 == 0 || i + 1 == rows.size()) {
            printf("  [%4zu/%zu]  %-10s  K=%7.1f  T=%.3f  type=%-4s  "
                   "amer=%.4f  eur=%.4f  prem=%.4f\n",
                   i + 1, rows.size(), row.symbol.c_str(), row.strike, row.T,
                   row.type_str.c_str(), r.american, r.european,
                   r.american - r.european);
        }
    }
    fclose(fo);
    free_buffers(b);

    printf("\nWrote %zu rows to %s\n", rows.size(), out.c_str());
    printf("Total GPU time: %.2f s  (avg %.2f ms/contract)\n",
           total_ms / 1000.0, total_ms / rows.size());
    return 0;
}

int main(int argc, char** argv) {
    Cfg c;
    parse_args(argc, argv, c);

    if (!c.input.empty()) return run_batch(c);
    return run_single(c);
}
