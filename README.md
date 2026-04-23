# NSE Options Pricing Engine

GPU-accelerated option pricer for Indian equity derivatives. Two engines share this repo:

- **BSM kernel** (Phase 0): closed-form Black-Scholes-Merton pricing and Greeks for the full NIFTY chain. One thread per contract, 1M contracts in 0.49 ms on an RTX 4050.
- **Longstaff-Schwartz engine** (Phase 1 + 2): American option pricing via Monte Carlo for NSE single stocks. Validated against Longstaff & Schwartz 2001 Table 1. Ships with a yfinance-driven pipeline that fetches real spot and realized volatility for five NSE tickers, prices a strike x expiry grid, and renders a 3D dashboard.

## Results

LSM validation (LS 2001 Table 1 base case: American put, S = K = 40, r = 6%, sigma = 20%, T = 1, 50 steps):

| | Reference (FD) | GPU OOS | GPU IS |
| --- | --- | --- | --- |
| American price | 2.313 | 2.298 | 2.316 |
| European BSM | 2.066 | 2.066 | 2.066 |
| Early-exercise premium | 0.247 | 0.232 | 0.249 |

Two estimators are reported. **OOS** (out-of-sample, Rasmussen 2005) is the primary: the exercise policy is fit on one path set, the price is evaluated on a disjoint set. This gives an honest lower bound on the American value and eliminates in-sample optimism. **IS** (in-sample LSM) is what Longstaff-Schwartz 2001 originally reported; it uses the same paths for policy and valuation. Both land within MC noise of the finite-difference reference.

LSM throughput at 1M paths per contract (50 exercise dates):

| | Time | Throughput |
| --- | --- | --- |
| CPU (single thread) | 3743 ms | 0.27 M paths/s |
| GPU (RTX 4050) | 508 ms | 1.97 M paths/s |

Speedup is **7.4x**. The host is in the critical path once per backward step for the 3x3 regression solve, which is the dominant bottleneck. Eliminating that roundtrip is the main Phase 3 target.

BSM benchmark (1M synthetic NIFTY contracts, embarrassingly parallel):

| | Time | Throughput |
| --- | --- | --- |
| CPU (single thread) | 130.6 ms | 7.7 M/s |
| GPU (RTX 4050) | 0.485 ms | 2062 M/s |

Speedup is **269x**. The kernel is pure closed-form evaluation with no synchronization.

## Pipeline

```
python fetch_nse_stocks.py
    -> data/contracts.csv       (~780 contracts: 5 stocks x ~27 strikes x 3 expiries x {call,put})

.\run.ps1 lsm_engine --input data/contracts.csv --output data/results.csv
    -> data/results.csv         (adds american, european, premium, gpu_ms)

python plot_american.py
    -> plots/dashboard.html     (3D premium surfaces per stock, heatmap, summary)
```

The default ticker set is RELIANCE, HDFCBANK, TCS, INFY, ICICIBANK. Override with `--tickers` on `fetch_nse_stocks.py`.

## Requirements

**Hardware**
- NVIDIA GPU with CUDA compute capability 6.0 or newer (built for sm_89 / Ada Lovelace by default; edit `build.sh` or `build.bat` to retarget)
- At least 2 GB VRAM for the default 100k paths; 6 GB for 10M paths
- x86_64 CPU with AVX2 for the CPU baseline

**Software (Docker path, works on any host including WDAC-locked)**
- Docker Desktop 20+ with WSL2 backend
- NVIDIA driver 470+ on Windows host (CUDA-on-WSL support)
- Python 3.11+ for the data pipeline

**Software (native Windows path, for unblocked hosts only)**
- Windows 10/11 x64
- CUDA Toolkit 12.4+
- Visual Studio 2019 Build Tools (Desktop development with C++)
- Python 3.11+

**Python packages** (same for both paths) — see `requirements.txt`:
```
python -m pip install -r requirements.txt
```

`pip.exe` itself may be blocked by WDAC; `python -m pip` invokes pip through the signed `python.exe` and works around it.

## Setup

### Docker (recommended)

Works on any host, including machines with Windows Device Guard / WDAC in enforced mode where unsigned binaries cannot run.

```powershell
# First run: build the image (~3 GB, one time)
.\run.ps1 build

# Compile the native targets into ./bin
.\run.ps1 build

# Run something
.\run.ps1 lsm_engine                                 # validation case
.\run.ps1 lsm_engine --paths 1000000                 # larger MC
.\run.ps1 lsm_engine --input data/contracts.csv --output data/results.csv
.\run.ps1 shell                                      # interactive shell
```

`run.ps1` wraps `docker run --rm --gpus all -v ${PWD}:/app nse-cuda <target> <args>`.

### Native Windows (alternative)

For hosts without WDAC.

```cmd
REM x64 Native Tools Command Prompt for VS 2019
build.bat
lsm_engine.exe --paths 1000000
```

## Running the full pipeline

```powershell
python fetch_nse_stocks.py
.\run.ps1 lsm_engine --input data/contracts.csv --output data/results.csv
python plot_american.py
```

Open `plots/dashboard.html` in a browser.

## Single-contract pricing

Useful for sanity checks and validation:

```powershell
# LS 2001 Table 1 validation (no args)
.\run.ps1 lsm_engine

# Custom contract
.\run.ps1 lsm_engine --S0 1500 --K 1450 --r 0.065 --sigma 0.28 --T 0.25 --type put --paths 1000000

# CPU baseline for comparison
.\run.ps1 lsm_cpu --paths 1000000
```

## Project layout

```
src/
  options_engine.cu       BSM kernel + Greeks (Phase 0)
  cpu_baseline.cpp        BSM sequential baseline
  lsm_engine.cu           LSM pricer, single + batch modes (Phase 1 + 2)
  lsm_cpu.cpp             LSM sequential baseline

fetch_nse_stocks.py       Phase 2 data fetch (yfinance + synthetic strikes)
plot_american.py          Phase 2 dashboard builder

Dockerfile                CUDA devel base + build tools
build.sh                  Linux build (runs inside container)
run.ps1                   PowerShell wrapper
build.bat                 Native Windows build

fetch_nse.py              Phase 0 legacy (Kite API + synthetic IV)
kite_login.py             Phase 0 legacy (Zerodha OAuth)
debug_nse.py              Phase 0 legacy (nsepython probe)
plot_iv_surface.py        Phase 0 IV surface visualizer

overview.txt              Deep code walkthrough
requirements.txt          Python dependencies
```

## How LSM works

Longstaff & Schwartz (2001) estimate the continuation value of an American option by regressing discounted future cash flows on a polynomial basis of the current underlying price. Stepping backward from expiry:

1. Simulate N paths under geometric Brownian motion. One thread per path. RNG is Philox via cuRAND.
2. Initialise V with the terminal payoff.
3. For t = M-1 down to 1:
   - Discount V by `exp(-r * dt)`.
   - For in-the-money paths only, accumulate seven moments (sums of `s`, `s^2`, `s^3`, `s^4`, `y`, `s*y`, `s^2 * y` with `s = S/K`, `y = V`) into device memory via `atomicAdd(double*, double)`.
   - Copy the moments back to host. Solve the 3x3 symmetric system for `beta` via explicit adjugate. Copy `beta` back to device.
   - Save `beta[t]` to the policy array. Update V[i] to `max(intrinsic, beta . basis(S[i][t]))` for ITM paths.
4. Discount once more, average V for the in-sample estimate (V_IS).
5. **Phase 2 (OOS valuation, Rasmussen 2005)**: simulate a fresh path set with an independent seed. For each path, walk forward; at each step, compute the continuation estimate using the saved `beta[t]`. Exercise the first time `intrinsic > continuation`; otherwise take the terminal payoff. Discount each path's cash flow to t=0 and average. This V_OOS is the primary result.

The 3x3 solve runs on the host because the matrix is too small for cuSOLVER to be worthwhile. This forces a sync per exercise date, which is the current performance ceiling.

### Why out-of-sample

In-sample LSM reuses the same paths for both policy fitting and price evaluation. The fit is optimistic (it finds coefficients that happen to score well on those specific paths), which partially cancels the downward bias from using a suboptimal polynomial policy. The two biases offset unpredictably, and for deep-ITM short-dated contracts the residual error can be large (we observed one TCS OTM put landing at V_IS = V_European − 4.78 before the correction). OOS eliminates the in-sample optimism entirely: V_OOS is a rigorous lower bound on the true American value, and extreme outliers collapse (the TCS put went from −4.78 to −0.14 after the correction). The tradeoff is that for non-dividend calls, where the optimistic and pessimistic biases roughly cancel in IS, OOS slightly widens the mean error (because only the downward component remains). This is the textbook behavior and documented in Rasmussen (2005).

## Data sources and honesty

Option chain scraping from nseindia.com is blocked at the application layer (HTTP 200 with empty `records` dict, verified with both `nsepython` and direct `requests` calls using standard browser headers). Rather than ship 300 MB of headless browser automation, the Phase 2 pipeline uses:

- **Real** spot prices, from yfinance `<TICKER>.NS`.
- **Real** 30-day annualized log-return volatility, from yfinance daily closes.
- **Real** expiry dates, last Thursday of each of the next 3 months (NSE convention).
- **Synthetic** strike ladders, generated on NSE's standard strike-step rules (piecewise by spot band).

The pricing is real. The strike grid it is priced on is a plausible simulation of what NSE actually lists.

## Phase 0 legacy

`fetch_nse.py` and `kite_login.py` use the paid Zerodha Kite Connect API (Rs. 2000/month). They remain in the repo for reference and still work if you have an active subscription. Phase 2 uses only free data sources.

## Known limitations

- FP32 single-precision throughout. Adequate for MC because sampling variance dominates FP64 differences.
- Three-term polynomial basis (`1, s, s^2`) with `s = S/K`. Fine for validation; higher-order bases or Laguerre polynomials would be marginal here.
- No dividend model. American calls on non-dividend stocks equal European calls. The OOS estimator shows a small consistent downward bias on call premiums (mean -0.7% of price) driven by spurious early exercise from regression-induced continuation-value noise. Clean fixes: richer basis, basis chosen conditionally per step, or exploit Merton's inequality for non-dividend calls.
- Current LSM speedup is 7x at 1M paths, and sustained SM utilization is 85-93% during the batch. Further speedup requires either more parallelism across contracts (streams) or algorithmic changes; kernel optimization alone has little headroom.
- Single-contract mode allocates device buffers per invocation; only batch mode reuses.

## References

- Longstaff, Francis A. and Eduardo S. Schwartz. "Valuing American Options by Simulation: A Simple Least-Squares Approach." *Review of Financial Studies* 14.1 (2001): 113-147.
- Rasmussen, Nicki S. "Control variates for Monte Carlo valuation of American options." *Journal of Computational Finance* 9.1 (2005): 83-118. (Out-of-sample valuation scheme adopted here.)
