```markdown
# NSE Options Pricing Engine

A high-throughput, parallelized options pricing engine built with C++ and CUDA. By offloading Black-Scholes-Merton calculations to an NVIDIA RTX 4050 GPU, the engine evaluates an entire NSE NIFTY 50 options chain concurrently, demonstrating a **269x speedup** over a sequential CPU implementation at scale.

## Motivation

The National Stock Exchange (NSE) of India is the largest derivatives exchange globally by volume. Pricing thousands of active option contracts sequentially creates a computational bottleneck, stale data and missed opportunities in a fast-moving market. This project addresses that bottleneck directly.

## Architecture

```
Live NSE Data (Zerodha Kite API)
|
v
Python Data Fetcher
  - Real NIFTY strikes and expiries via Kite
  - Spot price via yfinance
  - Realistic IV surface model (asymmetric skew + term structure)
|
v
CUDA Pricing Kernel (RTX 4050)
  - One thread per contract
  - Black-Scholes-Merton pricing
  - Full Greeks: Delta, Gamma, Theta, Vega, Rho
  - Newton-Raphson IV solver (GPU-side)
|
v
IV Surface Visualization (Plotly)
  - Interactive 3D surface
  - Strike x Expiry x Implied Volatility
  - ATM line, volatility smile, term structure
```

## Benchmark Results

| Implementation | Contracts | Time     | Throughput           |
|----------------|-----------|----------|----------------------|
| CPU (single)   | 1,000,000 | 130.6 ms | 7.66M contracts/sec  |
| GPU (RTX 4050) | 1,000,000 | 0.485 ms | 2,062M contracts/sec |

**Speedup: 269x**

## Hardware

- **CPU:** AMD Ryzen 5 6600H (6 cores, 12 threads)
- **GPU:** NVIDIA GeForce RTX 4050 Laptop (2560 CUDA cores, 6GB VRAM)
- **RAM:** 16GB DDR5 @ 5600MHz
- **CUDA:** 12.4, Driver: 551.76

## Project Structure

```
nse_options_engine/
├── src/
│   ├── options_engine.cu     # CUDA kernel, BSM pricing + Greeks
│   └── cpu_baseline.cpp      # Sequential CPU implementation for benchmarking
├── fetch_nse.py               # Live data fetcher (Kite API + yfinance)
├── plot_iv_surface.py         # 3D IV surface visualization
├── kite_login.py              # Zerodha Kite authentication flow
├── build.bat                  # Compile script for CUDA + CPU targets
└── README.md
```

## Setup

### Prerequisites

- Windows 10/11
- NVIDIA GPU with CUDA support
- CUDA Toolkit 12.4
- Visual Studio 2019 Build Tools (Desktop development with C++)
- Python 3.11+

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/nse_options_engine
   cd nse_options_engine
   ```

2. **Set up Python environment**
   ```bash
   python -m venv venv
   venv\Scripts\Activate.ps1
   pip install kiteconnect python-dotenv yfinance pandas plotly scipy
   ```

3. **Configure API credentials**
   Create a `.env` file in the project root:
   ```
   KITE_API_KEY=your_api_key
   KITE_API_SECRET=your_api_secret
   ```

4. **Authenticate with Zerodha** (required once per day)
   ```bash
   python kite_login.py
   ```

5. **Build the CUDA engine**
   Open `x64 Native Tools Command Prompt for VS 2019`, then:
   ```bash
   build.bat
   ```

### Running

| Task                          | Command                      |
|-------------------------------|------------------------------|
| Fetch live data               | `python fetch_nse.py`        |
| Run GPU pricing engine        | `options_engine.exe`         |
| Run CPU baseline              | `cpu_baseline.exe`           |
| Plot IV surface               | `python plot_iv_surface.py`  |

## Key Concepts

**Black-Scholes-Merton (BSM):** A mathematical model for pricing European-style options. Assumes constant volatility, log-normal returns, and continuous trading. Despite its known limitations, BSM remains the industry standard for quoting options in terms of implied volatility.

**Implied Volatility (IV):** The volatility value that, when plugged into BSM, reproduces the observed market price. Solving for IV requires numerical methods since there is no closed-form inverse of BSM. This engine uses Newton-Raphson iteration on the GPU.

**Volatility Smile/Skew:** If BSM were correct, IV would be constant across all strikes. In reality it is not. OTM puts carry higher IV than ATM options (the skew), and both OTM calls and puts carry more IV than ATM (the smile). The IV surface visualizes this phenomenon across strikes and expiries simultaneously.

**Embarrassingly Parallel:** BSM pricing of N contracts requires no inter-thread communication. Each contract is fully independent. This makes it a textbook GPU workload where all N contracts can be priced in a single parallel kernel launch.

## Data Source

Market data is sourced from the **Zerodha Kite API** (real NIFTY strikes and expiries) and **yfinance** (NIFTY 50 spot price). Option prices are generated using BSM with a realistic asymmetric volatility model calibrated to typical NSE market conditions. The engine is designed to accept live bid/ask prices directly when a Kite Connect subscription is available.

## Limitations

- Zerodha Personal plan does not include live market quotes. A Connect plan (or equivalent broker API) is required for real-time option prices.
- BSM assumes constant volatility and does not account for jumps, stochastic volatility, or discrete dividends.
- GPU timing measures kernel execution only. Full pipeline time including PCIe data transfer is higher.
