````markdown
# NSE Options Pricing Engine


> A high-throughput, parallelized options pricing engine built with C++ and CUDA. By offloading Black-Scholes-Merton calculations to an NVIDIA RTX 4050 GPU, the engine evaluates an entire NSE NIFTY 50 options chain concurrently, demonstrating a **269x speedup** over a sequential CPU implementation at scale.

---

## Motivation

The National Stock Exchange (NSE) of India is the largest derivatives exchange globally by volume. Pricing thousands of active option contracts sequentially creates a computational bottleneck, leading to stale data and missed opportunities in a fast-moving market. This project addresses that bottleneck directly by leveraging massive GPU parallelization.

## Architecture

```mermaid
flowchart TD
    A[Live NSE Data<br><i>Zerodha Kite API</i>] --> B(Python Data Fetcher)
    B -->|Real NIFTY strikes & expiries<br>Spot via yfinance<br>Asymmetric IV surface model| C{CUDA Pricing Kernel<br><i>RTX 4050</i>}
    C -->|One thread per contract<br>BSM Pricing & Full Greeks<br>GPU Newton-Raphson IV Solver| D(IV Surface Visualization<br><i>Plotly</i>)
    D --> E[Interactive 3D Surface<br><i>Strike × Expiry × IV</i>]
````

## Benchmark Results

| Implementation | Contracts | Execution Time | Throughput |
| :--- | :--- | :--- | :--- |
| **CPU (Single)** | 1,000,000 | `130.6 ms` | 7.66M contracts/sec |
| **GPU (RTX 4050)** | 1,000,000 | `0.485 ms` | **2,062M contracts/sec** |

**Overall Speedup: 269x**

## Hardware Specs

  - **CPU:** AMD Ryzen 5 6600H (6 cores, 12 threads)
  - **GPU:** NVIDIA GeForce RTX 4050 Laptop (2560 CUDA cores, 6GB VRAM)
  - **RAM:** 16GB DDR5 @ 5600MHz
  - **Environment:** CUDA 12.4 | Driver 551.76

## Project Structure

```text
nse_options_engine/
├── src/
│   ├── options_engine.cu      # CUDA kernel, BSM pricing + Greeks
│   └── cpu_baseline.cpp       # Sequential CPU implementation for benchmarking
├── fetch_nse.py               # Live data fetcher (Kite API + yfinance)
├── plot_iv_surface.py         # 3D IV surface visualization
├── kite_login.py              # Zerodha Kite authentication flow
├── build.bat                  # Compile script for CUDA + CPU targets
└── README.md
```

## Setup & Installation

### Prerequisites

  - Windows 10/11
  - NVIDIA GPU with CUDA support
  - [CUDA Toolkit 12.4](https://developer.nvidia.com/cuda-downloads)
  - Visual Studio 2019 Build Tools (Desktop development with C++)
  - Python 3.11+

### Installation Steps

1.  **Clone the repository**

    ```bash
    git clone [https://github.com/yourusername/nse_options_engine](https://github.com/yourusername/nse_options_engine)
    cd nse_options_engine
    ```

2.  **Set up Python environment**

    ```bash
    python -m venv venv
    venv\Scripts\Activate.ps1
    pip install kiteconnect python-dotenv yfinance pandas plotly scipy
    ```

3.  **Configure API credentials**
    Create a `.env` file in the project root:

    ```env
    KITE_API_KEY=your_api_key
    KITE_API_SECRET=your_api_secret
    ```

4.  **Authenticate with Zerodha** *(required once per day)*

    ```bash
    python kite_login.py
    ```

5.  **Build the CUDA engine**
    Open the `x64 Native Tools Command Prompt for VS 2019`, then run:

    ```cmd
    build.bat
    ```

## Usage

Use the following commands to interact with the engine:

| Task | Command |
| :--- | :--- |
| **Fetch live data** | `python fetch_nse.py` |
| **Run GPU pricing engine** | `options_engine.exe` |
| **Run CPU baseline** | `cpu_baseline.exe` |
| **Plot IV surface** | `python plot_iv_surface.py` |

## Key Concepts

  - **Black-Scholes-Merton (BSM):** A mathematical model for pricing European-style options. Assumes constant volatility, log-normal returns, and continuous trading. Despite its known limitations, BSM remains the industry standard for quoting options in terms of implied volatility.
  - **Implied Volatility (IV):** The volatility value that, when plugged into BSM, reproduces the observed market price. Solving for IV requires numerical methods since there is no closed-form inverse of BSM. This engine uses Newton-Raphson iteration on the GPU.
  - **Volatility Smile/Skew:** If BSM were perfectly accurate, IV would be constant across all strikes. In reality, it is not. Out-of-the-Money (OTM) puts carry higher IV than At-the-Money (ATM) options (the skew), and both OTM calls and puts carry more IV than ATM (the smile). The IV surface visualizes this phenomenon across strikes and expiries simultaneously.
  - **Embarrassingly Parallel:** BSM pricing of $N$ contracts requires no inter-thread communication. Each contract is fully independent. This makes it a textbook GPU workload where all $N$ contracts can be priced in a single parallel kernel launch.

## Data Source

Market data is sourced from the **Zerodha Kite API** (real NIFTY strikes and expiries) and **yfinance** (NIFTY 50 spot price). Option prices are generated using BSM with a realistic asymmetric volatility model calibrated to typical NSE market conditions. The engine is designed to accept live bid/ask prices directly when a Kite Connect subscription is active.

## Limitations

  - The Zerodha Personal plan does not include live market quotes. A Connect plan (or equivalent broker API) is required for real-time option prices.
  - BSM assumes constant volatility and does not account for jumps, stochastic volatility, or discrete dividends.
  - GPU timing measures kernel execution only. Full pipeline time, including PCIe data transfer overhead, will be higher.

```
