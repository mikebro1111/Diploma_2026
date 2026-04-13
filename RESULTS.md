# Benchmark Results

Benchmark environment: **Windows 11, Intel Core i5-13400F, 32 GB RAM**.
Each configuration was measured **N=20 times** (independent process runs) for statistical validity.

All p-values computed via **Welch's t-test** (unequal variance).

## Summary: No-GIL Speedup at 8 Threads

| Workload | Python 3.13t | Python 3.14t | Improvement |
|---|---|---|---|
| **Fibonacci** (fib(34) x 24) | 3.30x | **5.10x** | +55% |
| **Streaming** (2M events) | 4.13x | **4.66x** | +13% |
| **Monte Carlo** (300M samples) | 3.90x | **4.40x** | +13% |
| **Mandelbrot** (2236x2236) | 2.77x | **3.54x** | +28% |
| **Data Preprocessing** (10M rows) | 1.06x | 1.06x | = |
| **SIMD/NumPy** (500M elements) | ~1.0x | ~1.0x | = |
| **ML Training** (500K samples) | ~1.0x | ~1.0x | = |
| **Image Processing** (9K photos) | 0.63x | 0.71x | +13% |

> Speedup = GIL time / No-GIL time. Values > 1.0 mean No-GIL is faster.

## Detailed Results by Workload

### Pure Python CPU-bound (biggest wins)

These workloads are pure Python with no C-extension GIL releases — exactly where free-threading shines.

#### Fibonacci — fib(34) x 24 tasks

| Threads | 3.13 GIL | 3.13 No-GIL | Speedup | 3.14 GIL | 3.14 No-GIL | Speedup |
|---|---|---|---|---|---|---|
| 1 (seq) | 12.34s | 20.81s | 0.59x | 13.56s | 14.48s | 0.94x |
| 2 | 12.34s | 10.83s | 1.14x | 13.52s | 7.55s | **1.79x** |
| 4 | 12.32s | 6.00s | 2.05x | 13.57s | 4.26s | **3.18x** |
| 8 | 12.32s | 3.74s | **3.30x** | 13.53s | 2.66s | **5.10x** |

All results p < 0.001 (***). Note: sequential No-GIL is slower due to free-threading overhead, but threading scaling more than compensates.

Charts: [`results/3.13/charts/13_fibonacci.png`](results/3.13/charts/13_fibonacci.png), [`results/3.14/charts/13_fibonacci.png`](results/3.14/charts/13_fibonacci.png)

#### Monte Carlo Pi — 300M samples

| Threads | 3.13 GIL | 3.13 No-GIL | Speedup | 3.14 GIL | 3.14 No-GIL | Speedup |
|---|---|---|---|---|---|---|
| 1 (seq) | 28.93s | 38.63s | 0.75x | 26.88s | 31.60s | 0.85x |
| 2 | 28.85s | 20.40s | 1.41x | 26.94s | 16.61s | **1.62x** |
| 4 | 28.85s | 11.46s | 2.52x | 26.89s | 9.51s | **2.83x** |
| 8 | 28.91s | 7.42s | **3.90x** | 26.85s | 6.11s | **4.40x** |

Charts: [`results/3.13/charts/12_monte_carlo.png`](results/3.13/charts/12_monte_carlo.png), [`results/3.14/charts/12_monte_carlo.png`](results/3.14/charts/12_monte_carlo.png)

#### Mandelbrot — 2236x2236, max_iter=200

| Threads | 3.13 GIL | 3.13 No-GIL | Speedup | 3.14 GIL | 3.14 No-GIL | Speedup |
|---|---|---|---|---|---|---|
| 1 (seq) | 26.55s | 34.52s | 0.77x | 33.17s | 33.56s | 0.99x |
| 2 | 26.63s | 17.97s | 1.48x | 33.25s | 17.61s | **1.89x** |
| 4 | 26.68s | 14.84s | 1.80x | 33.23s | 14.38s | **2.31x** |
| 8 | 27.08s | 9.77s | **2.77x** | 33.17s | 9.36s | **3.54x** |

Charts: [`results/3.13/charts/11_mandelbrot.png`](results/3.13/charts/11_mandelbrot.png), [`results/3.14/charts/11_mandelbrot.png`](results/3.14/charts/11_mandelbrot.png)

### Streaming — 2M events, pure Python JSON/math

| Workers | 3.13 GIL | 3.13 No-GIL | Speedup | 3.14 GIL | 3.14 No-GIL | Speedup |
|---|---|---|---|---|---|---|
| 1 | 51.08s | 61.84s | 0.83x | 51.80s | 59.89s | 0.86x |
| 2 | 50.62s | 33.54s | 1.51x | 51.61s | 32.63s | **1.58x** |
| 4 | 50.75s | 17.41s | 2.91x | 51.35s | 16.82s | **3.05x** |
| 8 | 50.98s | 12.33s | **4.13x** | 51.62s | 11.08s | **4.66x** |

Throughput at 8 workers: 39K → **160K** events/sec (3.13t), 39K → **178K** events/sec (3.14t).

Charts: [`results/3.13/charts/05_streaming.png`](results/3.13/charts/05_streaming.png), [`results/3.14/charts/05_streaming.png`](results/3.14/charts/05_streaming.png)

### Data Preprocessing — 10M rows, pure NumPy

| Threads | 3.13 GIL | 3.13 No-GIL | Speedup | 3.14 GIL | 3.14 No-GIL | Speedup |
|---|---|---|---|---|---|---|
| 1 (seq) | 3.71s | 3.71s | 1.00x | 3.57s | 3.60s | 0.99x |
| 2 | 2.52s | 2.41s | 1.05x | 2.41s | 2.36s | 1.02x |
| 4 | 1.57s | 1.48s | 1.06x | 1.58s | 1.47s | **1.08x** |
| 8 | 1.19s | 1.13s | 1.06x | 1.15s | 1.09s | 1.06x |

Minimal difference — NumPy already releases the GIL internally for C-level operations.

Charts: [`results/3.13/charts/02_data_preprocessing.png`](results/3.13/charts/02_data_preprocessing.png), [`results/3.14/charts/02_data_preprocessing.png`](results/3.14/charts/02_data_preprocessing.png)

### Image Processing — 9144 Caltech-101 photos, Pillow

| Threads | 3.13 GIL | 3.13 No-GIL | Speedup | 3.14 GIL | 3.14 No-GIL | Speedup |
|---|---|---|---|---|---|---|
| 1 (seq) | 19.82s | 21.56s | 0.92x | 19.91s | 20.84s | 0.96x |
| 2 | 10.76s | 12.02s | 0.90x | 10.87s | 11.87s | 0.92x |
| 4 | 6.30s | 8.00s | 0.79x | 6.42s | 7.32s | 0.88x |
| 8 | 5.13s | 8.14s | **0.63x** | 5.13s | 7.22s | **0.71x** |

No-GIL is **slower** for Pillow — the free-threading overhead outweighs parallelism gains because Pillow's C code already releases the GIL. The per-thread overhead of free-threading (reference counting, memory allocation) adds up with 9K small images.

Charts: [`results/3.13/charts/03_image_processing.png`](results/3.13/charts/03_image_processing.png), [`results/3.14/charts/03_image_processing.png`](results/3.14/charts/03_image_processing.png)

### ML Training — scikit-learn via joblib

| Mode | 3.13 GIL | 3.13 No-GIL | Speedup | 3.14 GIL | 3.14 No-GIL | Speedup |
|---|---|---|---|---|---|---|
| LinReg seq | 2.11s | 2.09s | 1.01x | 2.11s | 2.10s | 1.01x |
| LinReg 8T | 2.12s | 2.07s | 1.02x | 2.09s | 2.10s | 1.00x |
| RF seq | 39.52s | 39.30s | 1.01x | 37.92s | 39.31s | 0.96x |
| RF 8T | 6.98s | 6.88s | 1.01x | 6.94s | 6.88s | 1.01x |

No meaningful difference — scikit-learn uses BLAS/LAPACK internally which already bypasses the GIL.

Charts: [`results/3.13/charts/04_ml_training.png`](results/3.13/charts/04_ml_training.png), [`results/3.14/charts/04_ml_training.png`](results/3.14/charts/04_ml_training.png)

### SIMD/NumPy Vectorization — 500M elements

| Mode | 3.13 GIL | 3.13 No-GIL | Speedup | 3.14 GIL | 3.14 No-GIL | Speedup |
|---|---|---|---|---|---|---|
| NumPy add (seq) | 0.86s | 0.84s | 1.02x | 0.86s | 0.85s | 1.01x |
| NumPy complex (seq) | 6.58s | 6.62s | 0.99x | 6.60s | 6.65s | 0.99x |
| NumPy threaded 8T | 0.33s | 0.32s | 1.01x | 0.32s | 0.33s | 0.98x |
| Matrix matmul | 0.009s | 0.009s | 1.00x | 0.009s | 0.009s | 1.05x |

NumPy operations are C-level and already release the GIL — no benefit from free-threading.

Charts: [`results/3.13/charts/06_simd_vectorization.png`](results/3.13/charts/06_simd_vectorization.png), [`results/3.14/charts/06_simd_vectorization.png`](results/3.14/charts/06_simd_vectorization.png)

## Key Findings

1. **Pure Python CPU-bound workloads benefit enormously** from free-threading: up to **5.1x** speedup on 8 threads (Fibonacci, Python 3.14t)
2. **Python 3.14t significantly improved** free-threading performance over 3.13t: Fibonacci went from 3.3x to 5.1x (+55%)
3. **C-extension-heavy workloads** (NumPy, scikit-learn, BLAS) show **no benefit** — they already release the GIL
4. **Pillow image processing is slower** under free-threading due to per-object overhead with many small allocations
5. **Sequential overhead** of free-threading is 5-25% (visible in single-threaded benchmarks), but threading gains compensate starting from 2 threads
6. **Near-linear scaling** observed for pure Python workloads up to 8 threads

## Raw Data

Full JSON results with all individual measurements:

| File | Description |
|------|-------------|
| [`results/3.13/multi_run_master.json`](results/3.13/multi_run_master.json) | Python 3.13: all timing + resource data |
| [`results/3.14/multi_run_master.json`](results/3.14/multi_run_master.json) | Python 3.14: all timing + resource data |
| [`results/3.13/statistical_analysis_multirun.json`](results/3.13/statistical_analysis_multirun.json) | Python 3.13: t-tests, p-values, speedups |
| [`results/3.14/statistical_analysis_multirun.json`](results/3.14/statistical_analysis_multirun.json) | Python 3.14: t-tests, p-values, speedups |

Per-workload JSON files are in `results/{version}/` (e.g., `fibonacci_results.json`, `streaming_results.json`).

## All Charts

Charts are in `results/{version}/charts/`. See [README.md](README.md#generated-charts) for the full list.

Overview charts:
- [`results/3.14/charts/01_overall_comparison.png`](results/3.14/charts/01_overall_comparison.png) — normalized comparison
- [`results/3.14/charts/07_speedup_heatmap.png`](results/3.14/charts/07_speedup_heatmap.png) — heatmap across all modes
- [`results/3.14/charts/08_confidence_intervals.png`](results/3.14/charts/08_confidence_intervals.png) — 95% CI for speedups
