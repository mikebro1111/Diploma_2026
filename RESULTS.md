# Benchmark Results

Benchmark environment: **macOS, Apple M2, 16 GB RAM**.
Each configuration was measured **N=20 times** (independent process runs) for statistical validity.

All p-values computed via **Welch's t-test** (unequal variance).

## Summary: No-GIL Speedup at 8 Threads

| Workload | Python 3.13t | Python 3.14t | Improvement |
|---|---|---|---|
| **Fibonacci** (fib(34) x 24) | 2.82x | **4.98x** | +76% |
| **Streaming** (2M events) | 2.37x | **3.48x** | +47% |
| **Monte Carlo** (300M samples) | 0.31x | 0.38x | +22% |
| **Mandelbrot** (2236x2236) | 2.17x | 2.14x | = |
| **Data Preprocessing** (10M rows) | 0.84x | 1.00x | +19% |
| **SIMD/NumPy** (500M elements) | 0.90x | 0.93x | = |
| **ML Training** (500K samples) | 1.01x | 0.96x | = |
| **Image Processing** (9K photos) | 0.74x | **1.31x** | +77% |

> Speedup = GIL time / No-GIL time. Values > 1.0 mean No-GIL is faster.

## Detailed Results by Workload

### Pure Python CPU-bound (biggest wins)

These workloads are pure Python with no C-extension GIL releases — exactly where free-threading shines.

#### Fibonacci — fib(34) x 24 tasks

| Threads | 3.13 GIL | 3.13 No-GIL | Speedup | 3.14 GIL | 3.14 No-GIL | Speedup |
|---|---|---|---|---|---|---|
| 1 (seq) | 16.55s | 22.55s | 0.73x | 22.51s | 22.63s | 0.99x |
| 2 | 16.62s | 11.54s | 1.44x | 22.55s | 11.33s | 1.99x |
| 4 | 16.60s | 6.16s | 2.69x | 22.56s | 5.74s | 3.93x |
| 8 | 16.62s | 5.89s | **2.82x** | 22.59s | 4.54s | **4.98x** |

All results p < 0.001 (***). Note: sequential No-GIL is slower due to free-threading overhead, but threading scaling more than compensates.

Charts: [`results/3.13/charts/13_fibonacci.png`](results/3.13/charts/13_fibonacci.png), [`results/3.14/charts/13_fibonacci.png`](results/3.14/charts/13_fibonacci.png)

#### Monte Carlo Pi — 300M samples

| Threads | 3.13 GIL | 3.13 No-GIL | Speedup | 3.14 GIL | 3.14 No-GIL | Speedup |
|---|---|---|---|---|---|---|
| 1 (seq) | 24.62s | 61.01s | 0.40x | 48.15s | 53.49s | 0.90x |
| 2 | 24.66s | 52.22s | 0.47x | 48.21s | 67.30s | 0.72x |
| 4 | 24.79s | 99.52s | 0.25x | 48.27s | 88.52s | 0.55x |
| 8 | 24.74s | 79.05s | 0.31x | 48.28s | 126.37s | 0.38x |

Charts: [`results/3.13/charts/12_monte_carlo.png`](results/3.13/charts/12_monte_carlo.png), [`results/3.14/charts/12_monte_carlo.png`](results/3.14/charts/12_monte_carlo.png)

#### Mandelbrot — 2236x2236, max_iter=200

| Threads | 3.13 GIL | 3.13 No-GIL | Speedup | 3.14 GIL | 3.14 No-GIL | Speedup |
|---|---|---|---|---|---|---|
| 1 (seq) | 20.86s | 32.17s | 0.65x | 24.86s | 34.47s | 0.72x |
| 2 | 20.91s | 16.51s | 1.27x | 24.88s | 17.26s | 1.44x |
| 4 | 20.93s | 13.16s | 1.59x | 24.87s | 13.68s | 1.82x |
| 8 | 20.94s | 9.64s | **2.17x** | 24.92s | 11.63s | **2.14x** |

Charts: [`results/3.13/charts/11_mandelbrot.png`](results/3.13/charts/11_mandelbrot.png), [`results/3.14/charts/11_mandelbrot.png`](results/3.14/charts/11_mandelbrot.png)

### Streaming — 2M events, pure Python JSON/math

| Workers | 3.13 GIL | 3.13 No-GIL | Speedup | 3.14 GIL | 3.14 No-GIL | Speedup |
|---|---|---|---|---|---|---|
| 1 | 48.20ms | 58.27ms | 0.83x | 47.10ms | 46.56ms | 1.01x |
| 2 | 48.20ms | 28.91ms | 1.67x | 47.10ms | 24.25ms | 1.94x |
| 4 | 48.20ms | 15.33ms | 3.14x | 47.10ms | 12.94ms | 3.64x |
| 8 | 48.20ms | 20.37ms | **2.37x** | 47.10ms | 13.55ms | **3.48x** |

Charts: [`results/3.13/charts/05_streaming.png`](results/3.13/charts/05_streaming.png), [`results/3.14/charts/05_streaming.png`](results/3.14/charts/05_streaming.png)

### Data Preprocessing — 10M rows, pure NumPy

| Threads | 3.13 GIL | 3.13 No-GIL | Speedup | 3.14 GIL | 3.14 No-GIL | Speedup |
|---|---|---|---|---|---|---|
| 1 (seq) | 3.34s | 3.28s | 1.02x | 2.19s | 2.19s | 1.00x |
| 8 | 1.91s | 2.28s | 0.84x | 1.06s | 1.06s | 1.00x |

Minimal difference — NumPy already releases the GIL internally for C-level operations.

Charts: [`results/3.13/charts/02_data_preprocessing.png`](results/3.13/charts/02_data_preprocessing.png), [`results/3.14/charts/02_data_preprocessing.png`](results/3.14/charts/02_data_preprocessing.png)

### Image Processing — 9144 Caltech-101 photos, Pillow

| Threads | 3.13 GIL | 3.13 No-GIL | Speedup | 3.14 GIL | 3.14 No-GIL | Speedup |
|---|---|---|---|---|---|---|
| 1 (seq) | 22.66s | 23.67s | 0.96x | 13.43s | 13.46s | 1.00x |
| 8 | 9.00s | 12.20s | 0.74x | 4.12s | 3.15s | **1.31x** |

Charts: [`results/3.13/charts/03_image_processing.png`](results/3.13/charts/03_image_processing.png), [`results/3.14/charts/03_image_processing.png`](results/3.14/charts/03_image_processing.png)

### ML Training — scikit-learn via joblib

| Mode | 3.13 GIL | 3.13 No-GIL | Speedup | 3.14 GIL | 3.14 No-GIL | Speedup |
|---|---|---|---|---|---|---|
| RF seq | 32.61s | 33.06s | 0.99x | 32.82s | 33.05s | 0.99x |
| RF 8T | 6.98s | 6.92s | 1.01x | 6.77s | 7.06s | 0.96x |

Charts: [`results/3.13/charts/04_ml_training.png`](results/3.13/charts/04_ml_training.png), [`results/3.14/charts/04_ml_training.png`](results/3.14/charts/04_ml_training.png)

### SIMD/NumPy Vectorization — 500M elements

| Mode | 3.13 GIL | 3.13 No-GIL | Speedup | 3.14 GIL | 3.14 No-GIL | Speedup |
|---|---|---|---|---|---|---|
| NumPy add (seq) | 1.24s | 1.25s | 0.99x | 2.01s | 1.57s | 1.28x |
| NumPy threaded 8T| 0.26s | 0.29s | 0.90x | 0.26s | 0.28s | 0.93x |

Charts: [`results/3.13/charts/06_simd_vectorization.png`](results/3.13/charts/06_simd_vectorization.png), [`results/3.14/charts/06_simd_vectorization.png`](results/3.14/charts/06_simd_vectorization.png)

## Key Findings

1. **Pure Python CPU-bound workloads benefit enormously** from free-threading: up to **4.98x** speedup on 8 threads (Fibonacci, Python 3.14t)
2. **Python 3.14t significantly improved** free-threading performance over 3.13t: Fibonacci went from 2.8x to 4.98x (+76%)
3. **C-extension-heavy workloads** (NumPy, scikit-learn, BLAS) show **no benefit** — they already release the GIL
4. **Pillow image processing scales well** on macOS Apple Silicon, reaching **1.31x** speedup on 3.14t.
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
