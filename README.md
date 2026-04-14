# Free-threaded Python Benchmark

Researching performance benefits of the free-threaded (no-GIL) Python 3.13/3.14 interpreter for CPU-bound and Data Science workloads.

## Project Structure

```
Diploma_2026/
├── workloads/                              # Workload implementations (8 benchmarks)
│   ├── workload_data_preprocessing.py      #   Pure NumPy tabular processing (10M rows)
│   ├── workload_image_processing.py        #   Pillow image pipeline (Caltech-101, 9K+ photos)
│   ├── workload_ml.py                      #   ML Training via scikit-learn/joblib (500K samples)
│   ├── workload_streaming.py               #   Pure Python event streaming (2M events)
│   ├── workload_simd.py                    #   SIMD/NumPy vectorization (500M elements)
│   ├── workload_mandelbrot.py              #   Mandelbrot set, pure Python (2236x2236)
│   ├── workload_monte_carlo.py             #   Monte Carlo Pi estimation (300M samples)
│   └── workload_fibonacci.py               #   Recursive Fibonacci fib(34) x 24 tasks
├── scripts/                                # Orchestration & analysis
│   ├── run_multiple_benchmarks.py          #   Multi-iteration benchmark runner (GIL vs No-GIL)
│   ├── visualize_results.py                #   Chart generation & statistical analysis
│   ├── download_real_datasets.py           #   Dataset downloader (Caltech-101, NYC Taxi)
│   ├── setup_313_windows.ps1               #   Windows: auto-install Python 3.13 + 3.13t
│   └── setup_314_windows.ps1               #   Windows: auto-install Python 3.14 + 3.14t
├── results/                                # Generated results (per Python version)
│   ├── 3.13/
│   │   ├── charts/                         #   14 charts (.png)
│   │   ├── multi_run_master.json           #   Aggregated timing + resource data
│   │   └── statistical_analysis_multirun.json
│   └── 3.14/
│       ├── charts/                         #   14 charts (.png)
│       ├── multi_run_master.json
│       └── statistical_analysis_multirun.json
├── data_input/                             # Downloaded datasets (gitignored)
│   └── nyc_taxi.parquet                    #   NYC Yellow Taxi (~3M rows)
├── images_input/                           # Image dataset (gitignored)
│   └── *.jpg                               #   Caltech-101 real photos (9144 images)
├── requirements_gil.txt                    # Dependencies for GIL builds
├── requirements_nogil.txt                  # Dependencies for No-GIL builds
├── RESULTS.md                              # Summary of benchmark results
└── README.md
```

## Prerequisites

- **macOS** (tested on Apple Silicon) or **Windows 10/11** (tested on Intel i5-13400F)
- Python **3.13** and/or **3.14** — both standard (GIL) and free-threaded (no-GIL) builds

## Installation

### 1. System Dependencies

<details>
<summary><b>macOS</b></summary>

```bash
xcode-select --install
brew install openssl readline sqlite3 xz zlib tcl-tk pyenv
```

Install Python via pyenv:
```bash
pyenv install 3.13.0 && pyenv install 3.13.0t
pyenv install 3.14-dev  # dev branch often includes free-threading options
```

### 2. Environment Setup

<details>
<summary><b>macOS / Linux (automated)</b></summary>

Shell script automates `venv` creation and dependency installation:
```bash
chmod +x scripts/setup_envs_macos.sh
./scripts/setup_envs_macos.sh
```
</details>

<details>
<summary><b>macOS / Linux (manual)</b></summary>

```bash
cd Diploma_2026
# Python 3.13
~/.pyenv/versions/3.13.0/bin/python3.13   -m venv venv_gil
~/.pyenv/versions/3.13.0t/bin/python3.13t -m venv venv_nogil
# Python 3.14
~/.pyenv/versions/3.14-dev/bin/python3.14 -m venv venv_gil_314
~/.pyenv/versions/3.14-dev/bin/python3.14 -m venv venv_nogil_314
```
</details>
cd Diploma_2026

# Python 3.13
~/.pyenv/versions/3.13.0/bin/python3.13   -m venv venv_gil
~/.pyenv/versions/3.13.0t/bin/python3.13t -m venv venv_nogil

# Python 3.14
~/.pyenv/versions/3.14.0/bin/python3.14   -m venv venv_gil_314
~/.pyenv/versions/3.14.0t/bin/python3.14t -m venv venv_nogil_314
```

### 3. Dependencies

<details>
<summary><b>macOS</b></summary>

```bash
venv_gil/bin/pip install -r requirements_gil.txt
venv_nogil/bin/pip install -r requirements_nogil.txt
# Repeat for 3.14 venvs
```
</details>

<details>
<summary><b>Windows</b></summary>

```powershell
venv_gil\Scripts\pip install -r requirements_gil.txt
venv_nogil\Scripts\pip install -r requirements_nogil.txt
```
</details>

> **Note:** `requirements_nogil.txt` excludes `numba`/`llvmlite` (incompatible with free-threaded Python). Install `pyarrow` separately if needed for parquet support.

### 4. Verification

```bash
# macOS
venv_gil/bin/python -c "import sys; print('GIL:', sys._is_gil_enabled())"
venv_nogil/bin/python -c "import sys; print('GIL:', sys._is_gil_enabled())"
```
```powershell
# Windows
venv_gil\Scripts\python -c "import sys; print('GIL:', sys._is_gil_enabled())"
venv_nogil\Scripts\python -c "import sys; print('GIL:', sys._is_gil_enabled())"
```

## Dataset Downloading

```bash
# macOS
venv_gil/bin/python scripts/download_real_datasets.py

# Windows
venv_gil\Scripts\python scripts\download_real_datasets.py
```

| Dataset | Source | Size | Usage |
|---------|--------|------|-------|
| **Caltech-101** | `data.caltech.edu` | ~131 MB | Image Processing (9144 real photos) |
| **NYC Yellow Taxi** | `nyc.gov` TLC | ~48 MB | Data Preprocessing (~3M rows, Parquet) |

## Running Benchmarks

### Full Run (Recommended)

Runs all 8 workloads × 2 variants (GIL, No-GIL) × N iterations:

```bash
# macOS — 20 iterations, both 3.13 and 3.14
venv_gil/bin/python scripts/run_multiple_benchmarks.py 20

# macOS — specific version only
venv_gil/bin/python scripts/run_multiple_benchmarks.py 20 3.14
```
```powershell
# Windows — set UTF-8 to avoid encoding issues
set "PYTHONUTF8=1" && venv_gil_314\Scripts\python scripts\run_multiple_benchmarks.py 20
set "PYTHONUTF8=1" && venv_gil_314\Scripts\python scripts\run_multiple_benchmarks.py 20 3.14
```

### Visualization

```bash
# macOS
venv_gil/bin/python scripts/visualize_results.py results/3.13
venv_gil/bin/python scripts/visualize_results.py results/3.14
```
```powershell
# Windows
set "PYTHONUTF8=1" && venv_gil_314\Scripts\python scripts\visualize_results.py results\3.13
set "PYTHONUTF8=1" && venv_gil_314\Scripts\python scripts\visualize_results.py results\3.14
```

### Individual Workloads

```bash
python workloads/workload_data_preprocessing.py 10000000
python workloads/workload_image_processing.py images_input images_output
python workloads/workload_ml.py 500000
python workloads/workload_streaming.py 2000000
python workloads/workload_simd.py 500000000
python workloads/workload_mandelbrot.py 2236 1
python workloads/workload_monte_carlo.py 300000000 1
python workloads/workload_fibonacci.py 34 24 1
```

## Generated Charts

14 charts per Python version in `results/{version}/charts/`:

| Chart | Description |
|-------|-------------|
| `01_overall_comparison.png` | Normalized GIL vs No-GIL comparison across all workloads |
| `02_data_preprocessing.png` | Data Preprocessing thread scaling (1→8) |
| `03_image_processing.png` | Image Processing thread scaling |
| `04_ml_training.png` | ML Training: algorithms + thread scaling |
| `05_boxplots.png` | Distribution box plots with t-test results |
| `05_streaming.png` | Streaming: latency and throughput |
| `06_simd_vectorization.png` | SIMD/NumPy sequential + threaded |
| `07_speedup_heatmap.png` | Heatmap of speedups across all modes |
| `08_confidence_intervals.png` | 95% confidence intervals for speedups |
| `09_cpu_utilization.png` | CPU utilization (mean / peak) |
| `10_memory_usage.png` | Memory usage (mean / peak) |
| `11_mandelbrot.png` | Mandelbrot thread scaling |
| `12_monte_carlo.png` | Monte Carlo thread scaling |
| `13_fibonacci.png` | Fibonacci thread scaling |

## Collected Metrics

| Metric | Tool | Description |
|--------|------|-------------|
| Execution time | `time.perf_counter()` | Wall time per workload mode |
| CPU Utilization | `psutil` | Mean and peak % CPU usage |
| Memory Usage | `psutil` | Mean and peak RAM (MB) |
| Speedup | GIL time / No-GIL time | Acceleration factor |
| Scalability | 1/2/4/8 threads | Scaling with thread count |
| Statistical significance | Welch's t-test | p-values (N=20 measurements) |

## Results

See [RESULTS.md](RESULTS.md) for a detailed summary of benchmark results.

---
**Author:** Mykhailo Brodiuk — Diploma Project, 2026
