# Free-threaded Python Benchmark

Researching performance benefits of the free-threaded (no-GIL) Python 3.13 interpreter for Data Science workloads.

## Project Structure

```
Diploma_2026/
├── workloads/                          # Workload implementations (5 benchmarks)
│   ├── workload_data_preprocessing.py  #   Tabular processing (NYC Taxi, ~3M rows)
│   ├── workload_image_processing.py    #   Image processing (Caltech-101, 9K+ photos)
│   ├── workload_ml.py                  #   ML Training (200K samples, 50 features)
│   ├── workload_streaming.py           #   Real-time event streaming (50K events)
│   └── workload_simd.py               #   SIMD-vectorization (50M elements)
├── scripts/                            # Implementation scripts
│   ├── download_real_datasets.py       #   Dataset downloader
│   ├── run_multiple_benchmarks.py      #   Multi-iteration benchmarker
│   └── visualize_results.py            #   Visualization and statistical analysis
├── results/                            # Generated results
│   ├── charts/                         #   11 charts (.png)
│   ├── multi_run_master.json           #   Aggregated JSON master file
│   └── statistical_analysis_multirun.json
├── data_input/                         # Downloaded datasets (gitignored)
│   └── nyc_taxi.parquet                #   NYC Yellow Taxi dataset
├── images_input/                       # Image dataset (gitignored)
│   └── *.jpg                           #   Caltech-101 real photos
├── requirements_gil.txt                # Dependencies for GIL version
├── requirements_nogil.txt              # Dependencies for No-GIL version
├── diploma_detailed_plan.md            # Detailed diploma project plan
└── README.md
```

## Prerequisites

- **macOS** (tested on Apple Silicon)
- **pyenv** (for Python version management)
- **Homebrew** (for system libraries)

## Installation

### 1. System Dependencies (macOS)

```bash
# Xcode Command Line Tools
xcode-select --install

# Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Required libraries for Python compilation
brew install openssl readline sqlite3 xz zlib tcl-tk
```

### 2. Installing Python 3.13 via pyenv

```bash
# Install pyenv
brew install pyenv

# Add to ~/.zshrc:
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
source ~/.zshrc

# Standard version (with GIL)
pyenv install 3.13.0

# Free-threaded version (without GIL) — "t" suffix
pyenv install 3.13.0t
```

> **Check:** verify both versions are installed:
> ```bash
> ~/.pyenv/versions/3.13.0/bin/python3.13 --version
> # Python 3.13.0
>
> ~/.pyenv/versions/3.13.0t/bin/python3.13t --version
> # Python 3.13.0 (free-threading build)
> ```

### 3. Setting Up Virtual Environments

```bash
cd Diploma_2026

# GIL variant
~/.pyenv/versions/3.13.0/bin/python3.13 -m venv venv_gil

# No-GIL (free-threaded) variant
~/.pyenv/versions/3.13.0t/bin/python3.13t -m venv venv_nogil
```

### 4. Installing Python Dependencies

```bash
# GIL environment
venv_gil/bin/pip install --upgrade pip
venv_gil/bin/pip install -r requirements_gil.txt

# No-GIL environment
venv_nogil/bin/pip install --upgrade pip
venv_nogil/bin/pip install -r requirements_nogil.txt
```

> **Note:** `requirements_nogil.txt` excludes `numba` and `llvmlite` as they are currently incompatible with free-threaded Python.

### 5. Verification

```bash
# Check GIL status
venv_gil/bin/python -c "import sys; print('GIL enabled:', sys._is_gil_enabled())"
# GIL enabled: True

venv_nogil/bin/python -c "import sys; print('GIL enabled:', sys._is_gil_enabled())"
# GIL enabled: False

# Check core libraries
venv_gil/bin/python -c "import numpy, pandas, sklearn, PIL, psutil; print('All OK')"
venv_nogil/bin/python -c "import numpy, pandas, sklearn, PIL, psutil; print('All OK')"
```

## Dataset Downloading

Benchmarks use real-world datasets. The following script downloads them automatically:

```bash
venv_gil/bin/python scripts/download_real_datasets.py
```

| Dataset | Source | Size | Usage |
|---------|--------|------|-------|
| **Caltech-101** | `data.caltech.edu` | ~150 MB | Image Processing (9144 real photos) |
| **NYC Yellow Taxi** | `nyc.gov` TLC | ~48 MB | Data Preprocessing (~3M rows, Parquet) |

## Running Benchmarks

### Full Run (Recommended)

Multi-iteration run with CPU/Memory monitoring:

```bash
# 3 iterations (each iteration contains 3 internal runs = 9 total measurements)
venv_gil/bin/python scripts/run_multiple_benchmarks.py 3
```

Parameter is the number of iterations (default: 5). More iterations = better statistics.

**Approximate time** (3 iterations, Apple M-series):

| Workload | GIL | No-GIL | Total |
|----------|-----|--------|-------|
| Data Preprocessing | ~45s × 3 | ~65s × 3 | ~6 min |
| Image Processing | ~250s × 3 | ~265s × 3 | ~26 min |
| ML Training | ~250s × 3 | ~250s × 3 | ~25 min |
| Streaming | ~28s × 3 | ~33s × 3 | ~3 min |
| SIMD Vectorization | ~3s × 3 | ~2s × 3 | ~0.5 min |
| **Total** | | | **~60 min** |

Results are saved to `results/multi_run_master.json`.

### Individual Workloads

```bash
# Data Preprocessing (auto = real NYC Taxi data)
venv_gil/bin/python workloads/workload_data_preprocessing.py auto

# Image Processing
venv_gil/bin/python workloads/workload_image_processing.py images_input images_output

# ML Training (200K samples)
venv_gil/bin/python workloads/workload_ml.py 200000

# Streaming (50K events, 100K events/sec)
venv_gil/bin/python workloads/workload_streaming.py 50000 100000

# SIMD Vectorization (50M elements)
venv_gil/bin/python workloads/workload_simd.py 50000000
```

## Visualization

```bash
venv_gil/bin/python scripts/visualize_results.py
```

Generates 11 charts in `results/charts/`:

| Plot | Description |
|------|-------------|
| `01_overall_comparison.png` | General comparison of GIL vs No-GIL |
| `02_data_preprocessing.png` | Data Preprocessing scaling |
| `03_image_processing.png` | Image Processing scaling |
| `04_ml_training.png` | ML Training time by algorithm |
| `05_boxplots.png` | Distribution box plots |
| `05_streaming.png` | Streaming: latency and throughput |
| `06_simd_vectorization.png` | SIMD benchmarks comparison |
| `07_speedup_heatmap.png` | Heatmap of speedups across all modes |
| `08_confidence_intervals.png` | 95% confidence intervals |
| `09_cpu_utilization.png` | CPU utilization (mean / peak) |
| `10_memory_usage.png` | Memory usage (mean / peak) |

Also generates `results/statistical_analysis_multirun.json` with t-tests and p-values.

## Collected Metrics

| Metric | Tool | Description |
|--------|------|-------------|
| Execution time | `time.perf_counter()` | Wall time for each workload mode |
| CPU Utilization | `psutil` | Mean and peak % CPU usage |
| Memory Usage | `psutil` | Mean and peak RAM usage (MB) |
| Speedup | GIL time / No-GIL time | Acceleration factor without GIL |
| Scalability | 1/2/4/8 threads | Performance scaling with thread count |
| Statistical significance | Welch's t-test | p-values for measurement groups |

---
**Author:** Mykhailo Brodiuk — Diploma Project, 2026
