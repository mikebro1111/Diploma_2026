"""
Data Preprocessing Benchmark — Pure NumPy Edition
==================================================
Uses numpy arrays exclusively (no pandas get_dummies, no DataFrame.copy()).

Why pure numpy:
  - numpy operations release the GIL (both GIL and no-GIL builds)
  - no Python-object allocation storm (no dict/list churn from get_dummies)
  - free-threaded Python (no-GIL) shows genuine parallelism benefit
    because 8 threads can run numpy kernels truly concurrently
  - GIL Python gains from numpy's C-level GIL releases too,
    but is limited by the GIL between numpy calls

Pipeline (identical for sequential / threading / multiprocessing):
    1.  Z-score normalisation  (column-wise)
    2.  Elementwise feature engineering:
        sum, product, ratio, diff, hypot, exp, log, sin, polynomial
    3.  Row-wise statistics: mean, std, max, min
    → result shape:  (N, n_cols + 13)
"""
import numpy as np
from threading import Thread
from multiprocessing import Pool
import time
import sys
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Core compute function — pure numpy, identical for ALL modes
# ---------------------------------------------------------------------------
def _preprocess_array(chunk: np.ndarray) -> np.ndarray:
    """
    Heavy numpy feature-engineering on a 2-D float64 array.
    Releases the GIL during every numpy call, so truly benefits from
    free-threaded parallelism when run in threads.
    """
    # ── 1. Z-score normalisation ──────────────────────────────────────────
    means = chunk.mean(axis=0)
    stds  = chunk.std(axis=0) + 1e-8
    n     = (chunk - means) / stds          # shape (rows, cols)

    # ── 2. Pair-wise feature engineering ─────────────────────────────────
    c0, c1, c2, c3 = n[:, 0], n[:, 1], n[:, 2], n[:, 3]

    feat_sum   = c0 + c1
    feat_prod  = c0 * c1
    feat_ratio = c0 / (c1 + 1e-8)
    feat_diff  = c2 - c3
    feat_hypot = np.sqrt(c0 ** 2 + c1 ** 2)
    feat_exp   = np.exp(np.clip(c2, -5.0, 5.0))
    feat_log   = np.log1p(np.abs(c3))
    feat_sin   = np.sin(c0 * np.pi)
    feat_poly  = c0 ** 2 + c1 ** 3 + c2 ** 2

    # ── 3. Row-wise statistics ────────────────────────────────────────────
    feat_mean = n.mean(axis=1)
    feat_std  = n.std(axis=1)
    feat_max  = n.max(axis=1)
    feat_min  = n.min(axis=1)

    # ── 4. Concatenate all features ───────────────────────────────────────
    extras = np.column_stack([
        feat_sum, feat_prod, feat_ratio, feat_diff,
        feat_hypot, feat_exp, feat_log, feat_sin, feat_poly,
        feat_mean, feat_std, feat_max, feat_min,
    ])
    return np.hstack([n, extras])   # (rows, cols+13)


# ---------------------------------------------------------------------------
# Worker wrapper (top-level, picklable for multiprocessing.Pool)
# ---------------------------------------------------------------------------
def _mp_worker(chunk: np.ndarray) -> np.ndarray:
    """Identical pipeline — used by Pool.map."""
    return _preprocess_array(chunk)


# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------
class DataPreprocessor:
    def __init__(self, num_rows: int = 10_000_000, num_cols: int = 8):
        """
        Generate a large float64 matrix that mimics tabular feature data.
        num_cols >= 4 required for the full feature-engineering pipeline.
        """
        np.random.seed(42)
        self.data = np.random.randn(num_rows, num_cols).astype(np.float64)
        # Add a few skewed / non-negative columns (more realistic)
        self.data[:, 2] = np.abs(self.data[:, 2]) * 10    # trip_distance-like
        self.data[:, 3] = np.random.exponential(5, num_rows)  # fare-like
        print(f"  Dataset: {num_rows:,} rows x {num_cols} cols  "
              f"({self.data.nbytes / 1e6:.0f} MB)")

    # ── Sequential ────────────────────────────────────────────────────────
    def process_sequential(self) -> np.ndarray:
        """Single-threaded baseline."""
        return _preprocess_array(self.data)

    # ── Threading ─────────────────────────────────────────────────────────
    def process_threading(self, num_threads: int) -> np.ndarray:
        """
        Split data into equal row-chunks; each thread calls _preprocess_array.
        With no-GIL Python, threads run truly in parallel (numpy releases GIL).
        """
        indexes = np.array_split(np.arange(len(self.data)), num_threads)
        results = np.empty((self.data.shape[0], self.data.shape[1] + 13), dtype=self.data.dtype)

        def worker(idx: int, rows: np.ndarray):
            results[rows] = _preprocess_array(self.data[rows])

        threads = [Thread(target=worker, args=(i, idx))
                   for i, idx in enumerate(indexes)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        return results

    # ── Multiprocessing ───────────────────────────────────────────────────
    def process_multiprocessing(self, num_processes: int) -> np.ndarray:
        """
        Split data into chunks; each subprocess calls _mp_worker (identical).
        True parallelism on both GIL and no-GIL builds.
        """
        chunks = np.array_split(self.data, num_processes)
        with Pool(num_processes) as pool:
            results = pool.map(_mp_worker, chunks)
        return np.vstack(results)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def run_benchmark(num_rows: int = 10_000_000, num_cols: int = 8,
                  num_runs: int = 1) -> dict:
    """Run all modes and return timing results."""
    preprocessor = DataPreprocessor(num_rows=num_rows, num_cols=num_cols)
    results      = {}

    modes = [
        ("sequential",        lambda: preprocessor.process_sequential()),
        # ("threading_1",       lambda: preprocessor.process_threading(1)),
        ("threading_2",       lambda: preprocessor.process_threading(2)),
        # ("threading_3",       lambda: preprocessor.process_threading(3)),
        ("threading_4",       lambda: preprocessor.process_threading(4)),
        # ("threading_5",       lambda: preprocessor.process_threading(5)),
        # ("threading_6",       lambda: preprocessor.process_threading(6)),
        # ("threading_7",       lambda: preprocessor.process_threading(7)),
        ("threading_8",       lambda: preprocessor.process_threading(8)),
        # ("multiprocessing_2", lambda: DataPreprocessor(num_rows).process_multiprocessing(2)),
        # ("multiprocessing_4", lambda: DataPreprocessor(num_rows).process_multiprocessing(4)),
        # ("multiprocessing_8", lambda: DataPreprocessor(num_rows).process_multiprocessing(8)),
    ]

    for mode_name, mode_func in modes:
        times = []
        for _ in range(num_runs):
            start   = time.perf_counter()
            _       = mode_func()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = float(np.mean(times))
        results[mode_name] = {
            "avg_time":  avg_time,
            "all_times": times,
        }
        print(f"  {mode_name:<22}: {avg_time:.4f}s")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    num_rows = 10_000_000
    num_runs = 1
    if len(sys.argv) > 1:
        try:
            num_rows = int(sys.argv[1])
        except ValueError:
            pass
    if len(sys.argv) > 2:
        try:
            num_runs = int(sys.argv[2])
        except ValueError:
            pass

    print("=" * 60)
    print(f"Data Preprocessing Benchmark (pure numpy)")
    print(f"  Rows: {num_rows:,} | Runs: {num_runs}")
    print("=" * 60)

    results = run_benchmark(num_rows=num_rows, num_runs=num_runs)

    out = Path("results/data_preprocessing_results.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out}")
