from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification, make_regression
import joblib
import numpy as np
import time
import sys
import json
from pathlib import Path


class MLBenchmark:
    def __init__(self, n_samples: int = 1_000_000, n_features: int = 20):
        # Generate datasets
        np.random.seed(42)
        print(f"  Dataset: {n_samples:,} rows × {n_features} cols")
        # Multi-target regression (n_targets > 1) is required for scikit-learn to utilize n_jobs in LinearRegression
        self.X_reg, self.y_reg = make_regression(
            n_samples=n_samples, n_features=n_features, noise=0.1, n_targets=5, random_state=42
        )
        self.X_clf, self.y_clf = make_classification(
            n_samples=n_samples, n_features=n_features, n_classes=2, random_state=42
        )


def run_benchmark(n_samples: int = 1_000_000, n_features: int = 20, num_runs: int = 3) -> dict:
    """Run ML benchmarks focusing on scikit-learn's native n_jobs scaling via Joblib."""
    benchmark = MLBenchmark(n_samples=n_samples, n_features=n_features)
    results = {}

    def measure(name, func, *args, **kwargs):
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            func(*args, **kwargs)
            times.append(time.perf_counter() - start)
        
        avg = float(np.mean(times))
        results[name] = {"avg_time": avg, "all_times": times}
        print(f"  {name:<30}: {avg:.4f}s")

    # -------------------------------------------------------------------------
    # 1. Linear Regression
    # -------------------------------------------------------------------------
    def train_lr(n_jobs):
        model = LinearRegression(n_jobs=n_jobs)
        model.fit(benchmark.X_reg, benchmark.y_reg)

    def train_lr_backend(n_jobs, backend):
        with joblib.parallel_backend(backend, n_jobs=n_jobs):
            model = LinearRegression(n_jobs=n_jobs)
            model.fit(benchmark.X_reg, benchmark.y_reg)

    measure("linear_regression_seq", train_lr, n_jobs=None)

    # Note: Using backend='threading' reveals GIL bottlenecks in python code
    measure("linear_reg_threading_4", train_lr_backend, 4, "threading")
    measure("linear_reg_threading_8", train_lr_backend, 8, "threading")
    
    measure("linear_reg_loky_4", train_lr_backend, 4, "loky")
    measure("linear_reg_loky_8", train_lr_backend, 8, "loky")

    # -------------------------------------------------------------------------
    # 2. Random Forest
    # -------------------------------------------------------------------------
    def train_rf(n_jobs, backend):
        with joblib.parallel_backend(backend, n_jobs=n_jobs):
            # Reduced estimators to keep benchmark time reasonable on large data
            model = RandomForestClassifier(n_estimators=30, n_jobs=n_jobs, random_state=42)
            model.fit(benchmark.X_clf, benchmark.y_clf)

    measure("random_forest_seq", train_rf, n_jobs=None, backend="loky")
    measure("random_forest_threading_8", train_rf, n_jobs=8, backend="threading")
    measure("random_forest_loky_8", train_rf, n_jobs=8, backend="loky")

    return results


if __name__ == "__main__":
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    num_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    print("=" * 60)
    print(f"ML Benchmark (Native Joblib backends: Threading vs Loky)")
    print("=" * 60)

    results = run_benchmark(n_samples=n_samples, num_runs=num_runs)

    out = Path("results/ml_results.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\\nResults saved to {out}")
