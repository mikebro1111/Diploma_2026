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


    # -------------------------------------------------------------------------
    # 1. Linear Regression
    # -------------------------------------------------------------------------
    experiments = [
        ("linear_regression_seq", LinearRegression(), 1, None),
        # ("linear_reg_loky_4",     LinearRegression(), 4, 'loky'),
        # ("linear_reg_loky_8",     LinearRegression(), 8, 'loky'),
        # ("linear_reg_threading_2", LinearRegression(), 2, 'threading'),
        # ("linear_reg_threading_3", LinearRegression(), 3, 'threading'),
        ("linear_reg_threading_4", LinearRegression(), 4, 'threading'),
        # ("linear_reg_threading_5", LinearRegression(), 5, 'threading'),
        # ("linear_reg_threading_6", LinearRegression(), 6, 'threading'),
        # ("linear_reg_threading_7", LinearRegression(), 7, 'threading'),
        ("linear_reg_threading_8", LinearRegression(), 8, 'threading'),
        ("random_forest_seq",      RandomForestClassifier(n_estimators=30, n_jobs=1), 1, None),
        # ("random_forest_loky_8",   RandomForestClassifier(n_estimators=30, n_jobs=8), 8, 'loky'),
        # ("random_forest_threading_2", RandomForestClassifier(n_estimators=30, n_jobs=2), 2, 'threading'),
        # ("random_forest_threading_3", RandomForestClassifier(n_estimators=30, n_jobs=3), 3, 'threading'),
        ("random_forest_threading_4", RandomForestClassifier(n_estimators=30, n_jobs=4), 4, 'threading'),
        # ("random_forest_threading_5", RandomForestClassifier(n_estimators=30, n_jobs=5), 5, 'threading'),
        # ("random_forest_threading_6", RandomForestClassifier(n_estimators=30, n_jobs=6), 6, 'threading'),
        # ("random_forest_threading_7", RandomForestClassifier(n_estimators=30, n_jobs=7), 7, 'threading'),
        ("random_forest_threading_8", RandomForestClassifier(n_estimators=30, n_jobs=8), 8, 'threading'),
    ]

    for name, model, n_jobs, backend in experiments:
        times = []
        for _ in range(num_runs):
            if backend:
                with joblib.parallel_backend(backend, n_jobs=n_jobs):
                    start = time.perf_counter()
                    if "regression" in name:
                        model.fit(benchmark.X_reg, benchmark.y_reg)
                    else:
                        model.fit(benchmark.X_clf, benchmark.y_clf)
                    times.append(time.perf_counter() - start)
            else:
                # Sequential or manual n_jobs (seq)
                start = time.perf_counter()
                if "regression" in name:
                    model.fit(benchmark.X_reg, benchmark.y_reg)
                else:
                    model.fit(benchmark.X_clf, benchmark.y_clf)
                times.append(time.perf_counter() - start)
        
        avg = float(np.mean(times))
        results[name] = {"avg_time": avg, "all_times": times}
        print(f"  {name:<30}: {avg:.4f}s")

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
