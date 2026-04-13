from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification, make_regression
import joblib
import numpy as np
import time
import sys
import json
from pathlib import Path


def run_benchmark(n_samples: int = 1_000_000, n_features: int = 20, num_runs: int = 1) -> dict:
    """Run ML benchmarks focusing on scikit-learn's native n_jobs scaling via Joblib."""
    # Scale requirements: 10x Linear Regression, 0.5x Random Forest
    n_samples_lr = n_samples * 10
    n_samples_rf = n_samples // 2

    print(f"  Linear Regression Dataset: {n_samples_lr:,} rows x {n_features} cols")
    X_reg, y_reg = make_regression(
        n_samples=n_samples_lr, n_features=n_features, noise=0.1, n_targets=5, random_state=42
    )
    print(f"  Random Forest Dataset: {n_samples_rf:,} rows x {n_features} cols")
    X_clf, y_clf = make_classification(
        n_samples=n_samples_rf, n_features=n_features, n_classes=2, random_state=42
    )

    results = {}

    experiments = [
        ("linear_regression_seq", LinearRegression(), 1, None, X_reg, y_reg),
        ("linear_reg_threading_2", LinearRegression(), 2, 'threading', X_reg, y_reg),
        ("linear_reg_threading_4", LinearRegression(), 4, 'threading', X_reg, y_reg),
        ("linear_reg_threading_8", LinearRegression(), 8, 'threading', X_reg, y_reg),
        
        ("random_forest_seq",      RandomForestClassifier(n_estimators=30, n_jobs=1), 1, None, X_clf, y_clf),
        ("random_forest_threading_2", RandomForestClassifier(n_estimators=30, n_jobs=2), 2, 'threading', X_clf, y_clf),
        ("random_forest_threading_4", RandomForestClassifier(n_estimators=30, n_jobs=4), 4, 'threading', X_clf, y_clf),
        ("random_forest_threading_8", RandomForestClassifier(n_estimators=30, n_jobs=8), 8, 'threading', X_clf, y_clf),
    ]

    for name, model, n_jobs, backend, X, y in experiments:
        times = []
        for _ in range(num_runs):
            if backend:
                with joblib.parallel_backend(backend, n_jobs=n_jobs):
                    start = time.perf_counter()
                    model.fit(X, y)
                    times.append(time.perf_counter() - start)
            else:
                start = time.perf_counter()
                model.fit(X, y)
                times.append(time.perf_counter() - start)
        
        avg = float(np.mean(times))
        results[name] = {"avg_time": avg, "all_times": times}
        print(f"  {name:<30}: {avg:.4f}s")

    return results


if __name__ == "__main__":
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    num_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    print("=" * 60)
    print(f"ML Benchmark (Native Joblib backends: Threading vs Loky)")
    print("=" * 60)

    results = run_benchmark(n_samples=n_samples, num_runs=num_runs)

    out = Path("results/ml_results.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out}")
