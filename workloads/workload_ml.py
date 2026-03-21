from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification, make_regression
import numpy as np
from threading import Thread
from multiprocessing import Pool
import time
import sys
import json


class MLBenchmark:
    def __init__(self, n_samples: int = 200000, n_features: int = 50):
        # Generate datasets
        np.random.seed(42)
        print(f"Generating datasets with {n_samples} samples and {n_features} features...")
        self.X_reg, self.y_reg = make_regression(n_samples=n_samples, n_features=n_features, noise=0.1, random_state=42)
        self.X_clf, self.y_clf = make_classification(n_samples=n_samples, n_features=n_features, n_classes=2, random_state=42)

    def train_linear_regression(self, X, y):
        """Train Linear Regression"""
        model = LinearRegression()
        model.fit(X, y)
        return model

    def train_ransac(self, X, y):
        """Train RANSAC"""
        model = RANSACRegressor(max_trials=10)
        model.fit(X, y)
        return model

    def train_kmeans(self, X, n_clusters=8):
        """Train K-Means"""
        model = KMeans(n_clusters=n_clusters, n_init=10)
        model.fit(X)
        return model

    def train_random_forest(self, X, y, n_estimators=100):
        """Train Random Forest"""
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X, y)
        return model

    def parallel_train_threading(self, train_func, X, y, num_threads: int):
        """Parallel training using threads"""
        # Split data
        chunk_size = len(X) // num_threads
        X_chunks = [X[i:i+chunk_size] for i in range(0, len(X), chunk_size)]
        y_chunks = [y[i:i+chunk_size] for i in range(0, len(y), chunk_size)]

        models = [None] * num_threads

        def worker(idx, X_chunk, y_chunk):
            models[idx] = train_func(X_chunk, y_chunk)

        threads = []
        for idx, (X_chunk, y_chunk) in enumerate(zip(X_chunks, y_chunks)):
            t = Thread(target=worker, args=(idx, X_chunk, y_chunk))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        return models


def run_ml_benchmark(n_samples: int = 100000, n_features: int = 20, num_runs: int = 3):
    """Run ML benchmarks"""
    results = {}

    benchmark = MLBenchmark(n_samples=n_samples, n_features=n_features)

    # Linear Regression
    print("Testing Linear Regression...")
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        model = benchmark.train_linear_regression(benchmark.X_reg, benchmark.y_reg)
        times.append(time.perf_counter() - start)
    results['linear_regression'] = {"avg_time": np.mean(times), "all_times": times}
    print(f"  Linear Regression: {np.mean(times):.2f}s")

    # RANSAC (faster with fewer iterations)
    print("Testing RANSAC...")
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        model = benchmark.train_ransac(benchmark.X_reg, benchmark.y_reg)
        times.append(time.perf_counter() - start)
    results['ransac'] = {"avg_time": np.mean(times), "all_times": times}
    print(f"  RANSAC: {np.mean(times):.2f}s")

    # K-Means
    print("Testing K-Means...")
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        model = benchmark.train_kmeans(benchmark.X_clf, n_clusters=8)
        times.append(time.perf_counter() - start)
    results['kmeans'] = {"avg_time": np.mean(times), "all_times": times}
    print(f"  K-Means: {np.mean(times):.2f}s")

    # Random Forest (smaller for speed)
    print("Testing Random Forest...")
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        model = benchmark.train_random_forest(benchmark.X_clf, benchmark.y_clf, n_estimators=50)
        times.append(time.perf_counter() - start)
    results['random_forest'] = {"avg_time": np.mean(times), "all_times": times}
    print(f"  Random Forest: {np.mean(times):.2f}s")

    # Threading tests
    for num_threads in [2, 4, 8]:
        print(f"Testing Linear Regression with {num_threads} threads...")
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            models = benchmark.parallel_train_threading(
                benchmark.train_linear_regression,
                benchmark.X_reg,
                benchmark.y_reg,
                num_threads
            )
            times.append(time.perf_counter() - start)
        results[f'linear_regression_threaded_{num_threads}'] = {"avg_time": np.mean(times), "all_times": times}
        print(f"  Linear Regression ({num_threads} threads): {np.mean(times):.2f}s")

    return results


if __name__ == "__main__":
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 100000

    print("=" * 60)
    print(f"ML Benchmark (samples: {n_samples:,})")
    print("=" * 60)

    results = run_ml_benchmark(n_samples=n_samples)

    # Save results
    with open("results/ml_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to results/ml_results.json")
