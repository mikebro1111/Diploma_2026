import numpy as np
import time
from threading import Thread
import sys
import json

# Try to import numba
try:
    from numba import jit, vectorize, float64, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: Numba not available, some tests will be skipped")


class SIMDBenchmark:
    def __init__(self, array_size: int = 10_000_000):
        """Initialize with large arrays for SIMD testing"""
        np.random.seed(42)
        self.array_size = array_size
        self.a = np.random.rand(array_size).astype(np.float64)
        self.b = np.random.rand(array_size).astype(np.float64)
        self.result = np.zeros(array_size, dtype=np.float64)

    # Pure Python implementation (no SIMD)
    def add_python(self, a, b):
        """Pure Python addition - slow, no vectorization"""
        result = []
        for i in range(len(a)):
            result.append(a[i] + b[i])
        return result

    # NumPy vectorized (uses SIMD internally)
    def add_numpy(self, a, b):
        """NumPy vectorized addition - uses SIMD"""
        return a + b

    def multiply_numpy(self, a, b):
        """NumPy vectorized multiplication"""
        return a * b

    def complex_numpy_operation(self, a, b):
        """Complex vectorized operation"""
        return np.sqrt(a ** 2 + b ** 2) * np.sin(a)

    # Numba JIT compiled
    if NUMBA_AVAILABLE:
        @staticmethod
        @jit(nopython=True, parallel=True)
        def add_numba_parallel(a, b):
            """Numba parallel addition - explicit SIMD + threading"""
            return a + b

        @staticmethod
        @jit(nopython=True, parallel=False)
        def add_numba_serial(a, b):
            """Numba serial addition - SIMD but no threading"""
            return a + b

    def add_numpy_threaded(self, num_threads: int):
        """Test SIMD operations with threading"""
        chunk_size = self.array_size // num_threads
        results = [None] * num_threads

        def worker_numpy(idx, start_idx, end_idx):
            """Thread worker using NumPy"""
            results[idx] = self.a[start_idx:end_idx] + self.b[start_idx:end_idx]

        # NumPy with threads
        threads = []
        start_time = time.perf_counter()
        for i in range(num_threads):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < num_threads - 1 else self.array_size
            t = Thread(target=worker_numpy, args=(i, start_idx, end_idx))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        return time.perf_counter() - start_time

    def benchmark_sequential(self):
        """Baseline: sequential NumPy operations"""
        results = {}

        # Pure Python (very slow, for reference) - use smaller array
        print("Testing Pure Python...")
        small_a = self.a[:10000]
        small_b = self.b[:10000]
        start = time.perf_counter()
        _ = self.add_python(small_a.tolist(), small_b.tolist())
        results['python'] = time.perf_counter() - start

        # NumPy vectorized
        print("Testing NumPy...")
        start = time.perf_counter()
        _ = self.add_numpy(self.a, self.b)
        results['numpy'] = time.perf_counter() - start

        # NumPy complex operation
        print("Testing NumPy complex...")
        start = time.perf_counter()
        _ = self.complex_numpy_operation(self.a, self.b)
        results['numpy_complex'] = time.perf_counter() - start

        # Numba if available
        if NUMBA_AVAILABLE:
            print("Testing Numba Serial...")
            start = time.perf_counter()
            _ = self.add_numba_serial(self.a, self.b)
            results['numba_serial'] = time.perf_counter() - start

            print("Testing Numba Parallel...")
            start = time.perf_counter()
            _ = self.add_numba_parallel(self.a, self.b)
            results['numba_parallel'] = time.perf_counter() - start

        return results

    def benchmark_threading(self):
        """Test threading with NumPy"""
        results = {}

        for num_threads in [2, 4, 8]:
            print(f"Testing NumPy threaded ({num_threads} threads)...")
            start = time.perf_counter()
            _ = self.add_numpy_threaded(num_threads)
            results[f'numpy_threaded_{num_threads}'] = time.perf_counter() - start

        return results

    def benchmark_matrix_operations(self):
        """Test matrix operations (more complex SIMD)"""
        # Create matrices
        size = 1000
        A = np.random.rand(size, size)
        B = np.random.rand(size, size)

        results = {}

        # Matrix multiplication (heavily uses SIMD/BLAS)
        print("Testing Matrix Multiplication...")
        start = time.perf_counter()
        C = np.dot(A, B)
        results['matmul'] = time.perf_counter() - start

        # Element-wise operations
        print("Testing Element-wise ops...")
        start = time.perf_counter()
        C = A * B + A / (B + 1)
        results['elementwise'] = time.perf_counter() - start

        # Reductions (sum, mean)
        print("Testing Reductions...")
        start = time.perf_counter()
        s = np.sum(A)
        m = np.mean(B)
        results['reductions'] = time.perf_counter() - start

        return results


def run_simd_benchmark(array_size: int = 10_000_000, num_runs: int = 3):
    """Run SIMD benchmarks"""
    results = {}

    print("=" * 60)
    print(f"SIMD/Vectorization Benchmark (Array size: {array_size:,})")
    print(f"Numba available: {NUMBA_AVAILABLE}")
    print("=" * 60)

    benchmark = SIMDBenchmark(array_size=array_size)

    # Sequential benchmarks
    print("\n### Sequential Operations ###")
    seq_results = benchmark.benchmark_sequential()
    for name, time_taken in seq_results.items():
        print(f"{name:20s}: {time_taken:.4f}s")
    results['sequential'] = seq_results

    # Threading benchmarks
    print("\n### Threaded Operations ###")
    thread_results = benchmark.benchmark_threading()
    for name, time_taken in thread_results.items():
        print(f"{name:20s}: {time_taken:.4f}s")
    results['threaded'] = thread_results

    # Matrix operations
    print("\n### Matrix Operations ###")
    matrix_results = benchmark.benchmark_matrix_operations()
    for name, time_taken in matrix_results.items():
        print(f"{name:20s}: {time_taken:.4f}s")
    results['matrix'] = matrix_results

    # Calculate speedups
    print("\n### Speedups ###")
    if 'numpy' in seq_results and 'python' in seq_results:
        speedup = seq_results['python'] / seq_results['numpy']
        print(f"NumPy vs Python: {speedup:.2f}x")
        results['speedup_numpy_vs_python'] = speedup

    if NUMBA_AVAILABLE and 'numba_parallel' in seq_results:
        if 'numba_serial' in seq_results:
            speedup = seq_results['numba_serial'] / seq_results['numba_parallel']
            print(f"Numba Parallel vs Serial: {speedup:.2f}x")

    return results


if __name__ == "__main__":
    array_size = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000

    results = run_simd_benchmark(array_size=array_size)

    # Save results
    with open("results/simd_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    print("\nResults saved to results/simd_results.json")
