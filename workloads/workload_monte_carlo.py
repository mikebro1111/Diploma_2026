import time
import json
import argparse
import random
import threading

def monte_carlo_worker(num_samples, results, idx):
    count = 0
    # Use a thread-local random instance or just random.random (protected by GIL normally)
    # In No-GIL, random is thread-safe.
    for _ in range(num_samples):
        x = random.random()
        y = random.random()
        if x*x + y*y <= 1.0:
            count += 1
    results[idx] = count

def run_monte_carlo_threaded(total_samples, num_threads):
    samples_per_thread = total_samples // num_threads
    threads = []
    results = [0] * num_threads
    
    start_time = time.perf_counter()
    for i in range(num_threads):
        t = threading.Thread(target=monte_carlo_worker, args=(samples_per_thread, results, i))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
    end_time = time.perf_counter()
    
    pi_est = (sum(results) / total_samples) * 4
    # print(f"Pi est: {pi_est}")
    
    return end_time - start_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("total_samples", type=int, default=30_000_000)
    parser.add_argument("num_runs", type=int, default=1)
    args = parser.parse_args()

    total_samples = args.total_samples
    
    variants = {
        "sequential": 1,
        "threading_2": 2,
        "threading_4": 4,
        "threading_8": 8
    }
    
    final_results = {}
    
    for name, n_threads in variants.items():
        times = []
        for _ in range(args.num_runs):
            t = run_monte_carlo_threaded(total_samples, n_threads)
            times.append(t)
        
        final_results[name] = {
            "min_time": min(times),
            "max": max(times),
            "avg_time": sum(times) / len(times),
            "all_times": times
        }
        print(f"  {name}: {final_results[name]['min_time']:.4f}s")

    with open("monte_carlo_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
