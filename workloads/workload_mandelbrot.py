import time
import json
import argparse
import sys
import threading
from typing import List
from pathlib import Path

def mandelbrot(c, max_iter):
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z*z + c
        n += 1
    return n

def mandelbrot_range(x_start, x_end, y_start, y_end, width, height, max_iter):
    rows = []
    for py in range(height):
        row = []
        y0 = y_start + (py / height) * (y_end - y_start)
        for px in range(width):
            x0 = x_start + (px / width) * (x_end - x_start)
            c = complex(x0, y0)
            row.append(mandelbrot(c, max_iter))
        rows.append(row) # Store the computed row
    return len(rows)

def worker(width, height, max_iter, start_row, end_row, results, idx):
    # This worker computes a horizontal strip of the Mandelbrot set
    count = 0
    for py in range(start_row, end_row):
        y0 = -1.0 + (py / height) * 2.0
        for px in range(width):
            x0 = -2.0 + (px / width) * 3.0
            c = complex(x0, y0)
            
            # Inline mandelbrot for speed
            z = 0
            n = 0
            while abs(z) <= 2 and n < max_iter:
                z = z*z + c
                n += 1
            count += 1
    results[idx] = count

def run_mandelbrot_threaded(width, height, max_iter, num_threads):
    threads = []
    results = [0] * num_threads
    rows_per_thread = height // num_threads
    
    start_time = time.perf_counter()
    for i in range(num_threads):
        start_row = i * rows_per_thread
        end_row = height if i == num_threads - 1 else (i + 1) * rows_per_thread
        t = threading.Thread(target=worker, args=(width, height, max_iter, start_row, end_row, results, i))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
    end_time = time.perf_counter()
    
    return end_time - start_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("size", type=int, default=1000)
    parser.add_argument("num_runs", type=int, default=1)
    args = parser.parse_args()

    # Fixed parameters for consistent benchmarking
    WIDTH = args.size
    HEIGHT = args.size
    MAX_ITER = 200
    
    # We test sequential, 2, 4, 8 threads
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
            t = run_mandelbrot_threaded(WIDTH, HEIGHT, MAX_ITER, n_threads)
            times.append(t)
        
        final_results[name] = {
            "min_time": min(times),
            "max": max(times),
            "avg_time": sum(times) / len(times),
            "all_times": times
        }
        print(f"  {name}: {final_results[name]['min_time']:.4f}s")

    out = Path("results/mandelbrot_results.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(final_results, f, indent=2)
