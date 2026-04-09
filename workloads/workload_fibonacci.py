import time
import json
import argparse
import threading

def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)

def fib_worker(n, results, idx):
    results[idx] = fib(n)

def run_fibonacci_threaded(n, num_tasks, num_threads):
    # We run 'num_tasks' instances of fib(n) across 'num_threads'
    threads = []
    results = [0] * num_tasks
    
    # Simple chunking for threads
    tasks_per_thread = num_tasks // num_threads
    
    def thread_group_worker(task_indices):
        for idx in task_indices:
            results[idx] = fib(n)

    start_time = time.perf_counter()
    for i in range(num_threads):
        start_idx = i * tasks_per_thread
        end_idx = num_tasks if i == num_threads - 1 else (i + 1) * tasks_per_thread
        t = threading.Thread(target=thread_group_worker, args=(range(start_idx, end_idx),))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
    end_time = time.perf_counter()
    
    return end_time - start_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int, default=34) # fib(34) takes some time
    parser.add_argument("num_tasks", type=int, default=8) # Total tasks to run
    parser.add_argument("num_runs", type=int, default=1)
    args = parser.parse_args()
    PARALLEL_INSTANCES = 24  # Increased from 8 to 24 (3x work)
    FIB_N = 34

    N = args.n
    NUM_TASKS = PARALLEL_INSTANCES
    
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
            t = run_fibonacci_threaded(N, NUM_TASKS, n_threads)
            times.append(t)
        
        final_results[name] = {
            "min_time": min(times),
            "max": max(times),
            "avg_time": sum(times) / len(times),
            "all_times": times
        }
        print(f"  {name}: {final_results[name]['min_time']:.4f}s")

    with open("fibonacci_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
