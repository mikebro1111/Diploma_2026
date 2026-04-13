import time
import queue
from threading import Thread
import json
import random
import sys
import numpy as np


def _generate_json_payload(size=100):
    """Generates a complex dictionary to simulate an API event payload."""
    return {
        "id": random.randint(1, 100000),
        "values": [random.random() for _ in range(size)],
        "metadata": {
            "source": ["app", "web", "mobile"][random.randint(0, 2)],
            "tags": ["prod", "us-east", "v2", f"tag_{random.randint(0,10)}"]
        }
    }


class StreamProcessor:
    def __init__(self):
        self.event_queue = queue.Queue()
        self.latencies = []
        self.num_processed = 0

    def generate_events(self, num_events: int):
        """Pre-generates events to isolate processing time from generation."""
        for i in range(num_events):
            # Serialize to string to force the worker to deserialize
            raw_str = json.dumps(_generate_json_payload())
            self.event_queue.put({"timestamp": time.time(), "raw": raw_str})
        
        # Poison pills to safely stop worker threads
        for _ in range(64):
            self.event_queue.put(None)

    def process_event(self, event: dict):
        """
        Pure Python processing. 
        No NumPy, No C-extensions computationally heavy tasks.
        If the GIL is present, this will bottleneck extremely hard.
        """
        # 1. Parse JSON from string
        parsed = json.loads(event["raw"])
        
        # 2. Pure Python Math loops
        vals = parsed["values"]
        sum_val = sum(vals)
        max_val = max(vals)
        min_val = min(vals)
        mean_val = sum_val / len(vals)
        
        # 3. String manipulation
        tags = "-".join(parsed["metadata"]["tags"]).upper()
        
        # 4. Dictionary allocation
        result = {
            "sum": sum_val, 
            "max": max_val, 
            "min": min_val, 
            "mean": mean_val, 
            "tag_hash": hash(tags)
        }

        # Track throughput
        self.num_processed += 1

    def run_worker(self):
        """Thread worker fetching events"""
        while True:
            event = self.event_queue.get()
            if event is None:
                self.event_queue.task_done()
                break
            self.process_event(event)
            self.event_queue.task_done()

    def run_benchmark(self, num_workers: int, num_events: int):
        # Reset queue state
        self.event_queue = queue.Queue()
        self.num_processed = 0

        # Pre-generate
        self.generate_events(num_events)

        start_time = time.perf_counter()
        workers = []
        for _ in range(num_workers):
            worker = Thread(target=self.run_worker)
            worker.start()
            workers.append(worker)

        for worker in workers:
            worker.join()
        
        total_time = time.perf_counter() - start_time

        if self.num_processed > 0:
            return {
                'num_processed': self.num_processed,
                # Simulate latency as absolute time for the visualizer script semantics
                'avg_latency': total_time, 
                'throughput': self.num_processed / (total_time + 1e-9)
            }
            
        return {'num_processed': 0, 'avg_latency': 0, 'throughput': 0}


def run_benchmark(num_events: int = 50000, events_per_second: int = 100, num_runs: int = 1):
    """Run pure-Python workload threaded scaling."""
    results = {}
    worker_counts = [1, 2, 4, 8]

    # Pre-generate serialized event strings ONCE — isolate processing from generation
    print(f"  Pre-generating {num_events:,} events...")
    pregenerated = [
        {"timestamp": 0.0, "raw": json.dumps(_generate_json_payload())}
        for _ in range(num_events)
    ]

    for num_workers in worker_counts:
        print(f"Testing with {num_workers} workers...")
        mode_results = []
        for run in range(num_runs):
            processor = StreamProcessor()
            # Fill queue from pre-generated events (no serialization overhead)
            for ev in pregenerated:
                processor.event_queue.put(ev)
            # Add poison pills
            for _ in range(64):
                processor.event_queue.put(None)

            start_time = time.perf_counter()
            workers = []
            for _ in range(num_workers):
                worker = Thread(target=processor.run_worker)
                worker.start()
                workers.append(worker)
            for worker in workers:
                worker.join()
            total_time = time.perf_counter() - start_time

            mode_results.append({
                'num_processed': processor.num_processed,
                'avg_latency': total_time,
                'throughput': processor.num_processed / (total_time + 1e-9)
            })

        avg_result = {
            'avg_latency': float(np.min([r['avg_latency'] for r in mode_results])),
            'throughput': float(np.max([r['throughput'] for r in mode_results]))
        }

        results[f'workers_{num_workers}'] = {
            'avg_latency': {'mean': avg_result['avg_latency']},
            'throughput': {'mean': avg_result['throughput']}
        }

        print(f"  Workers: {num_workers}, Time: {avg_result['avg_latency']:.2f}s, Throughput: {avg_result['throughput']:.2f} obj/s")

    return results


if __name__ == "__main__":
    num_events = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    num_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    print("=" * 60)
    print(f"Streaming Benchmark (Pure Python JSON/Math - no NumPy CPU)")
    print("=" * 60)
    
    results = run_benchmark(num_events=num_events, num_runs=num_runs)
    
    with open("results/streaming_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to results/streaming_results.json")
