import time
import queue
from threading import Thread
from collections import deque
import random
import numpy as np
import sys
import json


class StreamProcessor:
    def __init__(self):
        self.event_queue = queue.Queue()
        self.processed_events = deque(maxlen=10000)
        self.latencies = []
        self.num_processed = 0

    def generate_events(self, num_events: int, events_per_second: int):
        """Generates events at a given rate"""
        interval = 1.0 / events_per_second if events_per_second > 0 else 0

        for i in range(num_events):
            event = {
                'id': i,
                'timestamp': time.time(),
                'data': np.random.randn(100).tolist()  # Simulate some data
            }
            self.event_queue.put(event)
            if interval > 0:
                time.sleep(interval)

    def process_event(self, event: dict):
        """Processes a single event"""
        # Simulate processing
        data = np.array(event['data'])
        result = float(np.mean(data))

        # Calculate latency
        latency = time.time() - event['timestamp']
        self.latencies.append(latency)

        self.processed_events.append({
            'id': event['id'],
            'result': result,
            'latency': latency
        })
        self.num_processed += 1

    def run_worker(self):
        """Worker thread for processing events"""
        while True:
            try:
                event = self.event_queue.get(timeout=1)
                self.process_event(event)
                self.event_queue.task_done()
            except queue.Empty:
                break

    def run_benchmark(self, num_workers: int, num_events: int, events_per_second: int):
        """Runs the benchmark"""
        # Reset state
        self.event_queue = queue.Queue()
        self.processed_events = deque(maxlen=10000)
        self.latencies = []
        self.num_processed = 0

        # Start generator thread
        generator = Thread(target=self.generate_events, args=(num_events, events_per_second))
        generator.start()

        # Start worker threads
        workers = []
        for _ in range(num_workers):
            worker = Thread(target=self.run_worker)
            worker.start()
            workers.append(worker)

        # Wait for completion
        generator.join()
        self.event_queue.join()

        for worker in workers:
            worker.join()

        # Calculate statistics
        if self.latencies:
            avg_latency = np.mean(self.latencies)
            p50_latency = np.percentile(self.latencies, 50)
            p95_latency = np.percentile(self.latencies, 95)
            p99_latency = np.percentile(self.latencies, 99)

            # Throughput
            if len(self.latencies) > 1:
                throughput = len(self.latencies) / (max(self.latencies) - min(self.latencies))
            else:
                throughput = 0

            return {
                'num_processed': self.num_processed,
                'avg_latency': avg_latency,
                'p50_latency': p50_latency,
                'p95_latency': p95_latency,
                'p99_latency': p99_latency,
                'throughput': throughput
            }

        return {
            'num_processed': 0,
            'avg_latency': 0,
            'p50_latency': 0,
            'p95_latency': 0,
            'p99_latency': 0,
            'throughput': 0
        }


def run_benchmark(num_events: int = 10000, events_per_second: int = 100, num_runs: int = 3):
    """Runs benchmark for all configurations"""
    results = {}

    worker_counts = [1, 2, 4, 8]

    for num_workers in worker_counts:
        print(f"Testing with {num_workers} workers...")
        mode_results = []

        for run in range(num_runs):
            processor = StreamProcessor()
            result = processor.run_benchmark(
                num_workers=num_workers,
                num_events=num_events,
                events_per_second=events_per_second
            )
            mode_results.append(result)

        # Average results
        avg_result = {
            'avg_latency': np.mean([r['avg_latency'] for r in mode_results]),
            'p50_latency': np.mean([r['p50_latency'] for r in mode_results]),
            'p95_latency': np.mean([r['p95_latency'] for r in mode_results]),
            'p99_latency': np.mean([r['p99_latency'] for r in mode_results]),
            'throughput': np.mean([r['throughput'] for r in mode_results])
        }

        results[f'workers_{num_workers}'] = avg_result
        print(f"  Workers: {num_workers}, Avg latency: {avg_result['avg_latency']*1000:.2f}ms, "
              f"Throughput: {avg_result['throughput']:.2f} events/s")

    return results


if __name__ == "__main__":
    num_events = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    events_per_second = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    print("=" * 60)
    print(f"Streaming Benchmark (events: {num_events}, rate: {events_per_second}/s)")
    print("=" * 60)

    results = run_benchmark(num_events=num_events, events_per_second=events_per_second)

    # Save results
    with open("results/streaming_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to results/streaming_results.json")
