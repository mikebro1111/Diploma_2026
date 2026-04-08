from PIL import Image, ImageFilter
import os
from pathlib import Path
from threading import Thread
from multiprocessing import Pool
import time
import sys
import json


# ---------------------------------------------------------------------------
# Single, top-level picklable function — identical pipeline for ALL modes.
# Used by sequential loop, threading workers, AND multiprocessing pool.
# ---------------------------------------------------------------------------
def _process_one(args):
    """
    Process a single image through the full pipeline.
    Args is a tuple (image_path, output_dir) so it can be pickled for Pool.
    Pipeline (same for every mode):
      1. Open image
      2. Convert to grayscale (L)
      3. Resize to 224×224
      4. Apply Gaussian blur (radius=2)
      5. Detect edges (FIND_EDGES)
      6. Save blurred result to output_dir
    """
    image_path, output_dir = args
    image_path = Path(image_path)
    output_dir = Path(output_dir)

    try:
        img = Image.open(image_path)

        # Step 1 – Grayscale conversion
        img_gray = img.convert("L")

        # Step 2 – Resize
        img_resized = img_gray.resize((224, 224))

        # Step 3 – Gaussian blur
        img_blurred = img_resized.filter(ImageFilter.GaussianBlur(radius=2))

        # Step 4 – Edge detection  ← identical in every mode
        img_edges = img_resized.filter(ImageFilter.FIND_EDGES)

        # Step 5 – Save           ← identical in every mode
        output_path = output_dir / f"processed_{image_path.name}"
        img_blurred.save(output_path)

        return True
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False


class ImageProcessor:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.images = []
        if self.input_dir.exists():
            self.images = (
                list(self.input_dir.glob("*.jpg")) +
                list(self.input_dir.glob("*.png")) +
                list(self.input_dir.glob("*.jpeg"))
            )

    # -- Sequential -----------------------------------------------------------
    def process_sequential(self):
        """Sequential processing — calls _process_one for every image."""
        for img_path in self.images:
            _process_one((img_path, self.output_dir))

    # -- Threading ------------------------------------------------------------
    def process_threading(self, num_threads: int):
        """Threading — each worker calls _process_one (same as sequential)."""
        if not self.images:
            return

        chunk_size = max(1, len(self.images) // num_threads)
        chunks = [
            self.images[i:i + chunk_size]
            for i in range(0, len(self.images), chunk_size)
        ]

        def worker(images_chunk):
            for img_path in images_chunk:
                _process_one((img_path, self.output_dir))

        threads = []
        for chunk in chunks:
            t = Thread(target=worker, args=(chunk,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    # -- Multiprocessing ------------------------------------------------------
    def process_multiprocessing(self, num_processes: int):
        """Multiprocessing — pool.map over _process_one (same pipeline)."""
        if not self.images:
            return

        args = [(img_path, self.output_dir) for img_path in self.images]
        with Pool(num_processes) as pool:
            pool.map(_process_one, args)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def run_benchmark(input_dir: str, output_dir: str, num_runs: int = 1):
    """Runs benchmark for all modes using the unified _process_one pipeline."""
    results = {}

    # Pre-create one ImageProcessor to avoid re-scanning the directory in each lambda
    processor = ImageProcessor(input_dir, output_dir)

    modes = [
        ("sequential",       lambda: processor.process_sequential()),
        # ("threading_1",       lambda: processor.process_threading(1)),
        ("threading_2",       lambda: processor.process_threading(2)),
        # ("threading_3",       lambda: processor.process_threading(3)),
        ("threading_4",       lambda: processor.process_threading(4)),
        # ("threading_5",       lambda: processor.process_threading(5)),
        # ("threading_6",       lambda: processor.process_threading(6)),
        # ("threading_7",       lambda: processor.process_threading(7)),
        ("threading_8",       lambda: processor.process_threading(8)),
        # ("multiprocessing_2", lambda: processor.process_multiprocessing(2)),
        # ("multiprocessing_4", lambda: processor.process_multiprocessing(4)),
        # ("multiprocessing_8", lambda: processor.process_multiprocessing(8)),
    ]

    for mode_name, mode_func in modes:
        times = []
        for _ in range(num_runs):
            import shutil
            out = Path(output_dir)
            if out.exists():
                shutil.rmtree(out, ignore_errors=True)
            out.mkdir(exist_ok=True, parents=True)

            start = time.perf_counter()
            mode_func()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        results[mode_name] = {
            "avg_time": avg_time,
            "all_times": times,
        }
        print(f"{mode_name}: {avg_time:.4f}s")

    return results


if __name__ == "__main__":
    input_dir  = sys.argv[1] if len(sys.argv) > 1 else "./images_input"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./images_output"
    num_runs   = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    print("=" * 60)
    print("Image Processing Benchmark")
    print("=" * 60)

    results = run_benchmark(input_dir, output_dir, num_runs=num_runs)

    with open("results/image_processing_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to results/image_processing_results.json")
