from PIL import Image, ImageFilter
import os
from pathlib import Path
from threading import Thread
from multiprocessing import Pool
import time
import sys
import json

class ImageProcessor:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Find images
        self.images = []
        if self.input_dir.exists():
            self.images = list(self.input_dir.glob("*.jpg")) + \
                        list(self.input_dir.glob("*.png")) + \
                        list(self.input_dir.glob("*.jpeg"))

    def process_single_image(self, image_path: Path):
        """Process a single image"""
        try:
            img = Image.open(image_path)

            # Grayscale conversion
            img_gray = img.convert("L")

            # Resize
            img_resized = img_gray.resize((224, 224))

            # Apply Gaussian blur
            img_blurred = img_resized.filter(ImageFilter.GaussianBlur(radius=2))

            # Edge detection
            img_edges = img_resized.filter(ImageFilter.FIND_EDGES)

            # Save
            output_path = self.output_dir / f"processed_{image_path.name}"
            img_blurred.save(output_path)

            return True
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return False

    def process_sequential(self):
        """Sequential processing"""
        for img_path in self.images:
            self.process_single_image(img_path)

    def process_threading(self, num_threads: int):
        """Processing with threads"""
        if not self.images:
            return

        def worker(images_chunk):
            for img_path in images_chunk:
                self.process_single_image(img_path)

        # Split images into chunks
        chunk_size = max(1, len(self.images) // num_threads)
        chunks = [self.images[i:i+chunk_size] for i in range(0, len(self.images), chunk_size)]

        threads = []
        for chunk in chunks:
            t = Thread(target=worker, args=(chunk,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

    def process_multiprocessing(self, num_processes: int):
        """Processing with multiprocessing"""
        if not self.images:
            return

        # Need to use static function for multiprocessing
        with Pool(num_processes) as pool:
            pool.map(process_image_wrapper, self.images)


def process_image_wrapper(image_path):
    """Wrapper for multiprocessing"""
    from pathlib import Path
    from PIL import Image, ImageFilter

    try:
        img = Image.open(image_path)
        img_gray = img.convert("L")
        img_resized = img_gray.resize((224, 224))
        img_blurred = img_resized.filter(ImageFilter.GaussianBlur(radius=2))
        return True
    except:
        return False


def run_benchmark(input_dir: str, output_dir: str, num_runs: int = 3):
    """Runs benchmark for all modes"""
    results = {}

    modes = [
        ("sequential", lambda: ImageProcessor(input_dir, output_dir).process_sequential()),
        ("threading_2", lambda: ImageProcessor(input_dir, output_dir).process_threading(2)),
        ("threading_4", lambda: ImageProcessor(input_dir, output_dir).process_threading(4)),
        ("threading_8", lambda: ImageProcessor(input_dir, output_dir).process_threading(8)),
        ("multiprocessing_2", lambda: ImageProcessor(input_dir, output_dir).process_multiprocessing(2)),
        ("multiprocessing_4", lambda: ImageProcessor(input_dir, output_dir).process_multiprocessing(4)),
        ("multiprocessing_8", lambda: ImageProcessor(input_dir, output_dir).process_multiprocessing(8)),
    ]

    for mode_name, mode_func in modes:
        times = []
        for _ in range(num_runs):
            # Clean output dir
            import shutil
            out = Path(output_dir)
            if out.exists():
                shutil.rmtree(out)
            out.mkdir(exist_ok=True)

            start = time.perf_counter()
            mode_func()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        results[mode_name] = {
            "avg_time": avg_time,
            "all_times": times
        }
        print(f"{mode_name}: {avg_time:.4f}s")

    return results


if __name__ == "__main__":
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "./images_input"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./images_output"

    print("=" * 60)
    print("Image Processing Benchmark")
    print("=" * 60)

    results = run_benchmark(input_dir, output_dir)

    # Save results
    with open("results/image_processing_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to results/image_processing_results.json")
