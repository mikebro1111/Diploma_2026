"""
Download real-world datasets for benchmarking.

Datasets:
  1. Caltech-101 images (~9K real photos, various sizes) - for Image Processing
  2. NYC Yellow Taxi trip data (Parquet, ~3M rows) - for Data Preprocessing & ML
  3. Generate larger synthetic arrays - for SIMD & Streaming

This script prepares everything so benchmarks run on real, representative data.
"""
import os
import sys
import urllib.request
import tarfile
import shutil
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "images_input"
DATA_DIR = PROJECT_ROOT / "data_input"


def download_with_progress(url: str, dest: str, desc: str = ""):
    """Download a file with progress indicator."""
    print(f"\n📥 Downloading {desc or url}...")
    print(f"   URL: {url}")
    print(f"   Dest: {dest}")

    start = time.time()
    last_report = [0]

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb_done = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            if pct - last_report[0] >= 5:
                elapsed = time.time() - start
                speed = mb_done / elapsed if elapsed > 0 else 0
                print(f"   {pct:3d}% ({mb_done:.1f}/{mb_total:.1f} MB, {speed:.1f} MB/s)")
                last_report[0] = pct
        else:
            mb_done = downloaded / (1024 * 1024)
            if block_num % 500 == 0:
                print(f"   {mb_done:.1f} MB downloaded...")

    urllib.request.urlretrieve(url, dest, reporthook)
    elapsed = time.time() - start
    size_mb = os.path.getsize(dest) / (1024 * 1024)
    print(f"   ✅ Done! {size_mb:.1f} MB in {elapsed:.1f}s")


def download_caltech101():
    """Download Caltech-101 image dataset (~130MB compressed, ~9K images)."""
    dest_tar = str(PROJECT_ROOT / "caltech101.tar.gz")
    extract_dir = PROJECT_ROOT / "caltech101_raw"

    if IMAGES_DIR.exists() and len(list(IMAGES_DIR.glob("*.jpg"))) > 1000:
        print(f"\n✅ Images already present ({len(list(IMAGES_DIR.glob('*.jpg')))} files)")
        return

    # Caltech-101 from official source
    url = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip"
    dest_zip = str(PROJECT_ROOT / "caltech101.zip")

    if not os.path.exists(dest_zip):
        try:
            download_with_progress(url, dest_zip, "Caltech-101 images")
        except Exception as e:
            print(f"   ⚠️ Caltech-101 download failed: {e}")
            print("   Falling back to generating 1000 high-res synthetic images...")
            generate_large_synthetic_images()
            return

    # Extract
    print("\n📦 Extracting Caltech-101...")
    import zipfile
    try:
        with zipfile.ZipFile(dest_zip, 'r') as zf:
            zf.extractall(str(extract_dir))
    except Exception as e:
        print(f"   ⚠️ Extraction failed: {e}")
        print("   Falling back to synthetic images...")
        generate_large_synthetic_images()
        return

    # Move images to images_input (flatten directory structure)
    if IMAGES_DIR.exists():
        shutil.rmtree(IMAGES_DIR)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in extract_dir.rglob("*"):
        if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff'):
            dest = IMAGES_DIR / f"real_image_{count:05d}{img_path.suffix.lower()}"
            shutil.copy2(str(img_path), str(dest))
            count += 1
            if count % 1000 == 0:
                print(f"   Copied {count} images...")

    print(f"   ✅ Copied {count} real images to {IMAGES_DIR}")

    # Cleanup
    if extract_dir.exists():
        shutil.rmtree(extract_dir)


def generate_large_synthetic_images():
    """Fallback: generate 1000 high-resolution synthetic images."""
    from PIL import Image
    import numpy as np

    if IMAGES_DIR.exists():
        shutil.rmtree(IMAGES_DIR)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    num_images = 1000
    # Larger, more varied sizes
    sizes = [
        (1920, 1080),  # Full HD
        (1280, 960),
        (1024, 768),
        (2048, 1536),  # ~3MP
        (800, 600),
        (1600, 1200),
        (640, 480),
        (3840, 2160),  # 4K (a few)
    ]

    print(f"\n🖼️  Generating {num_images} high-res synthetic images...")

    for i in range(num_images):
        w, h = sizes[i % len(sizes)]
        # Create realistic-looking image with patterns
        img_array = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

        # Add structure
        x_grad = np.linspace(0, 255, w, dtype=np.uint8)
        y_grad = np.linspace(0, 255, h, dtype=np.uint8)
        img_array[:, :, 0] = (img_array[:, :, 0].astype(int) + x_grad[np.newaxis, :]) // 2
        img_array[:, :, 1] = (img_array[:, :, 1].astype(int) + y_grad[:, np.newaxis]) // 2

        # Add some circles/shapes for realism
        cy, cx = h // 2, w // 2
        y_coords, x_coords = np.ogrid[:h, :w]
        r = min(h, w) // 4
        mask = (x_coords - cx) ** 2 + (y_coords - cy) ** 2 < r ** 2
        img_array[mask, 2] = np.clip(img_array[mask, 2].astype(int) + 80, 0, 255).astype(np.uint8)

        img = Image.fromarray(img_array)
        img.save(IMAGES_DIR / f"synth_image_{i:05d}.jpg", quality=90)

        if (i + 1) % 100 == 0:
            print(f"   Generated {i + 1}/{num_images}")

    print(f"   ✅ Generated {num_images} images")


def download_nyc_taxi():
    """Download NYC Yellow Taxi trip data (1 month, ~3M rows)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    parquet_file = DATA_DIR / "nyc_taxi.parquet"

    if parquet_file.exists():
        size_mb = parquet_file.stat().st_size / (1024 * 1024)
        print(f"\n✅ NYC Taxi data already present ({size_mb:.1f} MB)")
        return

    # Try January 2024 (smaller) or January 2023
    urls = [
        ("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet",
         "NYC Yellow Taxi Jan 2024"),
        ("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet",
         "NYC Yellow Taxi Jan 2023"),
    ]

    for url, desc in urls:
        try:
            download_with_progress(url, str(parquet_file), desc)
            # Verify
            import pandas as pd
            df = pd.read_parquet(str(parquet_file))
            print(f"   📊 Loaded: {len(df):,} rows × {len(df.columns)} columns")
            print(f"   Columns: {list(df.columns[:8])}...")
            return
        except Exception as e:
            print(f"   ⚠️ Failed: {e}")
            if parquet_file.exists():
                parquet_file.unlink()
            continue

    # Fallback: generate large synthetic CSV
    print("\n   ⚠️ All NYC Taxi downloads failed, generating synthetic data...")
    generate_large_synthetic_tabular()


def generate_large_synthetic_tabular():
    """Generate large synthetic tabular dataset as fallback."""
    import numpy as np
    import pandas as pd

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_file = DATA_DIR / "synthetic_large.csv"

    if csv_file.exists():
        print(f"\n✅ Synthetic data already present")
        return

    print("\n📊 Generating 5M row synthetic dataset...")
    np.random.seed(42)
    n = 5_000_000

    df = pd.DataFrame({
        'trip_distance': np.random.exponential(3.0, n).astype(np.float32),
        'fare_amount': np.random.exponential(15.0, n).astype(np.float32),
        'tip_amount': np.random.exponential(2.5, n).astype(np.float32),
        'total_amount': np.random.exponential(20.0, n).astype(np.float32),
        'passenger_count': np.random.randint(1, 7, n).astype(np.int8),
        'payment_type': np.random.choice([1, 2, 3, 4], n).astype(np.int8),
        'pickup_hour': np.random.randint(0, 24, n).astype(np.int8),
        'pickup_day': np.random.randint(0, 7, n).astype(np.int8),
        'pickup_month': np.random.randint(1, 13, n).astype(np.int8),
        'feature_1': np.random.randn(n).astype(np.float32),
        'feature_2': np.random.randn(n).astype(np.float32),
        'feature_3': np.random.randn(n).astype(np.float32),
        'feature_4': np.random.randn(n).astype(np.float32),
        'feature_5': np.random.randn(n).astype(np.float32),
        'category_a': np.random.choice(['A', 'B', 'C', 'D', 'E'], n),
        'category_b': np.random.choice(['X', 'Y', 'Z'], n),
        'target': np.random.randint(0, 2, n).astype(np.int8),
    })

    df.to_csv(csv_file, index=False)
    size_mb = csv_file.stat().st_size / (1024 * 1024)
    print(f"   ✅ Generated {n:,} rows, {size_mb:.1f} MB")


def print_summary():
    """Print summary of all available data."""
    print(f"\n{'='*60}")
    print("DATA SUMMARY")
    print(f"{'='*60}")

    # Images
    if IMAGES_DIR.exists():
        images = list(IMAGES_DIR.glob("*.*"))
        total_size = sum(f.stat().st_size for f in images) / (1024 * 1024)
        print(f"\n📸 Images: {len(images)} files, {total_size:.0f} MB")
        print(f"   Path: {IMAGES_DIR}")
    else:
        print("\n📸 Images: NOT FOUND")

    # Tabular
    parquet = DATA_DIR / "nyc_taxi.parquet"
    csv_file = DATA_DIR / "synthetic_large.csv"
    if parquet.exists():
        size = parquet.stat().st_size / (1024 * 1024)
        print(f"\n📊 Tabular: NYC Taxi Parquet, {size:.0f} MB")
        print(f"   Path: {parquet}")
    elif csv_file.exists():
        size = csv_file.stat().st_size / (1024 * 1024)
        print(f"\n📊 Tabular: Synthetic CSV, {size:.0f} MB")
        print(f"   Path: {csv_file}")
    else:
        print("\n📊 Tabular: NOT FOUND")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    print("=" * 60)
    print("DOWNLOADING REAL-WORLD DATASETS FOR BENCHMARKS")
    print("=" * 60)

    # 1. Images
    print("\n" + "─" * 60)
    print("STEP 1: Image Dataset")
    print("─" * 60)
    download_caltech101()

    # 2. Tabular
    print("\n" + "─" * 60)
    print("STEP 2: Tabular Dataset")
    print("─" * 60)
    download_nyc_taxi()

    # Summary
    print_summary()

    print("\n🎯 Next: Update workloads and run benchmarks with real data!")
