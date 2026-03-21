import pandas as pd
import numpy as np
from threading import Thread
from multiprocessing import Pool
import time
import sys
import json
from pathlib import Path


# Numeric columns to normalize (set dynamically based on data source)
NUMERIC_COLS = []
CATEGORY_COLS = []


class DataPreprocessor:
    def __init__(self, data_source: str = "auto", num_rows: int = 3_000_000):
        """
        Load data from real NYC Taxi parquet or generate synthetic data.
        data_source: 'auto' (try real first), 'real', 'synthetic', or a file path
        """
        global NUMERIC_COLS, CATEGORY_COLS

        project_root = Path(__file__).resolve().parent.parent
        real_data = project_root / "data_input" / "nyc_taxi.parquet"

        if data_source == "auto" and real_data.exists():
            data_source = str(real_data)

        if data_source not in ("synthetic",) and Path(data_source).exists():
            print(f"  Loading real data: {data_source}")
            self.df = pd.read_parquet(str(data_source))
            # Select and prepare columns
            numeric_candidates = self.df.select_dtypes(include=[np.number]).columns.tolist()
            # Use key taxi columns
            keep_numeric = [c for c in ['trip_distance', 'fare_amount', 'tip_amount',
                                        'total_amount', 'passenger_count', 'tolls_amount',
                                        'extra', 'mta_tax', 'congestion_surcharge']
                           if c in self.df.columns]
            keep_cat = [c for c in ['payment_type', 'RatecodeID', 'VendorID',
                                    'store_and_fwd_flag'] if c in self.df.columns]
            # Add derived columns for extra work
            if 'tpep_pickup_datetime' in self.df.columns:
                self.df['pickup_hour'] = self.df['tpep_pickup_datetime'].dt.hour
                self.df['pickup_dow'] = self.df['tpep_pickup_datetime'].dt.dayofweek
                keep_numeric += ['pickup_hour', 'pickup_dow']
            self.df = self.df[keep_numeric + keep_cat].copy()
            # Fill NaN
            for c in keep_numeric:
                self.df[c] = self.df[c].fillna(0).astype(np.float64)
            for c in keep_cat:
                self.df[c] = self.df[c].astype(str)
            NUMERIC_COLS = keep_numeric
            CATEGORY_COLS = keep_cat
            print(f"  Loaded {len(self.df):,} rows × {len(self.df.columns)} columns "
                  f"({len(keep_numeric)} numeric, {len(keep_cat)} categorical)")
        else:
            print(f"  Generating synthetic data: {num_rows:,} rows")
            np.random.seed(42)
            self.df = pd.DataFrame({
                'feature1': np.random.randn(num_rows),
                'feature2': np.random.randn(num_rows),
                'feature3': np.random.randn(num_rows),
                'feature4': np.random.randn(num_rows),
                'feature5': np.random.randn(num_rows),
                'feature6': np.random.randn(num_rows),
                'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], num_rows),
                'subcategory': np.random.choice(['X', 'Y', 'Z'], num_rows),
                'target': np.random.randint(0, 2, num_rows)
            })
            NUMERIC_COLS = ['feature1', 'feature2', 'feature3', 'feature4',
                           'feature5', 'feature6']
            CATEGORY_COLS = ['category', 'subcategory']

    def preprocess_chunk(self, chunk_df: pd.DataFrame) -> pd.DataFrame:
        """Process a single chunk — heavy processing"""
        chunk_df = chunk_df.copy()

        # Normalize all numeric columns
        for col in NUMERIC_COLS:
            if col in chunk_df.columns:
                chunk_df[col] = (chunk_df[col] - chunk_df[col].mean()) / (chunk_df[col].std() + 1e-8)

        # One-hot encoding
        cat_cols_present = [c for c in CATEGORY_COLS if c in chunk_df.columns]
        if cat_cols_present:
            chunk_df = pd.get_dummies(chunk_df, columns=cat_cols_present)

        # Feature engineering — more operations for heavier workload
        ncols = [c for c in NUMERIC_COLS if c in chunk_df.columns]
        if len(ncols) >= 2:
            chunk_df['feat_sum_01'] = chunk_df[ncols[0]] + chunk_df[ncols[1]]
            chunk_df['feat_prod_01'] = chunk_df[ncols[0]] * chunk_df[ncols[1]]
            chunk_df['feat_ratio_01'] = chunk_df[ncols[0]] / (chunk_df[ncols[1]] + 1e-8)
        if len(ncols) >= 4:
            chunk_df['feat_diff_23'] = chunk_df[ncols[2]] - chunk_df[ncols[3]]
            chunk_df['feat_hypot'] = np.sqrt(chunk_df[ncols[0]]**2 + chunk_df[ncols[1]]**2)
            chunk_df['feat_exp'] = np.exp(chunk_df[ncols[2]].clip(-5, 5))
            chunk_df['feat_log'] = np.log1p(np.abs(chunk_df[ncols[3]]))
            chunk_df['feat_sin'] = np.sin(chunk_df[ncols[0]] * np.pi)
            chunk_df['feat_poly'] = (chunk_df[ncols[0]]**2 +
                                     chunk_df[ncols[1]]**3 +
                                     chunk_df[ncols[2]]**2)
        if len(ncols) >= 6:
            chunk_df['feat_mean_all'] = chunk_df[ncols].mean(axis=1)
            chunk_df['feat_std_all'] = chunk_df[ncols].std(axis=1)
            chunk_df['feat_max_all'] = chunk_df[ncols].max(axis=1)
            chunk_df['feat_min_all'] = chunk_df[ncols].min(axis=1)

        return chunk_df

    def process_sequential(self):
        """Sequential processing"""
        return self.preprocess_chunk(self.df)

    def process_threading(self, num_threads: int):
        """Processing with threads"""
        chunk_size = len(self.df) // num_threads
        chunks = [self.df.iloc[i:i+chunk_size] for i in range(0, len(self.df), chunk_size)]

        results = [None] * len(chunks)

        def worker(idx, chunk):
            results[idx] = self.preprocess_chunk(chunk)

        threads = []
        for idx, chunk in enumerate(chunks):
            t = Thread(target=worker, args=(idx, chunk))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        return pd.concat(results, ignore_index=True)

    def process_multiprocessing(self, num_processes: int):
        """Processing with multiprocessing"""
        chunk_size = len(self.df) // num_processes
        chunks = [self.df.iloc[i:i+chunk_size] for i in range(0, len(self.df), chunk_size)]

        with Pool(num_processes) as pool:
            results = pool.map(preprocess_chunk_static, chunks)

        return pd.concat(results, ignore_index=True)


def preprocess_chunk_static(chunk_df: pd.DataFrame) -> pd.DataFrame:
    """Static function for multiprocessing — uses same logic as class method."""
    chunk_df = chunk_df.copy()

    # Normalization
    for col in NUMERIC_COLS:
        if col in chunk_df.columns:
            chunk_df[col] = (chunk_df[col] - chunk_df[col].mean()) / (chunk_df[col].std() + 1e-8)

    # One-hot encoding
    cat_cols_present = [c for c in CATEGORY_COLS if c in chunk_df.columns]
    if cat_cols_present:
        chunk_df = pd.get_dummies(chunk_df, columns=cat_cols_present)

    # Feature engineering
    ncols = [c for c in NUMERIC_COLS if c in chunk_df.columns]
    if len(ncols) >= 2:
        chunk_df['feat_sum_01'] = chunk_df[ncols[0]] + chunk_df[ncols[1]]
        chunk_df['feat_prod_01'] = chunk_df[ncols[0]] * chunk_df[ncols[1]]
        chunk_df['feat_ratio_01'] = chunk_df[ncols[0]] / (chunk_df[ncols[1]] + 1e-8)
    if len(ncols) >= 4:
        chunk_df['feat_diff_23'] = chunk_df[ncols[2]] - chunk_df[ncols[3]]
        chunk_df['feat_hypot'] = np.sqrt(chunk_df[ncols[0]]**2 + chunk_df[ncols[1]]**2)
        chunk_df['feat_exp'] = np.exp(chunk_df[ncols[2]].clip(-5, 5))
        chunk_df['feat_log'] = np.log1p(np.abs(chunk_df[ncols[3]]))
        chunk_df['feat_sin'] = np.sin(chunk_df[ncols[0]] * np.pi)
        chunk_df['feat_poly'] = (chunk_df[ncols[0]]**2 +
                                 chunk_df[ncols[1]]**3 +
                                 chunk_df[ncols[2]]**2)
    if len(ncols) >= 6:
        chunk_df['feat_mean_all'] = chunk_df[ncols].mean(axis=1)
        chunk_df['feat_std_all'] = chunk_df[ncols].std(axis=1)

    return chunk_df


def run_benchmark(data_source: str = "auto", num_rows: int = 3_000_000, num_runs: int = 3):
    """Runs benchmark for all modes"""
    results = {}

    preprocessor = DataPreprocessor(data_source=data_source, num_rows=num_rows)

    modes = [
        ("sequential", lambda: preprocessor.process_sequential()),
        ("threading_2", lambda: preprocessor.process_threading(2)),
        ("threading_4", lambda: preprocessor.process_threading(4)),
        ("threading_8", lambda: preprocessor.process_threading(8)),
        ("multiprocessing_2", lambda: preprocessor.process_multiprocessing(2)),
        ("multiprocessing_4", lambda: preprocessor.process_multiprocessing(4)),
        ("multiprocessing_8", lambda: preprocessor.process_multiprocessing(8)),
    ]

    for mode_name, mode_func in modes:
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = mode_func()
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
    # Accept 'real', 'synthetic', or a number of rows
    data_source = "auto"
    num_rows = 3_000_000

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg in ("real", "auto", "synthetic"):
            data_source = arg
        else:
            try:
                num_rows = int(arg)
            except ValueError:
                data_source = arg  # treat as file path

    print("=" * 60)
    print(f"Data Preprocessing Benchmark")
    print(f"  Source: {data_source}, Fallback rows: {num_rows:,}")
    print("=" * 60)

    results = run_benchmark(data_source=data_source, num_rows=num_rows)

    # Save results
    with open("results/data_preprocessing_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to results/data_preprocessing_results.json")

