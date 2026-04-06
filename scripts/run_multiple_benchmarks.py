"""
Multiple-run benchmark suite for statistical validity.
Runs each workload N times under both GIL and no-GIL Python,
then aggregates all individual measurements into a single master file.

Each workload internally does 3 runs per mode → with 5 external runs
we get 15 measurements per mode per variant.

Usage:
    python scripts/run_multiple_benchmarks.py [num_iterations]
    Default: 5 iterations
"""
import subprocess
import json
import os
import sys
import time
import copy
from pathlib import Path
from datetime import datetime
import numpy as np
import psutil
import threading


# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_GIL_PYTHON = str(PROJECT_ROOT / "venv_gil" / "bin" / "python")
VENV_NOGIL_PYTHON = str(PROJECT_ROOT / "venv_nogil" / "bin" / "python")
VENV_NOGIL_314_PYTHON = str(PROJECT_ROOT / "venv_nogil_314" / "bin" / "python")
RESULTS_DIR = PROJECT_ROOT / "results"
CHARTS_DIR = RESULTS_DIR / "charts"


def check_python(python_path: str) -> dict:
    """Check Python version and GIL status."""
    try:
        result = subprocess.run(
            [python_path, "-c",
             "import sys; print(sys.version); print(sys._is_gil_enabled())"],
            capture_output=True, text=True, timeout=60
        )
        lines = result.stdout.strip().split('\n')
        return {
            "path": python_path,
            "version": lines[0] if lines else "unknown",
            "gil_enabled": lines[1].strip() == "True" if len(lines) > 1 else None,
            "ok": result.returncode == 0
        }
    except Exception as e:
        return {"path": python_path, "error": str(e), "ok": False}


def _monitor_process(pid: int, stop_event: threading.Event,
                     cpu_samples: list, mem_samples: list,
                     interval: float = 0.5):
    """Background thread: sample CPU% and RSS of *pid* until stop_event is set."""
    try:
        proc = psutil.Process(pid)
        # Initial call to seed cpu_percent (returns 0.0 the first time)
        proc.cpu_percent(interval=None)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return

    while not stop_event.is_set():
        try:
            cpu = proc.cpu_percent(interval=None)
            mem_mb = proc.memory_info().rss / (1024 * 1024)
            # Also try to include children (worker processes)
            for child in proc.children(recursive=True):
                try:
                    cpu += child.cpu_percent(interval=None)
                    mem_mb += child.memory_info().rss / (1024 * 1024)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            cpu_samples.append(cpu)
            mem_samples.append(mem_mb)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
        stop_event.wait(interval)


def run_single_workload(python_path: str, script: str, args: list,
                        timeout: int = 600) -> dict:
    """Run a single benchmark script, return wall time + CPU/memory metrics."""
    cmd = [python_path, script] + [str(a) for a in args]
    cpu_samples = []
    mem_samples = []
    start = time.perf_counter()
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd=str(PROJECT_ROOT)
        )
        stop_evt = threading.Event()
        monitor = threading.Thread(
            target=_monitor_process,
            args=(proc.pid, stop_evt, cpu_samples, mem_samples, 0.5),
            daemon=True,
        )
        monitor.start()

        stdout, stderr = proc.communicate(timeout=timeout)
        elapsed = time.perf_counter() - start
        stop_evt.set()
        monitor.join(timeout=2)

        # Compute resource metrics
        resource_metrics = {}
        if cpu_samples:
            resource_metrics["cpu_mean_percent"] = round(float(np.mean(cpu_samples)), 2)
            resource_metrics["cpu_peak_percent"] = round(float(np.max(cpu_samples)), 2)
            resource_metrics["cpu_samples"] = len(cpu_samples)
        if mem_samples:
            resource_metrics["mem_peak_mb"] = round(float(np.max(mem_samples)), 2)
            resource_metrics["mem_mean_mb"] = round(float(np.mean(mem_samples)), 2)

        return {
            "success": proc.returncode == 0,
            "wall_time": round(elapsed, 4),
            "stdout": stdout[-500:] if stdout else "",
            "stderr": stderr[-300:] if stderr else "",
            "resources": resource_metrics,
        }
    except subprocess.TimeoutExpired:
        proc.kill()
        return {"success": False, "error": "TIMEOUT", "wall_time": timeout,
                "resources": {}}
    except Exception as e:
        return {"success": False, "error": str(e), "wall_time": 0,
                "resources": {}}


def read_result_file(result_path: Path) -> dict:
    """Read JSON result file if it exists."""
    try:
        if result_path.exists():
            with open(result_path) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def merge_detailed_results(all_runs: list) -> dict:
    """
    Merge detailed results from multiple runs.
    Each run has keys like 'sequential' -> {'avg_time': ..., 'all_times': [...]}.
    We combine all 'all_times' lists and recompute statistics.
    """
    if not all_runs:
        return {}

    merged = {}

    # Get all mode keys from the first run
    all_keys = set()
    for run in all_runs:
        all_keys.update(run.keys())

    for key in all_keys:
        all_times_combined = []
        all_values = {}

        for run in all_runs:
            if key not in run:
                continue

            val = run[key]

            if isinstance(val, dict):
                if "all_times" in val:
                    # Mode with timing data (e.g. threading_4)
                    all_times_combined.extend(val["all_times"])
                elif "avg_latency" in val:
                    # Streaming results — accumulate all metric lists
                    for metric_name, metric_val in val.items():
                        if isinstance(metric_val, (int, float)):
                            if metric_name not in all_values:
                                all_values[metric_name] = []
                            all_values[metric_name].append(float(metric_val))
                        elif isinstance(metric_val, dict) and "mean" in metric_val:
                            if metric_name not in all_values:
                                all_values[metric_name] = []
                            all_values[metric_name].append(float(metric_val["mean"]))
                else:
                    # Nested dict (e.g. SIMD sequential/threaded/matrix)
                    if key not in merged:
                        merged[key] = {}
                    for subkey, subval in val.items():
                        if isinstance(subval, (int, float)):
                            if subkey not in merged[key]:
                                merged[key][subkey] = {"all_values": []}
                            merged[key][subkey]["all_values"].append(float(subval))
            elif isinstance(val, (int, float)):
                all_times_combined.append(float(val))

        if all_times_combined:
            arr = np.array(all_times_combined)
            min_t = float(np.min(arr))
            max_t = float(np.max(arr))
            tail = max_t - min_t
            merged[key] = {
                "min_time": min_t,
                "tail_length": tail,
                
                "avg_time": float(np.mean(arr)),
                "std_time": float(np.std(arr, ddof=1)),
                "max_time": max_t,
                "median_time": float(np.median(arr)),
                "n_runs": len(arr),
                "all_times": [float(x) for x in arr],
                # 95% confidence interval
                "ci_95": [
                    float(np.mean(arr) - 1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr))),
                    float(np.mean(arr) + 1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr))),
                ],
            }
        elif all_values:
            # Streaming-type results
            merged[key] = {}
            for metric_name, vals in all_values.items():
                arr = np.array(vals)
                merged[key][metric_name] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr, ddof=1)),
                    "n_runs": len(arr),
                    "all_values": [float(x) for x in arr],
                }
        elif key in merged and isinstance(merged[key], dict):
            # Finalize nested dicts (SIMD)
            for subkey in list(merged[key].keys()):
                if isinstance(merged[key][subkey], dict) and "all_values" in merged[key][subkey]:
                    vals = merged[key][subkey]["all_values"]
                    arr = np.array(vals)
                    merged[key][subkey] = {
                        "mean": float(np.mean(arr)),
                        "std": float(np.std(arr, ddof=1)),
                        "n_runs": len(arr),
                        "all_values": [float(x) for x in arr],
                    }

    return merged


# -- Workload definitions (LARGE DATA) --
WORKLOADS = [
    {
        "name": "Data Preprocessing",
        "script": "workloads/workload_data_preprocessing.py",
        "args": ["10000000"],
        "params": "10M rows",
        "result_file": "data_preprocessing_results.json",
        "timeout": 3600,
    },
    {
        "name": "Image Processing",
        "script": "workloads/workload_image_processing.py",
        "args": ["images_input", "images_output"],
        "params": "10K images",
        "result_file": "image_processing_results.json",
        "timeout": 7200,
    },
    {
        "name": "ML Training",
        "script": "workloads/workload_ml.py",
        "args": ["500000"],
        "params": "500K samples",
        "result_file": "ml_results.json",
        "timeout": 7200,
    },
    {
        "name": "Streaming",
        "script": "workloads/workload_streaming.py",
        "args": ["1000000"],
        "params": "1M events",
        "result_file": "streaming_results.json",
        "timeout": 1800,
    },
    {
        "name": "SIMD Vectorization",
        "script": "workloads/workload_simd.py",
        "args": ["100000000"],
        "params": "100M elements",
        "result_file": "simd_results.json",
        "timeout": 1200,
    },
]



def run_suite(version_str, gil_path, nogil_path, num_iterations):
    print("=" * 70)
    print(f"FREE-THREADED PYTHON {version_str} — MULTI-RUN BENCHMARK SUITE")
    print(f"Iterations: {num_iterations}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    out_dir = RESULTS_DIR / version_str
    out_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Checking Python interpreters ---")
    gil_info = check_python(gil_path)
    nogil_info = check_python(nogil_path)
    print(f"  GIL:    {gil_info.get('version','?')[:50]}  GIL={gil_info.get('gil_enabled')}")
    print(f"  No-GIL: {nogil_info.get('version','?')[:50]}  GIL={nogil_info.get('gil_enabled')}")

    if not gil_info["ok"] or not nogil_info["ok"]:
        print(f"❌ ERROR: Python interpreter problem for {version_str}!")
        sys.exit(1)

    master = {
        "metadata": {
            "version_group": version_str,
            "timestamp": datetime.now().isoformat(),
            "num_iterations": num_iterations,
            "gil_python": gil_info,
            "nogil_python": nogil_info,
        },
        "workloads": {},
    }

    total_workloads = len(WORKLOADS)
    total_runs = total_workloads * 2 * num_iterations  # 2 variants (GIL, No-GIL)
    current_run = 0

    for wl_def in WORKLOADS:
        wl_name = wl_def["name"]
        result_file = RESULTS_DIR / wl_def["result_file"]

        print(f"\n{'='*70}")
        print(f"WORKLOAD: {wl_name} ({wl_def['params']})")
        print(f"{'='*70}")

        wl_result = {
            "name": wl_name,
            "params": wl_def["params"],
            "script": wl_def["script"],
            "gil": {"wall_times": [], "run_details": [], "resource_samples": []},
            "nogil": {"wall_times": [], "run_details": [], "resource_samples": []},
        }

        for variant, python_path, label in [
            ("gil", gil_path, "GIL"),
            ("nogil", nogil_path, "No-GIL"),
        ]:
            print(f"\n  >>> {label} — {num_iterations} iterations...")
            for i in range(num_iterations):
                current_run += 1
                progress = f"[{current_run}/{total_runs}]"

                if result_file.exists():
                    result_file.unlink()

                print(f"    {progress} Run {i+1}/{num_iterations}...", end=" ", flush=True)

                # Force each script to do 1 run to have independent iterations
                # Most scripts take num_runs as the last argument if provided
                run_args = wl_def["args"] + ["1"] 
                run_out = run_single_workload(python_path, wl_def["script"], run_args, timeout=wl_def["timeout"])

                wl_result[variant]["wall_times"].append(run_out["wall_time"])
                wl_result[variant]["resource_samples"].append(run_out.get("resources", {}))

                if run_out["success"]:
                    parsed = read_result_file(result_file)
                    if parsed:
                        wl_result[variant]["run_details"].append(parsed)
                    res = run_out.get("resources", {})
                    cpu_i = f"CPU={res.get('cpu_mean_percent','?')}%" if res else ""
                    mem_i = f"MEM={res.get('mem_peak_mb','?')}MB" if res else ""
                    print(f"✅ {run_out['wall_time']:.2f}s {cpu_i} {mem_i}")
                else:
                    err = run_out.get('error', run_out.get('stderr', ''))[:100]
                    print(f"❌ {err}")

            # Summary
            wall_arr = np.array(wl_result[variant]["wall_times"])
            print(f"  {label} summary: min={np.min(wall_arr):.4f}s, mean={np.mean(wall_arr):.4f}s, max={np.max(wall_arr):.4f}s, median={np.median(wall_arr):.4f}s")

        for variant in ["gil", "nogil"]:
            runs = wl_result[variant]["run_details"]
            wl_result[variant]["merged"] = merge_detailed_results(runs)
            
            wt = np.array(wl_result[variant]["wall_times"])
            wl_result[variant]["wall_stats"] = {
                "min": float(np.min(wt)),
                "mean": float(np.mean(wt)),
                "max": float(np.max(wt)),
                "median": float(np.median(wt)),
                "std": float(np.std(wt, ddof=1)),
                "n": len(wt),
            }

        master["workloads"][wl_name] = wl_result

    master_file = out_dir / "multi_run_master.json"
    with open(master_file, "w") as f:
        json.dump(master, f, indent=2)

    # Copy individual workload results to the version-specific directory
    for wl_def in WORKLOADS:
        src = RESULTS_DIR / wl_def["result_file"]
        if src.exists():
            import shutil
            shutil.copy(src, out_dir / wl_def["result_file"])

    print(f"\n\n{'='*70}")
    print(f"{version_str} BENCHMARK COMPLETE")
    print(f"Results: {master_file} and individual JSONs in {out_dir}")
    print(f"{'='*70}")


def main():
    num_iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    version_filter = sys.argv[2] if len(sys.argv) > 2 else None

    # Run 3.13 comparison
    if not version_filter or version_filter == "3.13":
        run_suite("3.13", VENV_GIL_PYTHON, VENV_NOGIL_PYTHON, num_iterations)
    
    # Run 3.14 comparison
    if not version_filter or version_filter == "3.14":
        VENV_GIL_314_PYTHON = str(PROJECT_ROOT / "venv_gil_314" / "bin" / "python")
        run_suite("3.14", VENV_GIL_314_PYTHON, VENV_NOGIL_314_PYTHON, num_iterations)

if __name__ == "__main__":
    main()
