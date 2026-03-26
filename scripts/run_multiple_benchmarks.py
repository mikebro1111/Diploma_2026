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
            merged[key] = {
                "avg_time": float(np.mean(arr)),
                "std_time": float(np.std(arr, ddof=1)),
                "min_time": float(np.min(arr)),
                "max_time": float(np.max(arr)),
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
        "args": ["auto"],  # Uses real NYC Taxi data (~3M rows) or 3M synthetic
        "result_file": "data_preprocessing_results.json",
        "timeout": 1200,
    },
    {
        "name": "Image Processing",
        "script": "workloads/workload_image_processing.py",
        "args": ["images_input", "images_output"],  # 9K+ real Caltech-101 images
        "result_file": "image_processing_results.json",
        "timeout": 1200,
    },
    {
        "name": "ML Training",
        "script": "workloads/workload_ml.py",
        "args": ["200000"],  # 200K samples (was 50K)
        "result_file": "ml_results.json",
        "timeout": 1800,
    },
    {
        "name": "Streaming",
        "script": "workloads/workload_streaming.py",
        "args": ["50000", "100000"],  # 50K events at 100K/s (was 5K at 10K/s)
        "result_file": "streaming_results.json",
        "timeout": 600,
    },
    {
        "name": "SIMD Vectorization",
        "script": "workloads/workload_simd.py",
        "args": ["50000000"],  # 50M elements (was 5M)
        "result_file": "simd_results.json",
        "timeout": 1200,
    },
]


def main():
    num_iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    print("=" * 70)
    print("FREE-THREADED PYTHON — MULTI-RUN BENCHMARK SUITE")
    print(f"Iterations: {num_iterations} (each workload does 3 internal runs)")
    print(f"Total measurements per mode: up to {num_iterations * 3}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    # Check interpreters
    print("\n--- Checking Python interpreters ---")
    gil_info = check_python(VENV_GIL_PYTHON)
    nogil_info = check_python(VENV_NOGIL_PYTHON)
    print(f"  GIL:    {gil_info.get('version','?')[:50]}  GIL={gil_info.get('gil_enabled')}")
    print(f"  No-GIL: {nogil_info.get('version','?')[:50]}  GIL={nogil_info.get('gil_enabled')}")

    if not gil_info["ok"] or not nogil_info["ok"]:
        print("❌ ERROR: Python interpreter problem!")
        sys.exit(1)

    # Prepare master result structure
    master = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_iterations": num_iterations,
            "measurements_per_mode": f"up to {num_iterations * 3}",
            "gil_python": gil_info,
            "nogil_python": nogil_info,
        },
        "workloads": {},
    }

    total_workloads = len(WORKLOADS)
    total_runs = total_workloads * 2 * num_iterations  # 2 variants (gil/nogil)
    current_run = 0

    for wl_def in WORKLOADS:
        wl_name = wl_def["name"]
        result_file = RESULTS_DIR / wl_def["result_file"]

        print(f"\n{'='*70}")
        print(f"WORKLOAD: {wl_name}")
        print(f"{'='*70}")

        wl_result = {
            "name": wl_name,
            "script": wl_def["script"],
            "gil": {"wall_times": [], "run_details": [], "resource_samples": []},
            "nogil": {"wall_times": [], "run_details": [], "resource_samples": []},
        }

        for variant, python_path, label in [
            ("gil", VENV_GIL_PYTHON, "GIL"),
            ("nogil", VENV_NOGIL_PYTHON, "No-GIL"),
        ]:
            print(f"\n  >>> {label} — {num_iterations} iterations...")

            for i in range(num_iterations):
                current_run += 1
                progress = f"[{current_run}/{total_runs}]"

                # Remove old result file
                if result_file.exists():
                    result_file.unlink()

                print(f"    {progress} Run {i+1}/{num_iterations}...", end=" ", flush=True)

                run_out = run_single_workload(
                    python_path, wl_def["script"], wl_def["args"],
                    timeout=wl_def["timeout"]
                )

                wl_result[variant]["wall_times"].append(run_out["wall_time"])
                wl_result[variant]["resource_samples"].append(
                    run_out.get("resources", {})
                )

                if run_out["success"]:
                    parsed = read_result_file(result_file)
                    if parsed:
                        wl_result[variant]["run_details"].append(parsed)
                    res = run_out.get("resources", {})
                    cpu_info = f"CPU={res.get('cpu_mean_percent','?')}%" if res else ""
                    mem_info = f"MEM={res.get('mem_peak_mb','?')}MB" if res else ""
                    print(f"✅ {run_out['wall_time']:.2f}s {cpu_info} {mem_info}")
                else:
                    err = run_out.get('error', run_out.get('stderr', ''))[:100]
                    print(f"❌ {err}")

            # Summary for this variant
            wall_arr = np.array(wl_result[variant]["wall_times"])
            print(f"  {label} summary: mean={np.mean(wall_arr):.2f}s, "
                  f"std={np.std(wall_arr, ddof=1):.2f}s, "
                  f"min={np.min(wall_arr):.2f}s, max={np.max(wall_arr):.2f}s")

        # Merge detailed results from all runs
        for variant in ["gil", "nogil"]:
            runs = wl_result[variant]["run_details"]
            wl_result[variant]["merged"] = merge_detailed_results(runs)

            # Wall time stats
            wt = np.array(wl_result[variant]["wall_times"])
            wl_result[variant]["wall_stats"] = {
                "mean": float(np.mean(wt)),
                "std": float(np.std(wt, ddof=1)),
                "min": float(np.min(wt)),
                "max": float(np.max(wt)),
                "median": float(np.median(wt)),
                "n": len(wt),
                "ci_95": [
                    float(np.mean(wt) - 1.96 * np.std(wt, ddof=1) / np.sqrt(len(wt))),
                    float(np.mean(wt) + 1.96 * np.std(wt, ddof=1) / np.sqrt(len(wt))),
                ],
            }

            # Aggregate resource metrics across runs
            res_list = wl_result[variant]["resource_samples"]
            cpu_means = [r["cpu_mean_percent"] for r in res_list if "cpu_mean_percent" in r]
            cpu_peaks = [r["cpu_peak_percent"] for r in res_list if "cpu_peak_percent" in r]
            mem_peaks = [r["mem_peak_mb"] for r in res_list if "mem_peak_mb" in r]
            mem_means = [r["mem_mean_mb"] for r in res_list if "mem_mean_mb" in r]

            wl_result[variant]["resource_stats"] = {
                "cpu_mean_percent": {
                    "mean": round(float(np.mean(cpu_means)), 2) if cpu_means else None,
                    "std": round(float(np.std(cpu_means, ddof=1)), 2) if len(cpu_means) > 1 else 0,
                    "all_values": cpu_means,
                },
                "cpu_peak_percent": {
                    "mean": round(float(np.mean(cpu_peaks)), 2) if cpu_peaks else None,
                    "max": round(float(np.max(cpu_peaks)), 2) if cpu_peaks else None,
                    "all_values": cpu_peaks,
                },
                "mem_peak_mb": {
                    "mean": round(float(np.mean(mem_peaks)), 2) if mem_peaks else None,
                    "max": round(float(np.max(mem_peaks)), 2) if mem_peaks else None,
                    "all_values": mem_peaks,
                },
                "mem_mean_mb": {
                    "mean": round(float(np.mean(mem_means)), 2) if mem_means else None,
                    "all_values": mem_means,
                },
            }

            # Drop run_details and raw resource_samples (too large for master)
            del wl_result[variant]["run_details"]
            del wl_result[variant]["resource_samples"]

        # Compute speedup
        g_mean = wl_result["gil"]["wall_stats"]["mean"]
        n_mean = wl_result["nogil"]["wall_stats"]["mean"]
        wl_result["wall_speedup"] = round(g_mean / n_mean, 4) if n_mean > 0 else 0

        master["workloads"][wl_name] = wl_result

    # ===== Save master =====
    master_file = RESULTS_DIR / "multi_run_master.json"
    with open(master_file, "w") as f:
        json.dump(master, f, indent=2, default=str)

    # ===== Print final summary =====
    print(f"\n\n{'='*70}")
    print("MULTI-RUN BENCHMARK COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results: {master_file}")
    print(f"{'='*70}")

    print(f"\n{'Workload':<22} {'GIL mean':>9} {'GIL std':>8} "
          f"{'NoGIL mean':>10} {'NoGIL std':>9} {'Speedup':>8} "
          f"{'GIL CPU%':>9} {'NoGIL CPU%':>10} {'GIL Mem':>8} {'NoGIL Mem':>9}")
    print("-" * 110)
    for name, wl in master["workloads"].items():
        gs = wl["gil"]["wall_stats"]
        ns = wl["nogil"]["wall_stats"]
        gr = wl["gil"].get("resource_stats", {})
        nr = wl["nogil"].get("resource_stats", {})
        sp = wl.get("wall_speedup", 0)
        g_cpu = gr.get("cpu_mean_percent", {}).get("mean", "?")
        n_cpu = nr.get("cpu_mean_percent", {}).get("mean", "?")
        g_mem = gr.get("mem_peak_mb", {}).get("mean", "?")
        n_mem = nr.get("mem_peak_mb", {}).get("mean", "?")
        g_cpu_s = f"{g_cpu}%" if g_cpu != "?" else "?"
        n_cpu_s = f"{n_cpu}%" if n_cpu != "?" else "?"
        g_mem_s = f"{g_mem}MB" if g_mem != "?" else "?"
        n_mem_s = f"{n_mem}MB" if n_mem != "?" else "?"
        print(f"{name:<22} {gs['mean']:>8.2f}s {gs['std']:>7.2f}s "
              f"{ns['mean']:>9.2f}s {ns['std']:>8.2f}s {sp:>7.2f}x "
              f"{g_cpu_s:>9} {n_cpu_s:>10} {g_mem_s:>8} {n_mem_s:>9}")

    # ===== Per-mode detailed comparison =====
    print(f"\n{'='*70}")
    print("DETAILED MODE-LEVEL COMPARISON (merged across all runs)")
    print(f"{'='*70}")
    for name, wl in master["workloads"].items():
        g_merged = wl["gil"].get("merged", {})
        n_merged = wl["nogil"].get("merged", {})
        if not g_merged or not n_merged:
            continue

        print(f"\n--- {name} ---")
        print(f"  {'Mode':<30} {'GIL':>12} {'NoGIL':>12} {'Speedup':>9} {'N':>4} {'p-value':>9}")
        print(f"  {'-'*76}")

        for key in g_merged:
            if key not in n_merged:
                continue

            g_data = g_merged[key]
            n_data = n_merged[key]

            if isinstance(g_data, dict) and "all_times" in g_data and \
               isinstance(n_data, dict) and "all_times" in n_data:
                from scipy import stats

                g_times = g_data["all_times"]
                n_times = n_data["all_times"]
                g_mean = g_data["avg_time"]
                n_mean = n_data["avg_time"]
                speedup = g_mean / n_mean if n_mean > 0 else 0
                n_samples = min(len(g_times), len(n_times))

                # Welch's t-test (equal_var=False — does not assume equal variances)
                t_stat, p_val = stats.ttest_ind(g_times, n_times, equal_var=False)

                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                faster = "NoGIL" if speedup > 1 else "GIL"

                print(f"  {key:<30} {g_mean:>11.4f}s {n_mean:>11.4f}s "
                      f"{speedup:>8.2f}x {n_samples:>4} {p_val:>8.4f} {sig}")

    print(f"\nSignificance: *** p<0.001, ** p<0.01, * p<0.05")


if __name__ == "__main__":
    main()
