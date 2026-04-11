"""
Comprehensive visualization and statistical analysis for multi-run benchmark results.
Reads from multi_run_master.json and generates publication-quality charts.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import sys

TARGET_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/3.13")
CHARTS_DIR = TARGET_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

def load_data() -> dict:
    f = TARGET_DIR / "multi_run_master.json"
    if not f.exists():
        print(f"❌ Not found: {f}")
        sys.exit(1)
    with open(f) as fh:
        data = json.load(fh)
    # Define globally for plot titles
    global n_runs
    meta = data.get("metadata", {})
    n_runs = meta.get("num_iterations")
    if n_runs is None:
        # Fallback to checking n_runs in the first workload's merged data
        wls = data.get("workloads", {})
        if wls:
            first_wl = list(wls.values())[0]
            for variant in ["gil", "nogil"]:
                merged = first_wl.get(variant, {}).get("merged", {})
                if merged:
                    first_mode = list(merged.values())[0]
                    # Could be simple or nested
                    if "n_runs" in first_mode:
                        n_runs = first_mode["n_runs"]
                    elif isinstance(first_mode, dict):
                        for subval in first_mode.values():
                            if isinstance(subval, dict) and "n_runs" in subval:
                                n_runs = subval["n_runs"]
                                break
                    if n_runs: break
    if n_runs is None: n_runs = 20
    return data


def setup_style():
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'figure.figsize': (12, 6),
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


GIL_COLOR = '#4A90D9'
NOGIL_COLOR = '#E8505B'
NOGIL_314_COLOR = '#2ECC71'
IDEAL_COLOR = '#888888'


def _get_mode_data(wl, variant, mode):
    """Extract data for a specific mode from merged results."""
    merged = wl.get(variant, {}).get("merged", {})
    return merged.get(mode, {})


def _get_times(wl, variant, mode):
    """Get all_times list for a specific mode."""
    d = _get_mode_data(wl, variant, mode)
    return d.get("all_times", [])


def _get_avg(wl, variant, mode):
    """Get min_time for a mode (or mean for nested)."""
    d = _get_mode_data(wl, variant, mode)
    if "min_time" in d:
        return d["min_time"]
    if "avg_latency" in d: # Nested
        lat = d["avg_latency"]
        return lat.get("min_time") or lat.get("avg_time", 0)
    if isinstance(d, dict):
        # SIMD or other nested
        for k, v in d.items():
            if isinstance(v, dict) and "min_time" in v:
                return v["min_time"]
    return 0.0

def _get_std(wl, variant, mode):
    """Get statistical error (95% CI width preferred, fallback to SD)."""
    d = _get_mode_data(wl, variant, mode)
    
    # Check for direct metrics (top-level or nested)
    def extract_err(metric_dict):
        # Prefer 95% Confidence Interval for "Adequate" representation
        if "ci_95" in metric_dict:
            ci = metric_dict["ci_95"]
            if len(ci) == 2:
                # Width from mean to 95% boundary
                mean_v = metric_dict.get("avg_time") or metric_dict.get("avg_val") or metric_dict.get("mean", 0)
                return max(0, ci[1] - mean_v)
        
        # Fallback to Standard Deviation
        if "std_time" in metric_dict: return metric_dict["std_time"]
        if "std_val" in metric_dict: return metric_dict["std_val"]
        if "std" in metric_dict: return metric_dict["std"]
        
        # Last resort: Tail / 2 (conservative estimate)
        if "tail_length" in metric_dict: return metric_dict["tail_length"] / 2
        return 0.0

    if "min_time" in d or "ci_95" in d:
        return extract_err(d)
        
    if "avg_latency" in d: # Streaming
        return extract_err(d["avg_latency"])
        
    if isinstance(d, dict):
        # SIMD or other nested (fetch first available)
        for v in d.values():
            if isinstance(v, dict):
                return extract_err(v)
    return 0.0


# =============================================================================
# Chart 1: Overall Wall-Time Comparison
# =============================================================================
def plot_01_overall(data):
    names = []
    gil_vals = []
    gil_errs = []
    nogil_vals = []
    nogil_errs = []
    speedups = []
    
    # Define which workloads/modes to show for a complete "Parallel Matrix"
    configs = [
        ("Data Preprocessing", "threading_8", "Data Prep"),
        ("Image Processing", "threading_8", "Image Proc"),
        ("ML Training", "linear_reg_threading_8", "ML: LinReg"),
        ("ML Training", "random_forest_threading_8", "ML: RandForest"),
        ("Streaming", "workers_8", "Streaming"),
        ("SIMD Vectorization", "matrix: matmul", "SIMD: Matmul"),
        ("SIMD Vectorization", "sequential: numpy", "SIMD: NumPy Seq"),
        ("Mandelbrot", "threading_8", "Mandelbrot"),
        ("Monte Carlo", "threading_8", "Monte Carlo"),
        ("Fibonacci", "threading_8", "Fibonacci")
    ]

    for wl_key, mode_key, disp_name in configs:
        wl = data["workloads"].get(wl_key)
        if not wl: continue
        
        # Verify mode exists
        if mode_key not in wl.get("gil", {}).get("merged", {}) and \
           mode_key not in wl.get("nogil", {}).get("merged", {}):
            if "sequential" in wl.get("gil", {}).get("merged", {}):
                mode_key = "sequential"
            else:
                continue

        t_gil = _get_avg(wl, "gil", mode_key)
        err_gil = _get_std(wl, "gil", mode_key)
        t_nogil = _get_avg(wl, "nogil", mode_key)
        err_nogil = _get_std(wl, "nogil", mode_key)
        
        if t_gil == 0: continue
            
        gil_vals.append(100.0)
        gil_errs.append((err_gil / t_gil) * 100.0)
        
        nogil_vals.append((t_nogil / t_gil) * 100.0)
        nogil_errs.append((err_nogil / t_gil) * 100.0)
        
        # Speedup label
        if t_nogil < t_gil:
            factor = t_gil/t_nogil
            speedups.append(f"{factor:.1f}x Faster" if factor >= 1.05 else "")
        else:
            factor = t_nogil/t_gil
            speedups.append(f"{factor:.1f}x Slower" if factor >= 1.05 else "")

        setup = mode_key.replace("threading_", "T").replace("workers_", "W")
        if setup == "sequential": setup = "Seq"
        names.append(f"{disp_name}\n({setup})")

    x = np.arange(len(names))
    w = 0.38

    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Grid and style
    ax.set_facecolor('#fdfdfd')
    ax.grid(axis='y', alpha=0.3, which='major', linestyle='-', linewidth=1)
    ax.grid(axis='y', alpha=0.1, which='minor', linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)
    
    gil_yerr = [np.zeros(len(gil_errs)), gil_errs]
    nogil_yerr = [np.zeros(len(nogil_errs)), nogil_errs]

    # Plot bars with subtle gradients (simulated by alpha)
    ax.bar(x - w/2, gil_vals, w, yerr=gil_yerr, capsize=5, ecolor='#333333',
           label='With GIL (100% Baseline)', color=GIL_COLOR, alpha=0.85, edgecolor='black', linewidth=0.8)
    ax.bar(x + w/2, nogil_vals, w, yerr=nogil_yerr, capsize=5, ecolor='#333333',
           label='Without GIL (Free-threaded)', color=NOGIL_COLOR, alpha=0.85, edgecolor='black', linewidth=0.8)

    # Annotations with collision avoidance
    for i in range(len(names)):
        # GIL Label
        ax.text(x[i] - w/2, 100 + gil_errs[i] + 4, "100%", ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#444444')
        
        # No-GIL Label (higher if close to 100% to avoid collision)
        pct_val = nogil_vals[i]
        err_val = nogil_errs[i]
        
        # Vertical stacking if bars are nearly equal height
        y_pos = pct_val + err_val + 5
        if abs(pct_val - 100) < 15:
            # Shift No-GIL labels slightly higher to clear the shared "100%" space
            y_pos += 8
            
        label = f"{pct_val:.0f}%"
        if speedups[i]:
            label += f"\n({speedups[i]})"
            
        ax.text(x[i] + w/2, y_pos, label, ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='#c0392b')

    ax.set_ylabel('Execution Time Relative to GIL (%)', fontsize=12, labelpad=15)
    ax.set_title('Python 3.14 Performance Evolution: Normalized Matrix\n(Bars = Min time, Whiskers = 95% Confidence Interval; Lower is Faster)', fontsize=15, pad=35, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=10)
    ax.axhline(100, color='black', linewidth=1.5, linestyle='--', alpha=0.4, label='GIL Baseline')
    
    # Dynamic Y-limit
    all_tops = [nogil_vals[i] + nogil_errs[i] for i in range(len(nogil_vals))] + [100 + e for e in gil_errs]
    ax.set_ylim(0, max(all_tops) * 1.35)
    ax.yaxis.set_minor_locator(plt.MultipleLocator(10))
    
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='gray', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "01_overall_comparison.png", dpi=200)
    plt.close()
    print("✅ 01_overall_comparison.png (Polished aesthetics)")


# =============================================================================
# Chart 2: Data Preprocessing threading scaling
# =============================================================================
def plot_02_data_preprocessing(data):
    wl = data["workloads"].get("Data Preprocessing")
    if not wl: return

    modes = ["sequential", "threading_2", "threading_4", "threading_8"]
    labels = ["1 (seq)", "2", "4", "8"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for variant, color, lbl in [("gil", GIL_COLOR, "GIL"), ("nogil", NOGIL_COLOR, "No-GIL")]:
        times = [_get_avg(wl, variant, m) for m in modes]
        stds = [_get_std(wl, variant, m) for m in modes]
        ax1.errorbar(labels, times, yerr=stds, marker='o', linewidth=2, markersize=8, label=lbl, color=color, capsize=4)
        if times[0] > 0:
            speedups = [times[0] / t if t > 0 else 0 for t in times]
            ax2.plot(labels, speedups, marker='s', linewidth=2, markersize=8, label=lbl, color=color)

    ax1.set_xlabel('Thread Count')
    ax1.set_ylabel('Execution Time (min\n+ tail)')
    ax1.set_title(f'Data Prep: Execution Time\n({wl.get("params", "")})')
    ax1.set_ylim(bottom=0)
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(labels, [1, 2, 4, 8], '--', alpha=0.3, color=IDEAL_COLOR, label='Ideal linear')
    ax2.set_xlabel('Thread Count'); ax2.set_ylabel('Speedup (vs Sequential)')
    ax2.set_title('Data Preprocessing: Speedup')
    ax2.set_ylim(bottom=0); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Data Preprocessing Benchmark (N={n_runs} measurements)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "02_data_preprocessing.png")
    plt.close()
    print("✅ 02_data_preprocessing.png")


# =============================================================================
# Chart 3: Image Processing threading scaling
# =============================================================================
def plot_03_image_processing(data):
    wl = data["workloads"].get("Image Processing")
    if not wl:
        return

    modes = ["sequential", "threading_2", "threading_4", "threading_8"]
    labels = ["1 (seq)", "2", "4", "8"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for variant, color, lbl in [("gil", GIL_COLOR, "GIL"), ("nogil", NOGIL_COLOR, "No-GIL")]:
        times = [_get_avg(wl, variant, m) for m in modes]
        stds = [_get_std(wl, variant, m) for m in modes]

        ax1.errorbar(labels, times, yerr=stds, marker='o', linewidth=2,
                     markersize=8, label=lbl, color=color, capsize=4)

        if times[0] > 0:
            speedups = [times[0] / t if t > 0 else 0 for t in times]
            ax2.plot(labels, speedups, marker='s', linewidth=2, markersize=8,
                     label=lbl, color=color)

    ax1.set_xlabel('Thread Count')
    ax1.set_ylabel('Execution Time (min\n+ tail)')
    ax1.set_title(f'Image Proc: Execution Time\n({wl.get("params", "")})')
    ax1.set_ylim(bottom=0)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(labels, [1, 2, 4, 8], '--', alpha=0.3, color=IDEAL_COLOR, label='Ideal linear')
    ax2.set_xlabel('Thread Count')
    ax2.set_ylabel('Speedup (vs Sequential)')
    ax2.set_title('Image Processing: Speedup')
    ax2.set_ylim(bottom=0)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Image Processing Benchmark (N={n_runs} measurements)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "03_image_processing.png")
    plt.close()
    print("✅ 03_image_processing.png")


# =============================================================================
# Chart 4: ML Training
# =============================================================================
def plot_04_ml_training(data):
    wl = data["workloads"].get("ML Training")
    if not wl:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Sequential algorithms
    algorithms = ["linear_regression_seq", "random_forest_seq"]
    algo_labels = ["Linear\nRegression", "Random\nForest"]

    x = np.arange(len(algorithms))
    w = 0.35

    for variant, color, lbl, offset in [
        ("gil", GIL_COLOR, "With GIL", -w/2),
        ("nogil", NOGIL_COLOR, "Without GIL", w/2),
    ]:
        avgs = [_get_avg(wl, variant, a) for a in algorithms]
        stdevs = [_get_std(wl, variant, a) for a in algorithms]
        ax1.bar(x + offset, avgs, w, yerr=stdevs, capsize=3,
                label=lbl, color=color, edgecolor='white', linewidth=0.5)

    ax1.set_xlabel('Algorithm')
    ax1.set_ylabel('Training Time (seconds) [Min + Tail]')
    ax1.set_title(f'ML Training Time\n({wl.get("params", "")})')
    ax1.set_xticks(x)
    ax1.set_xticklabels(algo_labels)
    ax1.set_ylim(bottom=0)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Threading scaling for linear regression
    thread_modes = ["linear_regression_seq", "linear_reg_threading_2", "linear_reg_threading_4", "linear_reg_threading_8"]
    thread_labels = ["1 (seq)", "2", "4", "8"]

    for variant, color, lbl in [("gil", GIL_COLOR, "GIL"), ("nogil", NOGIL_COLOR, "No-GIL")]:
        times = [_get_avg(wl, variant, m) for m in thread_modes]
        stds = [_get_std(wl, variant, m) for m in thread_modes]

        if times[0] > 0:
            speedups = [times[0] / t if t > 0 else 0 for t in times]
            ax2.plot(thread_labels, speedups, marker='o', linewidth=2,
                     markersize=8, label=lbl, color=color)

    ax2.plot(thread_labels, [1, 2, 4, 8], '--', alpha=0.3, color=IDEAL_COLOR, label='Ideal linear')
    ax2.set_xlabel('Thread Count')
    ax2.set_ylabel('Speedup (vs Sequential)')
    ax2.set_title('Linear Regression: Scaling')
    ax2.set_ylim(bottom=0)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'ML Training Benchmark (N={n_runs} measurements)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "04_ml_training.png")
    plt.close()
    print("✅ 04_ml_training.png")


# =============================================================================
# Chart 5: Box plot comparison (key modes)
# =============================================================================
def plot_05_boxplot(data):
    """Box plot showing distribution of measurements for key modes."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()

    plot_configs = [
        ("Data Preprocessing", "sequential", "Data Prep: Sequential"),
        ("Data Preprocessing", "threading_8", "Data Prep: 8 Threads"),
        ("Image Processing", "threading_8", "Image Proc: 8 Threads"),
        ("ML Training", "linear_reg_threading_8", "ML: LR 8 Threads"),
        ("Streaming", "workers_8", "Streaming: 8 Threads"),
        ("Mandelbrot", "sequential", "Mandelbrot: Sequential"),
        ("Mandelbrot", "threading_8", "Mandelbrot: 8 Threads"),
        ("Fibonacci", "threading_8", "Fibonacci: 8 Threads"),
        ("Monte Carlo", "threading_8", "Monte Carlo: 8 Threads"),
    ]

    for ax, (wl_name, mode, title) in zip(axes, plot_configs):
        wl = data["workloads"].get(wl_name)
        if not wl:
            continue

        gil_times = _get_times(wl, "gil", mode)
        nogil_times = _get_times(wl, "nogil", mode)

        if not gil_times or not nogil_times:
            ax.set_title(f"{title}\n(no data)")
            continue

        bp = ax.boxplot(
            [gil_times, nogil_times],
            tick_labels=['GIL', 'No-GIL'],
            patch_artist=True,
            widths=0.5,
        )
        bp['boxes'][0].set_facecolor(GIL_COLOR)
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor(NOGIL_COLOR)
        bp['boxes'][1].set_alpha(0.7)

        # Welch's t-test
        t_stat, p_val = stats.ttest_ind(gil_times, nogil_times, equal_var=False)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        speedup = np.min(gil_times) / np.min(nogil_times) if np.min(nogil_times) > 0 else 0

        ax.set_title(f"{title}\nSpeedup (3.14t vs GIL): {speedup:.2f}x  p={p_val:.4f} {sig}", fontsize=10)
        ax.set_ylabel('Time (s)')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle(f'Distribution of Measurements (N={n_runs} per group)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "05_boxplots.png")
    plt.close()
    print("✅ 05_boxplots.png")


# =============================================================================
# Chart 5b: Streaming Latency & Throughput
# =============================================================================
def plot_05b_streaming(data):
    """Streaming benchmark: latency and throughput by worker count."""
    wl = data["workloads"].get("Streaming")
    if not wl:
        print("⚠️ Streaming workload not found")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    worker_modes = ["workers_1", "workers_2", "workers_4", "workers_8"]
    labels = ["1", "2", "4", "8"]

    for variant, color, lbl in [
        ("gil", GIL_COLOR, "With GIL"),
        ("nogil", NOGIL_COLOR, "Without GIL"),
    ]:
        merged = wl.get(variant, {}).get("merged", {})
        latencies = []
        throughputs = []

        for wm in worker_modes:
            d = merged.get(wm, {})
            if isinstance(d, dict):
                # Streaming results use new nested dict structure
                avg_lat_stats = d.get("avg_latency", {})
                tp_stats = d.get("throughput", {})
                
                # Latency (using min_time if available, else avg_time)
                lat_val = avg_lat_stats.get("min_time") or avg_lat_stats.get("avg_time", 0)
                latencies.append(lat_val * 1000)  # to ms

                # Throughput (using max_val if available, else avg_val)
                tp_val = tp_stats.get("max_val") or tp_stats.get("avg_val", 0)
                throughputs.append(tp_val)
            else:
                latencies.append(0)
                throughputs.append(0)

        if any(l > 0 for l in latencies):
            ax1.plot(labels, latencies, marker='o', linewidth=2,
                     markersize=8, label=lbl, color=color)
        if any(t > 0 for t in throughputs):
            ax2.plot(labels, throughputs, marker='s', linewidth=2,
                     markersize=8, label=lbl, color=color)

    ax1.set_xlabel('Worker Count')
    ax1.set_ylabel('Average Latency (ms)')
    ax1.set_title(f'Streaming: Event Latency\n({wl.get("params", "")})')
    ax1.set_ylim(bottom=0)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Worker Count')
    ax2.set_ylabel('Throughput (events/second)')
    ax2.set_title('Streaming: Throughput')
    ax2.set_ylim(bottom=0)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Streaming Benchmark (N={n_runs} measurements)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "05_streaming.png")
    plt.close()
    print("✅ 05_streaming.png")


# =============================================================================
# Chart 6: SIMD Comparison
# =============================================================================
def plot_06_simd(data):
    wl = data["workloads"].get("SIMD Vectorization")
    if not wl:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Sequential methods
    for variant, color, lbl in [("gil", GIL_COLOR, "GIL"), ("nogil", NOGIL_COLOR, "No-GIL")]:
        merged = wl.get(variant, {}).get("merged", {})
        seq = merged.get("sequential", {})
        if not isinstance(seq, dict):
            continue

        methods = []
        means = []
        stds = []
        for k, v in seq.items():
            if isinstance(v, dict):
                # New format uses min_time
                val = v.get("min_time") or v.get("avg_time", 0)
                if val > 0:
                    methods.append(k.replace("_", "\n"))
                    means.append(val)
                    stds.append(v.get("tail_length", 0))

        if methods:
            x = np.arange(len(methods))
            w = 0.35
            offset = -w/2 if variant == "gil" else w/2
            ax1.bar(x + offset, means, w, yerr=stds, capsize=3,
                    label=lbl, color=color, edgecolor='white')
            ax1.set_xticks(x)
            ax1.set_xticklabels(methods, fontsize=8)

    ax1.set_ylabel('Execution Time (min\n+ tail)')
    ax1.set_title(f'SIMD Seq Methods\n({wl.get("params", "")})')
    ax1.set_ylim(bottom=0)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Threading
    thread_counts = ["2", "4", "8"]
    for variant, color, lbl in [("gil", GIL_COLOR, "GIL"), ("nogil", NOGIL_COLOR, "No-GIL")]:
        merged = wl.get(variant, {}).get("merged", {})
        threaded = merged.get("threaded", {})
        if not isinstance(threaded, dict):
            continue

        times = []
        for tc in [2, 4, 8]:
            key = f"numpy_threaded_{tc}"
            if key in threaded and isinstance(threaded[key], dict):
                # New format uses min_time
                times.append(threaded[key].get("min_time") or threaded[key].get("avg_time", 0))
            else:
                times.append(0)

        if any(t > 0 for t in times):
            ax2.plot(thread_counts, times, marker='o', linewidth=2,
                     markersize=8, label=lbl, color=color)

    ax2.set_xlabel('Thread Count')
    ax2.set_ylabel('Execution Time (seconds)')
    ax2.set_title('NumPy Threaded scaling')
    ax2.set_ylim(bottom=0)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('SIMD/Vectorization Benchmark', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "06_simd_vectorization.png")
    plt.close()
    print("✅ 06_simd_vectorization.png")


# =============================================================================
# Chart 7: Speedup Heatmap
# =============================================================================
def plot_07_heatmap(data):
    """Heatmap of mode-level speedups."""
    workloads_data = {}

    for name, wl in data["workloads"].items():
        g_merged = wl.get("gil", {}).get("merged", {})
        n_merged = wl.get("nogil", {}).get("merged", {})
        if not g_merged or not n_merged:
            continue

        speedups = {}
        for key in g_merged:
            if key not in n_merged:
                continue
            g_d = g_merged[key]
            n_d = n_merged[key]
            if isinstance(g_d, dict) and "avg_time" in g_d and \
               isinstance(n_d, dict) and "avg_time" in n_d:
                gt = g_d["min_time"]
                nt = n_d["min_time"]
                if nt > 0:
                    speedups[key] = round(gt / nt, 2)

        if speedups:
            workloads_data[name] = speedups

    if not workloads_data:
        print("⚠️ No heatmap data")
        return

    all_modes = sorted(set(m for modes in workloads_data.values() for m in modes))
    wl_names = list(workloads_data.keys())

    matrix = np.full((len(wl_names), len(all_modes)), np.nan)
    for i, n in enumerate(wl_names):
        for j, m in enumerate(all_modes):
            matrix[i, j] = workloads_data[n].get(m, np.nan)

    fig, ax = plt.subplots(figsize=(max(14, len(all_modes) * 1.5),
                                     max(4, len(wl_names) * 1.5)))

    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0.3, vmax=3.0)

    ax.set_xticks(range(len(all_modes)))
    ax.set_xticklabels([m.replace("_", "\n") for m in all_modes],
                       rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(wl_names)))
    ax.set_yticklabels(wl_names)

    for i in range(len(wl_names)):
        for j in range(len(all_modes)):
            if not np.isnan(matrix[i, j]):
                v = matrix[i, j]
                c = "white" if v < 0.6 or v > 2.0 else "black"
                ax.text(j, i, f"{v:.2f}x", ha="center", va="center", fontsize=7, color=c)

    plt.colorbar(im, label='Speedup (GIL / No-GIL)\n>1 = No-GIL faster')
    ax.set_title('Performance Speedup Heatmap (N=15 measurements)', fontsize=14)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "07_speedup_heatmap.png")
    plt.close()
    print("✅ 07_speedup_heatmap.png")


# =============================================================================
# Chart 8: Confidence Intervals
# =============================================================================
def plot_08_ci(data):
    """Confidence interval chart for key comparisons."""
    fig, ax = plt.subplots(figsize=(12, 8))

    comparisons = []

    for name, wl in data["workloads"].items():
        g_merged = wl.get("gil", {}).get("merged", {})
        n_merged = wl.get("nogil", {}).get("merged", {})

        for key in g_merged:
            if key not in n_merged:
                continue
            g_d = g_merged[key]
            n_d = n_merged[key]
            if isinstance(g_d, dict) and "all_times" in g_d and \
               isinstance(n_d, dict) and "all_times" in n_d:
                gt = np.array(g_d["all_times"])
                nt = np.array(n_d["all_times"])
                n_obs = min(len(gt), len(nt))

                # Compute speedup per sample (independent samples)
                speedups_all = []
                for g in gt:
                    for n in nt:
                        if n > 0:
                            speedups_all.append(g / n)

                if speedups_all:
                    arr = np.array(speedups_all)
                    mean_sp = np.mean(arr)
                    ci_lo = np.percentile(arr, 2.5)
                    ci_hi = np.percentile(arr, 97.5)

                    comparisons.append({
                        "label": f"{name}\n{key}",
                        "speedup": mean_sp,
                        "ci_lo": ci_lo,
                        "ci_hi": ci_hi,
                    })

    if not comparisons:
        print("⚠️ No CI data")
        return

    # Sort by speedup
    comparisons.sort(key=lambda x: x["speedup"])

    y = np.arange(len(comparisons))
    means = [c["speedup"] for c in comparisons]
    lo = [c["speedup"] - c["ci_lo"] for c in comparisons]
    hi = [c["ci_hi"] - c["speedup"] for c in comparisons]
    labels = [c["label"] for c in comparisons]
    colors = [NOGIL_COLOR if m > 1 else GIL_COLOR for m in means]

    ax.barh(y, means, xerr=[lo, hi], capsize=3, color=colors,
            edgecolor='white', linewidth=0.5, alpha=0.8)
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5,
               label='No difference')
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Speedup (GIL / No-GIL)\n>1 means No-GIL is faster')
    ax.set_title('Speedup with 95% Confidence Intervals\n(Blue=GIL faster, Red=No-GIL faster)', fontsize=14)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "08_confidence_intervals.png")
    plt.close()
    print("✅ 08_confidence_intervals.png")


# =============================================================================
# Chart 9: CPU Utilization Comparison
# =============================================================================
def plot_09_cpu_utilization(data):
    """Bar chart of CPU utilization across workloads."""
    names = []
    gil_cpu_means = []
    nogil_cpu_means = []
    gil_cpu_peaks = []
    nogil_cpu_peaks = []

    for name, wl in data["workloads"].items():
        gr = wl.get("gil", {}).get("resource_stats", {})
        nr = wl.get("nogil", {}).get("resource_stats", {})
        if not gr and not nr:
            continue

        names.append(name)
        gil_cpu_means.append(gr.get("cpu_mean_percent", {}).get("mean", 0) or 0)
        nogil_cpu_means.append(nr.get("cpu_mean_percent", {}).get("mean", 0) or 0)
        gil_cpu_peaks.append(gr.get("cpu_peak_percent", {}).get("mean", 0) or 0)
        nogil_cpu_peaks.append(nr.get("cpu_peak_percent", {}).get("mean", 0) or 0)

    if not names:
        print("⚠️ No CPU data available (run benchmarks with updated runner)")
        return

    x = np.arange(len(names))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Mean CPU %
    ax1.bar(x - w/2, gil_cpu_means, w, label='With GIL', color=GIL_COLOR,
            edgecolor='white', linewidth=0.5)
    ax1.bar(x + w/2, nogil_cpu_means, w, label='Without GIL', color=NOGIL_COLOR,
            edgecolor='white', linewidth=0.5)
    for i in range(len(names)):
        if gil_cpu_means[i] > 0:
            ax1.annotate(f'{gil_cpu_means[i]:.0f}%',
                         xy=(x[i]-w/2, gil_cpu_means[i]),
                         xytext=(0, 5), textcoords="offset points", ha='center', fontsize=8)
        if nogil_cpu_means[i] > 0:
            ax1.annotate(f'{nogil_cpu_means[i]:.0f}%',
                         xy=(x[i]+w/2, nogil_cpu_means[i]),
                         xytext=(0, 5), textcoords="offset points", ha='center', fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha='right')
    ax1.set_ylabel('CPU Utilization (%)')
    ax1.set_title('Mean CPU Utilization')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Peak CPU %
    ax2.bar(x - w/2, gil_cpu_peaks, w, label='With GIL', color=GIL_COLOR,
            edgecolor='white', linewidth=0.5)
    ax2.bar(x + w/2, nogil_cpu_peaks, w, label='Without GIL', color=NOGIL_COLOR,
            edgecolor='white', linewidth=0.5)
    for i in range(len(names)):
        if gil_cpu_peaks[i] > 0:
            ax2.annotate(f'{gil_cpu_peaks[i]:.0f}%',
                         xy=(x[i]-w/2, gil_cpu_peaks[i]),
                         xytext=(0, 5), textcoords="offset points", ha='center', fontsize=8)
        if nogil_cpu_peaks[i] > 0:
            ax2.annotate(f'{nogil_cpu_peaks[i]:.0f}%',
                         xy=(x[i]+w/2, nogil_cpu_peaks[i]),
                         xytext=(0, 5), textcoords="offset points", ha='center', fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15, ha='right')
    ax2.set_ylabel('Peak CPU Utilization (%)')
    ax2.set_title('Peak CPU Utilization')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle('CPU Utilization: GIL vs Free-threaded Python', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "09_cpu_utilization.png")
    plt.close()
    print("✅ 09_cpu_utilization.png")


# =============================================================================
# Chart 10: Memory Usage Comparison
# =============================================================================
def plot_10_memory_usage(data):
    """Bar chart of memory usage across workloads."""
    names = []
    gil_mem_peaks = []
    nogil_mem_peaks = []
    gil_mem_means = []
    nogil_mem_means = []

    for name, wl in data["workloads"].items():
        gr = wl.get("gil", {}).get("resource_stats", {})
        nr = wl.get("nogil", {}).get("resource_stats", {})
        if not gr and not nr:
            continue

        names.append(name)
        gil_mem_peaks.append(gr.get("mem_peak_mb", {}).get("mean", 0) or 0)
        nogil_mem_peaks.append(nr.get("mem_peak_mb", {}).get("mean", 0) or 0)
        gil_mem_means.append(gr.get("mem_mean_mb", {}).get("mean", 0) or 0)
        nogil_mem_means.append(nr.get("mem_mean_mb", {}).get("mean", 0) or 0)

    if not names:
        print("⚠️ No memory data available (run benchmarks with updated runner)")
        return

    x = np.arange(len(names))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Peak Memory
    ax1.bar(x - w/2, gil_mem_peaks, w, label='With GIL', color=GIL_COLOR,
            edgecolor='white', linewidth=0.5)
    ax1.bar(x + w/2, nogil_mem_peaks, w, label='Without GIL', color=NOGIL_COLOR,
            edgecolor='white', linewidth=0.5)
    for i in range(len(names)):
        if gil_mem_peaks[i] > 0:
            ax1.annotate(f'{gil_mem_peaks[i]:.0f}',
                         xy=(x[i]-w/2, gil_mem_peaks[i]),
                         xytext=(0, 5), textcoords="offset points", ha='center', fontsize=8)
        if nogil_mem_peaks[i] > 0:
            ax1.annotate(f'{nogil_mem_peaks[i]:.0f}',
                         xy=(x[i]+w/2, nogil_mem_peaks[i]),
                         xytext=(0, 5), textcoords="offset points", ha='center', fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha='right')
    ax1.set_ylabel('Peak Memory (MB)')
    ax1.set_title('Peak Memory Usage')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Mean Memory
    ax2.bar(x - w/2, gil_mem_means, w, label='With GIL', color=GIL_COLOR,
            edgecolor='white', linewidth=0.5)
    ax2.bar(x + w/2, nogil_mem_means, w, label='Without GIL', color=NOGIL_COLOR,
            edgecolor='white', linewidth=0.5)
    for i in range(len(names)):
        if gil_mem_means[i] > 0:
            ax2.annotate(f'{gil_mem_means[i]:.0f}',
                         xy=(x[i]-w/2, gil_mem_means[i]),
                         xytext=(0, 5), textcoords="offset points", ha='center', fontsize=8)
        if nogil_mem_means[i] > 0:
            ax2.annotate(f'{nogil_mem_means[i]:.0f}',
                         xy=(x[i]+w/2, nogil_mem_means[i]),
                         xytext=(0, 5), textcoords="offset points", ha='center', fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15, ha='right')
    ax2.set_ylabel('Mean Memory (MB)')
    ax2.set_title('Mean Memory Usage')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle('Memory Usage: GIL vs Free-threaded Python', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "10_memory_usage.png")
    plt.close()
    print("✅ 10_memory_usage.png")


# =============================================================================
# Statistical Analysis Report
# =============================================================================
def run_statistical_analysis(data):
    print(f"\n{'='*70}")
    print("COMPREHENSIVE STATISTICAL ANALYSIS")
    print(f"{'='*70}")

    analysis = {}

    for name, wl in data["workloads"].items():
        g_merged = wl.get("gil", {}).get("merged", {})
        n_merged = wl.get("nogil", {}).get("merged", {})
        if not g_merged or not n_merged:
            continue

        print(f"\n{'─'*60}")
        print(f"  {name}")
        print(f"{'─'*60}")
        print(f"  {'Mode':<30} {'GIL avg':>10} {'NoGIL avg':>10} {'Speedup':>9} "
              f"{'N':>4} {'t-stat':>8} {'p-value':>9} {'Sig':>5}")

        def _process_recursive(prefix, g_node, n_node):
            for k in sorted(g_node.keys()):
                if k not in n_node:
                    continue
                gd = g_node[k]
                nd = n_node[k]

                full_key = f"{prefix}: {k}" if prefix else k
                if isinstance(gd, dict):
                    # Check for timing data or general values
                    gt = gd.get("all_times") or gd.get("all_vals") or gd.get("all_values")
                    nt = nd.get("all_times") or nd.get("all_vals") or nd.get("all_values")

                    if gt and nt and len(gt) > 0 and len(nt) > 0:
                        is_throughput = "throughput" in full_key.lower()
                        
                        if is_throughput:
                            g_mean = gd.get("max_val") or gd.get("mean") or np.max(gt)
                            n_mean = nd.get("max_val") or nd.get("mean") or np.max(nt)
                        else:
                            g_mean = gd.get("min_time") or gd.get("mean") or np.min(gt)
                            n_mean = nd.get("min_time") or nd.get("mean") or np.min(nt)
                            
                        speedup = g_mean / n_mean if n_mean > 0 else 0

                        # Stats
                        if len(gt) > 1 and len(nt) > 1:
                            t_stat, p_val = stats.ttest_ind(gt, nt, equal_var=False)
                            # Cohen's d
                            pooled_std = np.sqrt((np.var(gt, ddof=1) + np.var(nt, ddof=1)) / 2)
                            cohens_d = (np.mean(gt) - np.mean(nt)) / pooled_std if pooled_std > 0 else 0
                        else:
                            t_stat, p_val, cohens_d = 0.0, 1.0, 0.0

                        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                        n_obs = min(len(gt), len(nt))

                        print(f"  {full_key[:30]:<30} {g_mean:>9.4f}s {n_mean:>9.4f}s {speedup:>8.2f}x "
                              f"{n_obs:>4} {t_stat:>8.2f} {p_val:>8.4f} {sig:>5}")

                        analysis[f"{name}_{full_key}"] = {
                            "gil_mean": float(g_mean),
                            "nogil_mean": float(n_mean),
                            "speedup": float(speedup),
                            "p_value": float(p_val),
                            "significant": bool(p_val < 0.05),
                        }
                    else:
                        # Continue recursion for truly nested structs (SIMD)
                        if any(isinstance(v, dict) for v in gd.values()):
                            _process_recursive(full_key, gd, nd)

        _process_recursive("", g_merged, n_merged)

    print(f"\n  Significance levels: *** p<0.001, ** p<0.01, * p<0.05")

    # Save
    out = TARGET_DIR / "statistical_analysis_multirun.json"
    with open(out, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\n✅ Saved: {out}")

    return analysis


# =============================================================================
# Main
# =============================================================================
def main():
    setup_style()

    print("=" * 70)
    print("MULTI-RUN VISUALIZATION & ANALYSIS")
    print("=" * 70)

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()

    n_iter = data.get("metadata", {}).get("num_iterations", "?")
    print(f"Data from {n_iter} iterations\n")

    plot_01_overall(data)
    plot_02_data_preprocessing(data)
    plot_03_image_processing(data)
    plot_04_ml_training(data)
    plot_05_boxplot(data)
    plot_05b_streaming(data)
    plot_06_simd(data)
    plot_07_heatmap(data)
    plot_08_ci(data)
    plot_09_cpu_utilization(data)
    plot_10_memory_usage(data)
    
    # New High-Impact Workloads
    _plot_generic_scaling(data, "Mandelbrot", "11_mandelbrot.png")
    _plot_generic_scaling(data, "Monte Carlo", "12_monte_carlo.png")
    _plot_generic_scaling(data, "Fibonacci", "13_fibonacci.png")

    run_statistical_analysis(data)


def _plot_generic_scaling(data, wl_name, filename):
    wl = data["workloads"].get(wl_name)
    if not wl: return

    modes = ["sequential", "threading_2", "threading_4", "threading_8"]
    labels = ["1 (seq)", "2", "4", "8"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for variant, color, lbl in [("gil", GIL_COLOR, "GIL"), ("nogil", NOGIL_COLOR, "No-GIL")]:
        times = [_get_avg(wl, variant, m) for m in modes]
        stds = [_get_std(wl, variant, m) for m in modes]
        ax1.errorbar(labels, times, yerr=stds, marker='o', linewidth=2, markersize=8, label=lbl, color=color, capsize=4)
        if times[0] > 0:
            speedups = [times[0] / t if t > 0 else 0 for t in times]
            ax2.plot(labels, speedups, marker='s', linewidth=2, markersize=8, label=lbl, color=color)

    ax1.set_xlabel('Thread Count')
    ax1.set_ylabel('Execution Time (min\n+ tail)')
    ax1.set_title(f'{wl_name}: Execution Time\n({wl.get("params", "")})')
    ax1.set_ylim(bottom=0)
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(labels, [1, 2, 4, 8], '--', alpha=0.3, color=IDEAL_COLOR, label='Ideal linear')
    ax2.set_xlabel('Thread Count'); ax2.set_ylabel('Speedup (vs Sequential)')
    ax2.set_title(f'{wl_name}: Speedup Scaling')
    ax2.set_ylim(bottom=0); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.suptitle(f'{wl_name} Benchmark (N={n_runs} measurements)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / filename)
    plt.close()
    print(f"✅ {filename}")

    print(f"\n{'='*70}")
    print(f"All charts saved to: {CHARTS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
