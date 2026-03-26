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


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
CHARTS_DIR = RESULTS_DIR / "charts"


def load_data() -> dict:
    f = RESULTS_DIR / "multi_run_master.json"
    if not f.exists():
        print(f"❌ Not found: {f}")
        sys.exit(1)
    with open(f) as fh:
        return json.load(fh)


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
    """Get avg_time for a mode."""
    d = _get_mode_data(wl, variant, mode)
    return d.get("avg_time", 0)


def _get_std(wl, variant, mode):
    """Get std_time for a mode."""
    d = _get_mode_data(wl, variant, mode)
    return d.get("std_time", 0)


# =============================================================================
# Chart 1: Overall Wall-Time Comparison
# =============================================================================
def plot_01_overall(data):
    names = []
    gil_means, gil_stds = [], []
    nogil_means, nogil_stds = [], []

    for name, wl in data["workloads"].items():
        names.append(name)
        gs = wl["gil"]["wall_stats"]
        ns = wl["nogil"]["wall_stats"]
        gil_means.append(gs["mean"])
        gil_stds.append(gs["std"])
        nogil_means.append(ns["mean"])
        nogil_stds.append(ns["std"])

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - w/2, gil_means, w, yerr=gil_stds, capsize=4,
           label='With GIL', color=GIL_COLOR, edgecolor='white', linewidth=0.5)
    ax.bar(x + w/2, nogil_means, w, yerr=nogil_stds, capsize=4,
           label='Without GIL (Free-threaded)', color=NOGIL_COLOR, edgecolor='white', linewidth=0.5)

    # Value labels
    for i in range(len(names)):
        ax.annotate(f'{gil_means[i]:.1f}s',
                    xy=(x[i] - w/2, gil_means[i] + gil_stds[i]),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=8)
        ax.annotate(f'{nogil_means[i]:.1f}s',
                    xy=(x[i] + w/2, nogil_means[i] + nogil_stds[i]),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=8)

    ax.set_xlabel('Workload')
    ax.set_ylabel('Wall Time (seconds) ± std')
    ax.set_title('Overall Performance: GIL vs Free-threaded Python 3.13\n(5 runs, error bars = 1σ)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "01_overall_comparison.png")
    plt.close()
    print("✅ 01_overall_comparison.png")


# =============================================================================
# Chart 2: Data Preprocessing threading scaling
# =============================================================================
def plot_02_data_preprocessing(data):
    wl = data["workloads"].get("Data Preprocessing")
    if not wl:
        return

    modes = ["sequential", "threading_2", "threading_4", "threading_8"]
    labels = ["1 (seq)", "2", "4", "8"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for variant, color, lbl in [("gil", GIL_COLOR, "With GIL"), ("nogil", NOGIL_COLOR, "Without GIL")]:
        times = [_get_avg(wl, variant, m) for m in modes]
        stds = [_get_std(wl, variant, m) for m in modes]

        ax1.errorbar(labels, times, yerr=stds, marker='o', linewidth=2,
                     markersize=8, label=lbl, color=color, capsize=4)

        if times[0] > 0:
            speedups = [times[0] / t if t > 0 else 0 for t in times]
            ax2.plot(labels, speedups, marker='s', linewidth=2, markersize=8,
                     label=lbl, color=color)

    ax1.set_xlabel('Thread Count')
    ax1.set_ylabel('Execution Time (seconds) ± σ')
    ax1.set_title('Data Preprocessing: Execution Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(labels, [1, 2, 4, 8], '--', alpha=0.3, color=IDEAL_COLOR, label='Ideal linear')
    ax2.set_xlabel('Thread Count')
    ax2.set_ylabel('Speedup (vs Sequential)')
    ax2.set_title('Data Preprocessing: Speedup')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Data Preprocessing Benchmark (N=15 measurements)', fontsize=14, y=1.02)
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

    for variant, color, lbl in [("gil", GIL_COLOR, "With GIL"), ("nogil", NOGIL_COLOR, "Without GIL")]:
        times = [_get_avg(wl, variant, m) for m in modes]
        stds = [_get_std(wl, variant, m) for m in modes]

        ax1.errorbar(labels, times, yerr=stds, marker='o', linewidth=2,
                     markersize=8, label=lbl, color=color, capsize=4)

        if times[0] > 0:
            speedups = [times[0] / t if t > 0 else 0 for t in times]
            ax2.plot(labels, speedups, marker='s', linewidth=2, markersize=8,
                     label=lbl, color=color)

    ax1.set_xlabel('Thread Count')
    ax1.set_ylabel('Execution Time (seconds) ± σ')
    ax1.set_title('Image Processing: Execution Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(labels, [1, 2, 4, 8], '--', alpha=0.3, color=IDEAL_COLOR, label='Ideal linear')
    ax2.set_xlabel('Thread Count')
    ax2.set_ylabel('Speedup (vs Sequential)')
    ax2.set_title('Image Processing: Speedup')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Image Processing Benchmark (N=15 measurements)', fontsize=14, y=1.02)
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
    algorithms = ["linear_regression", "ransac", "kmeans", "random_forest"]
    algo_labels = ["Linear\nRegression", "RANSAC", "K-Means", "Random\nForest"]

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
    ax1.set_ylabel('Training Time (seconds) ± σ')
    ax1.set_title('ML Algorithm Training Time')
    ax1.set_xticks(x)
    ax1.set_xticklabels(algo_labels)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Threading scaling for linear regression
    thread_modes = ["linear_regression", "linear_regression_threaded_2",
                    "linear_regression_threaded_4", "linear_regression_threaded_8"]
    thread_labels = ["1 (seq)", "2", "4", "8"]

    for variant, color, lbl in [("gil", GIL_COLOR, "With GIL"), ("nogil", NOGIL_COLOR, "Without GIL")]:
        times = [_get_avg(wl, variant, m) for m in thread_modes]
        stds = [_get_std(wl, variant, m) for m in thread_modes]

        if times[0] > 0:
            speedups = [times[0] / t if t > 0 else 0 for t in times]
            ax2.plot(thread_labels, speedups, marker='o', linewidth=2,
                     markersize=8, label=lbl, color=color)

    ax2.plot(thread_labels, [1, 2, 4, 8], '--', alpha=0.3, color=IDEAL_COLOR, label='Ideal linear')
    ax2.set_xlabel('Thread Count')
    ax2.set_ylabel('Speedup (vs Sequential)')
    ax2.set_title('Linear Regression: Threading Scaling')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('ML Training Benchmark (N=15 measurements)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "04_ml_training.png")
    plt.close()
    print("✅ 04_ml_training.png")


# =============================================================================
# Chart 5: Box plot comparison (key modes)
# =============================================================================
def plot_05_boxplot(data):
    """Box plot showing distribution of measurements for key modes."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    plot_configs = [
        ("Data Preprocessing", "sequential", "Data Prep: Sequential"),
        ("Data Preprocessing", "threading_8", "Data Prep: 8 Threads"),
        ("Image Processing", "threading_4", "Image Proc: 4 Threads"),
        ("Image Processing", "threading_8", "Image Proc: 8 Threads"),
        ("ML Training", "linear_regression_threaded_4", "ML: LR 4 Threads"),
        ("ML Training", "linear_regression_threaded_8", "ML: LR 8 Threads"),
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
            labels=['GIL', 'No-GIL'],
            patch_artist=True,
            widths=0.5,
        )
        bp['boxes'][0].set_facecolor(GIL_COLOR)
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor(NOGIL_COLOR)
        bp['boxes'][1].set_alpha(0.7)

        # Welch's t-test (does not assume equal variances between groups)
        t_stat, p_val = stats.ttest_ind(gil_times, nogil_times, equal_var=False)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        speedup = np.mean(gil_times) / np.mean(nogil_times) if np.mean(nogil_times) > 0 else 0

        ax.set_title(f"{title}\nSpeedup: {speedup:.2f}x  Welch p={p_val:.4f} {sig}", fontsize=10)
        ax.set_ylabel('Time (s)')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Distribution of Measurements (N=15 per group)', fontsize=14, y=1.02)
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
                # Streaming results use nested dict with 'mean' key
                avg_lat = d.get("avg_latency", {})
                tp = d.get("throughput", {})
                if isinstance(avg_lat, dict):
                    latencies.append(avg_lat.get("mean", 0) * 1000)  # to ms
                elif isinstance(avg_lat, (int, float)):
                    latencies.append(avg_lat * 1000)
                else:
                    latencies.append(0)

                if isinstance(tp, dict):
                    throughputs.append(tp.get("mean", 0))
                elif isinstance(tp, (int, float)):
                    throughputs.append(tp)
                else:
                    throughputs.append(0)
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
    ax1.set_title('Streaming: Average Event Latency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Worker Count')
    ax2.set_ylabel('Throughput (events/second)')
    ax2.set_title('Streaming: Throughput')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Streaming Benchmark', fontsize=14, y=1.02)
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
    for variant, color, lbl in [("gil", GIL_COLOR, "With GIL"), ("nogil", NOGIL_COLOR, "Without GIL")]:
        merged = wl.get(variant, {}).get("merged", {})
        seq = merged.get("sequential", {})
        if not isinstance(seq, dict):
            continue

        methods = []
        means = []
        stds = []
        for k, v in seq.items():
            if isinstance(v, dict) and "mean" in v:
                methods.append(k.replace("_", "\n"))
                means.append(v["mean"])
                stds.append(v.get("std", 0))

        if methods:
            x = np.arange(len(methods))
            w = 0.35
            offset = -w/2 if variant == "gil" else w/2
            ax1.bar(x + offset, means, w, yerr=stds, capsize=3,
                    label=lbl, color=color, edgecolor='white')
            ax1.set_xticks(x)
            ax1.set_xticklabels(methods, fontsize=8)

    ax1.set_ylabel('Execution Time (seconds) ± σ')
    ax1.set_title('SIMD Sequential Methods')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Threading
    thread_counts = ["2", "4", "8"]
    for variant, color, lbl in [("gil", GIL_COLOR, "With GIL"), ("nogil", NOGIL_COLOR, "Without GIL")]:
        merged = wl.get(variant, {}).get("merged", {})
        threaded = merged.get("threaded", {})
        if not isinstance(threaded, dict):
            continue

        times = []
        for tc in [2, 4, 8]:
            key = f"numpy_threaded_{tc}"
            if key in threaded and isinstance(threaded[key], dict):
                times.append(threaded[key].get("mean", 0))
            else:
                times.append(0)

        if any(t > 0 for t in times):
            ax2.plot(thread_counts, times, marker='o', linewidth=2,
                     markersize=8, label=lbl, color=color)

    ax2.set_xlabel('Thread Count')
    ax2.set_ylabel('Execution Time (seconds)')
    ax2.set_title('NumPy Threaded: GIL vs No-GIL')
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
                gt = g_d["avg_time"]
                nt = n_d["avg_time"]
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

        for key in sorted(g_merged.keys()):
            if key not in n_merged:
                continue
            g_d = g_merged[key]
            n_d = n_merged[key]

            if isinstance(g_d, dict) and "all_times" in g_d and \
               isinstance(n_d, dict) and "all_times" in n_d:

                gt = g_d["all_times"]
                nt = n_d["all_times"]
                g_mean = g_d["avg_time"]
                n_mean = n_d["avg_time"]
                speedup = g_mean / n_mean if n_mean > 0 else 0

                # Welch's t-test (equal_var=False — does not assume equal variances)
                t_stat, p_val = stats.ttest_ind(gt, nt, equal_var=False)

                # Cohen's d (effect size)
                pooled_std = np.sqrt((np.var(gt, ddof=1) + np.var(nt, ddof=1)) / 2)
                cohens_d = (np.mean(gt) - np.mean(nt)) / pooled_std if pooled_std > 0 else 0

                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                n_obs = min(len(gt), len(nt))

                print(f"  {key:<30} {g_mean:>9.4f}s {n_mean:>9.4f}s {speedup:>8.2f}x "
                      f"{n_obs:>4} {t_stat:>8.2f} {p_val:>8.4f} {sig:>5}")

                analysis[f"{name}_{key}"] = {
                    "test": "Welch t-test",
                    "equal_var": False,
                    "gil_mean": float(g_mean),
                    "gil_std": float(g_d.get("std_time", 0)),
                    "nogil_mean": float(n_mean),
                    "nogil_std": float(n_d.get("std_time", 0)),
                    "speedup": float(speedup),
                    "n_samples": n_obs,
                    "t_statistic": float(t_stat),
                    "p_value": float(p_val),
                    "significant": bool(p_val < 0.05),
                    "cohens_d": float(cohens_d),
                    "faster": "No-GIL" if speedup > 1 else "GIL",
                    "ci_95_gil": g_d.get("ci_95", []),
                    "ci_95_nogil": n_d.get("ci_95", []),
                }

    print(f"\n  Significance levels: *** p<0.001, ** p<0.01, * p<0.05")

    # Save
    out = RESULTS_DIR / "statistical_analysis_multirun.json"
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

    run_statistical_analysis(data)

    print(f"\n{'='*70}")
    print(f"All charts saved to: {CHARTS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
