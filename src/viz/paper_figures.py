"""
Figure generation for the research paper.

Produces Matplotlib/Seaborn visualisations for:
    - Average Latency comparison (bar + error bars)
    - Throughput comparison (bar + error bars)
    - Drop rate comparison (bar)
    - Hop count distribution (box plot)
    - Latency CDF (line)
    - Failure resilience (paired bars: healthy vs degraded)
"""

import os
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logging import get_logger

logger = get_logger("viz.paper_figures")

# Auto-numbering
FIGURE_COUNTER = 0

def _next_fig_num() -> int:
    global FIGURE_COUNTER
    FIGURE_COUNTER += 1
    return FIGURE_COUNTER

# Styling config for publication quality
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# Standard colours for routers to maintain consistency
ROUTER_COLORS = {
    "xy": "#1f77b4",          # Blue
    "odd_even": "#ff7f0e",    # Orange
    "valiant": "#2ca02c",     # Green
    "gnn": "#9467bd",         # Purple
    "grl": "#d62728",         # Red
}

ROUTER_LABELS = {
    "xy": "XY",
    "odd_even": "Odd-Even",
    "valiant": "Valiant",
    "gnn": "Supervised GNN",
    "grl": "GRL (Ours)",
}

ROUTER_ORDER = ["xy", "odd_even", "valiant", "gnn", "grl"]


def _save_fig(fig, output_dir: str, basename: str, formats: List[str] = None) -> str:
    """Save figure in multiple formats. Returns PDF path."""
    if formats is None:
        formats = ["pdf", "png"]
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for fmt in formats:
        path = os.path.join(output_dir, f"{basename}.{fmt}")
        fig.savefig(path, format=fmt, dpi=300)
        paths.append(path)
        logger.info(f"Saved figure: {path}")
    plt.close(fig)
    return paths[0] if paths else ""


def _get_ordered_routers(df: pd.DataFrame) -> List[str]:
    """Get routers present in data, in standard order."""
    available = set(df["router"].unique())
    return [r for r in ROUTER_ORDER if r in available]


def plot_latency_comparison(
    summary_df: pd.DataFrame,
    grid_size: int,
    output_dir: str,
    metric: str = "avg_latency_ns_mean",
    y_err: str = "avg_latency_ns_std",
    title: Optional[str] = None,
    suffix: str = "",
) -> Tuple[str, str]:
    """Plot bar chart comparing latency across traffic patterns.

    Returns:
        (figure_path, caption_string)
    """
    df_grid = summary_df[summary_df["grid_size"] == grid_size].copy()
    if df_grid.empty:
        logger.warning(f"No data for grid size {grid_size}")
        return "", ""

    fig_num = _next_fig_num()
    df_grid["Router"] = df_grid["router"].map(ROUTER_LABELS)
    df_grid["Traffic"] = df_grid["traffic_pattern"].str.capitalize()

    traffics = sorted(df_grid["Traffic"].unique())
    routers = _get_ordered_routers(df_grid)

    x = np.arange(len(traffics))
    width = 0.8 / max(len(routers), 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, router in enumerate(routers):
        router_data = df_grid[df_grid["router"] == router]
        means, errs = [], []
        for t in traffics:
            t_data = router_data[router_data["Traffic"] == t]
            if len(t_data) > 0:
                means.append(t_data[metric].iloc[0])
                errs.append(t_data[y_err].iloc[0] if y_err in t_data.columns else 0)
            else:
                means.append(0)
                errs.append(0)

        offset = (i - len(routers) / 2 + 0.5) * width
        ax.bar(
            x + offset, means, width, yerr=errs,
            label=ROUTER_LABELS[router],
            color=ROUTER_COLORS[router],
            capsize=4, edgecolor='black', linewidth=0.5,
        )

    ax.set_ylabel("Average Latency (ns)")
    ax.set_xlabel("Traffic Pattern")
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Fig. {fig_num}: Average Latency ({grid_size}×{grid_size} Torus)")

    ax.set_xticks(x)
    ax.set_xticklabels(traffics)
    ax.legend(title="Routing Algorithm", loc="upper right")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    basename = f"fig{fig_num}_latency_{grid_size}x{grid_size}{suffix}"
    path = _save_fig(fig, output_dir, basename)
    caption = f"Fig. {fig_num}: Average end-to-end latency (±1σ) across traffic patterns on a {grid_size}×{grid_size} 2D torus."
    return path, caption


def plot_throughput_comparison(
    summary_df: pd.DataFrame,
    grid_size: int,
    output_dir: str,
    suffix: str = "",
) -> Tuple[str, str]:
    """Grouped bar chart of throughput_pps_mean ± std per router per traffic."""
    df_grid = summary_df[summary_df["grid_size"] == grid_size].copy()
    if df_grid.empty or "throughput_pps_mean" not in df_grid.columns:
        return "", ""

    fig_num = _next_fig_num()
    df_grid["Router"] = df_grid["router"].map(ROUTER_LABELS)
    df_grid["Traffic"] = df_grid["traffic_pattern"].str.capitalize()

    traffics = sorted(df_grid["Traffic"].unique())
    routers = _get_ordered_routers(df_grid)

    x = np.arange(len(traffics))
    width = 0.8 / max(len(routers), 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, router in enumerate(routers):
        router_data = df_grid[df_grid["router"] == router]
        means, errs = [], []
        for t in traffics:
            t_data = router_data[router_data["Traffic"] == t]
            if len(t_data) > 0:
                means.append(t_data["throughput_pps_mean"].iloc[0] * 1e3)  # Convert to pkts/µs
                errs.append(t_data.get("throughput_pps_std", pd.Series([0])).iloc[0] * 1e3)
            else:
                means.append(0)
                errs.append(0)

        offset = (i - len(routers) / 2 + 0.5) * width
        ax.bar(
            x + offset, means, width, yerr=errs,
            label=ROUTER_LABELS[router],
            color=ROUTER_COLORS[router],
            capsize=4, edgecolor='black', linewidth=0.5,
        )

    ax.set_ylabel("Throughput (packets/µs)")
    ax.set_xlabel("Traffic Pattern")
    ax.set_title(f"Fig. {fig_num}: Throughput ({grid_size}×{grid_size} Torus)")
    ax.set_xticks(x)
    ax.set_xticklabels(traffics)
    ax.legend(title="Routing Algorithm", loc="upper right")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    basename = f"fig{fig_num}_throughput_{grid_size}x{grid_size}{suffix}"
    path = _save_fig(fig, output_dir, basename)
    caption = f"Fig. {fig_num}: Throughput (±1σ) across traffic patterns on a {grid_size}×{grid_size} 2D torus."
    return path, caption


def plot_hop_count_box(
    raw_df: pd.DataFrame,
    grid_size: int,
    output_dir: str,
    suffix: str = "",
) -> Tuple[str, str]:
    """Box-and-whisker plot of avg_hop_count from raw per-seed data."""
    df_grid = raw_df[raw_df["grid_size"] == grid_size].copy()
    if df_grid.empty or "avg_hop_count" not in df_grid.columns:
        return "", ""

    fig_num = _next_fig_num()
    df_grid["Router"] = df_grid["router"].map(ROUTER_LABELS)

    routers = _get_ordered_routers(df_grid)
    router_labels = [ROUTER_LABELS[r] for r in routers]
    palette = [ROUTER_COLORS[r] for r in routers]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=df_grid, x="Router", y="avg_hop_count",
        order=router_labels, palette=palette, ax=ax,
        showmeans=True, meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black"},
    )

    ax.set_ylabel("Average Hop Count")
    ax.set_xlabel("Routing Algorithm")
    ax.set_title(f"Fig. {fig_num}: Hop Count Distribution ({grid_size}×{grid_size} Torus)")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    basename = f"fig{fig_num}_hops_box_{grid_size}x{grid_size}{suffix}"
    path = _save_fig(fig, output_dir, basename)
    caption = f"Fig. {fig_num}: Box plot of average hop count per router across all seeds and traffic patterns ({grid_size}×{grid_size} torus). Diamonds indicate means."
    return path, caption


def plot_latency_cdf(
    raw_df: pd.DataFrame,
    grid_size: int,
    output_dir: str,
    metric: str = "avg_latency_ns",
    suffix: str = "",
) -> Tuple[str, str]:
    """Empirical CDF of latency across seeds."""
    df_grid = raw_df[raw_df["grid_size"] == grid_size].copy()
    if df_grid.empty or metric not in df_grid.columns:
        return "", ""

    fig_num = _next_fig_num()
    routers = _get_ordered_routers(df_grid)

    fig, ax = plt.subplots(figsize=(10, 6))

    for router in routers:
        router_data = df_grid[df_grid["router"] == router][metric].dropna().sort_values()
        if len(router_data) == 0:
            continue
        cdf = np.arange(1, len(router_data) + 1) / len(router_data)
        ax.step(
            router_data.values, cdf,
            label=ROUTER_LABELS[router],
            color=ROUTER_COLORS[router],
            linewidth=2,
        )

    ax.set_xlabel("Average Latency (ns)")
    ax.set_ylabel("CDF")
    ax.set_title(f"Fig. {fig_num}: Empirical CDF of Latency ({grid_size}×{grid_size} Torus)")
    ax.legend(title="Routing Algorithm", loc="lower right")
    ax.grid(linestyle='--', alpha=0.7)

    basename = f"fig{fig_num}_latency_cdf_{grid_size}x{grid_size}{suffix}"
    path = _save_fig(fig, output_dir, basename)
    caption = f"Fig. {fig_num}: Empirical CDF of average latency across all experimental runs ({grid_size}×{grid_size} torus)."
    return path, caption


def plot_failure_resilience(
    summary_df: pd.DataFrame,
    grid_size: int,
    output_dir: str,
    metric: str = "avg_latency_ns_mean",
    suffix: str = "",
) -> Tuple[str, str]:
    """Paired bars: latency under failure_rate=0.0 vs 0.1 side-by-side per router."""
    df_grid = summary_df[summary_df["grid_size"] == grid_size].copy()
    if df_grid.empty:
        return "", ""

    fig_num = _next_fig_num()
    routers = _get_ordered_routers(df_grid)

    # Average across traffic patterns for clean comparison
    failure_rates = sorted(df_grid["failure_rate"].unique())
    if len(failure_rates) < 2:
        return "", ""

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(routers))
    width = 0.35

    for j, fr in enumerate(failure_rates[:2]):
        fr_data = df_grid[df_grid["failure_rate"] == fr]
        means = []
        errs = []
        for router in routers:
            r_data = fr_data[fr_data["router"] == router]
            if len(r_data) > 0:
                means.append(r_data[metric].mean())
                errs.append(r_data[metric].std() if len(r_data) > 1 else 0)
            else:
                means.append(0)
                errs.append(0)

        offset = -width/2 + j * width
        label = f"Failure Rate = {fr:.0%}"
        ax.bar(
            x + offset, means, width, yerr=errs,
            label=label,
            color=[ROUTER_COLORS[r] for r in routers],
            alpha=0.5 + 0.5 * j,  # Lighter for healthy, darker for failed
            capsize=4, edgecolor='black', linewidth=0.5,
            hatch='///' if j == 1 else None,
        )

    ax.set_ylabel("Average Latency (ns)")
    ax.set_xlabel("Routing Algorithm")
    ax.set_title(f"Fig. {fig_num}: Failure Resilience ({grid_size}×{grid_size} Torus)")
    ax.set_xticks(x)
    ax.set_xticklabels([ROUTER_LABELS[r] for r in routers])
    ax.legend(loc="upper left")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    basename = f"fig{fig_num}_failure_resilience_{grid_size}x{grid_size}{suffix}"
    path = _save_fig(fig, output_dir, basename)
    caption = f"Fig. {fig_num}: Average latency under healthy (0%) vs degraded (10%) link failure conditions ({grid_size}×{grid_size} torus). Hatched bars indicate failure scenarios."
    return path, caption


def plot_drop_rate(
    summary_df: pd.DataFrame,
    grid_size: int,
    output_dir: str,
    suffix: str = "",
) -> Tuple[str, str]:
    """Plot bar chart comparing drop rates."""
    df_grid = summary_df[summary_df["grid_size"] == grid_size].copy()
    if df_grid.empty:
        return "", ""

    fig_num = _next_fig_num()
    df_grid["Router"] = df_grid["router"].map(ROUTER_LABELS)
    df_grid["Traffic"] = df_grid["traffic_pattern"].str.capitalize()

    traffics = sorted(df_grid["Traffic"].unique())
    routers = _get_ordered_routers(df_grid)

    x = np.arange(len(traffics))
    width = 0.8 / max(len(routers), 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, router in enumerate(routers):
        router_data = df_grid[df_grid["router"] == router]
        means, errs = [], []
        for t in traffics:
            t_data = router_data[router_data["Traffic"] == t]
            if len(t_data) > 0:
                means.append(t_data["drop_rate_mean"].iloc[0] * 100)
                errs.append(t_data.get("drop_rate_std", pd.Series([0])).iloc[0] * 100)
            else:
                means.append(0)
                errs.append(0)

        offset = (i - len(routers) / 2 + 0.5) * width
        ax.bar(
            x + offset, means, width, yerr=errs,
            label=ROUTER_LABELS[router],
            color=ROUTER_COLORS[router],
            capsize=4, edgecolor='black', linewidth=0.5,
        )

    ax.set_ylabel("Packet Drop Rate (%)")
    ax.set_xlabel("Traffic Pattern")
    ax.set_title(f"Fig. {fig_num}: Packet Loss ({grid_size}×{grid_size} Torus)")
    ax.set_xticks(x)
    ax.set_xticklabels(traffics)
    ax.legend(title="Routing Algorithm", loc="upper right")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    basename = f"fig{fig_num}_drop_rate_{grid_size}x{grid_size}{suffix}"
    path = _save_fig(fig, output_dir, basename)
    caption = f"Fig. {fig_num}: Packet drop rate (±1σ) across traffic patterns on a {grid_size}×{grid_size} 2D torus."
    return path, caption


def generate_all_figures(csv_path: str, output_dir: str, formats: List[str] = None):
    """Load results and generate all paper figures.

    Returns:
        List of (path, caption) tuples for each generated figure.
    """
    global FIGURE_COUNTER
    FIGURE_COUNTER = 0  # Reset counter

    if not os.path.exists(csv_path):
        logger.error(f"Results CSV not found: {csv_path}")
        return []

    df = pd.read_csv(csv_path)

    # Filter out errored rows
    if "error" in df.columns:
        df = df[df["error"].isna()].copy()

    from src.experiments.analysis import compute_summary_stats
    summary = compute_summary_stats(df)

    grid_sizes = sorted(summary["grid_size"].unique())
    figures = []

    for size in grid_sizes:
        size_int = int(size)

        # --- Healthy network figures ---
        df_no_fault = summary[summary["failure_rate"] == 0.0]
        df_raw_no_fault = df[df["failure_rate"] == 0.0] if "failure_rate" in df.columns else df

        if not df_no_fault.empty:
            figures.append(plot_latency_comparison(
                df_no_fault, size_int, output_dir,
                title=f"Average Latency — Normal Traffic ({size_int}×{size_int} Torus)",
            ))
            figures.append(plot_throughput_comparison(
                df_no_fault, size_int, output_dir,
            ))
            figures.append(plot_drop_rate(
                df_no_fault, size_int, output_dir,
            ))

        # --- Box plot from raw data (all conditions) ---
        figures.append(plot_hop_count_box(df, size_int, output_dir))

        # --- CDF from raw data ---
        figures.append(plot_latency_cdf(df, size_int, output_dir))

        # --- Failure resilience (needs both failure rates in summary) ---
        figures.append(plot_failure_resilience(summary, size_int, output_dir))

        # --- Failure scenario figures ---
        df_fault = summary[summary["failure_rate"] > 0.0]
        if not df_fault.empty:
            figures.append(plot_latency_comparison(
                df_fault, size_int, output_dir,
                title=f"Average Latency — 10% Link Failures ({size_int}×{size_int} Torus)",
                suffix="_fault",
            ))
            figures.append(plot_drop_rate(
                df_fault, size_int, output_dir, suffix="_fault",
            ))

    # Filter out empty results
    figures = [(p, c) for p, c in figures if p]

    logger.info(f"Generated {len(figures)} figures total")

    # Print figure index
    print(f"\n  Figure Index:")
    for path, caption in figures:
        size_kb = os.path.getsize(path) / 1024 if os.path.exists(path) else 0
        print(f"    {os.path.basename(path)} ({size_kb:.1f} KB): {caption}")

    return figures

