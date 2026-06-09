"""
Statistical analysis for experiment results.

Computes:
    - Mean ± std across seeds per (topology, router, traffic, failure) config
    - Mann-Whitney U test for pairwise router comparisons
    - Latex tables for the paper
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("experiments.analysis")


def load_results(csv_path: str) -> pd.DataFrame:
    """Load experiment results from CSV."""
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} results from {csv_path}")
    return df


def compute_summary_stats(
    df: pd.DataFrame,
    group_cols: List[str] = ["grid_size", "router", "traffic_pattern", "failure_rate"],
    metric_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute mean ± std per experiment configuration.

    Args:
        df: Raw results DataFrame.
        group_cols: Columns to group by.
        metric_cols: Metric columns to aggregate. Defaults to standard metrics.

    Returns:
        DataFrame with mean and std for each metric per config.
    """
    if metric_cols is None:
        metric_cols = [
            "avg_latency_ns", "p95_latency_ns", "throughput_pps",
            "drop_rate", "mean_utilisation", "avg_hop_count",
        ]

    # Filter to available columns
    metric_cols = [c for c in metric_cols if c in df.columns]
    group_cols = [c for c in group_cols if c in df.columns]

    agg_funcs = {}
    for col in metric_cols:
        agg_funcs[f"{col}_mean"] = (col, "mean")
        agg_funcs[f"{col}_std"] = (col, "std")
        agg_funcs[f"{col}_median"] = (col, "median")

    summary = df.groupby(group_cols).agg(**agg_funcs).reset_index()
    summary["n_seeds"] = df.groupby(group_cols).size().values

    return summary


def pairwise_comparison(
    df: pd.DataFrame,
    router_a: str,
    router_b: str,
    metric: str = "avg_latency_ns",
    group_cols: List[str] = ["grid_size", "traffic_pattern", "failure_rate"],
) -> pd.DataFrame:
    """Mann-Whitney U test comparing two routers across configs.

    Args:
        df: Raw results DataFrame.
        router_a: First router name.
        router_b: Second router name.
        metric: Metric to compare.
        group_cols: Config columns to group by.

    Returns:
        DataFrame with U-statistic, p-value, and effect size per config.
    """
    from scipy.stats import mannwhitneyu

    results = []

    df_a = df[df["router"] == router_a]
    df_b = df[df["router"] == router_b]

    for name, group_a in df_a.groupby(group_cols):
        if isinstance(name, tuple):
            config = dict(zip(group_cols, name))
        else:
            config = {group_cols[0]: name}

        # Find matching config in router_b
        mask = True
        for col, val in config.items():
            mask = mask & (df_b[col] == val)
        group_b = df_b[mask]

        if len(group_a) < 2 or len(group_b) < 2:
            continue

        vals_a = group_a[metric].dropna().values
        vals_b = group_b[metric].dropna().values

        if len(vals_a) == 0 or len(vals_b) == 0:
            continue

        try:
            u_stat, p_value = mannwhitneyu(vals_a, vals_b, alternative="two-sided")
            # Effect size (rank-biserial correlation)
            n1, n2 = len(vals_a), len(vals_b)
            effect_size = 1 - (2 * u_stat) / (n1 * n2)
        except ValueError:
            u_stat, p_value, effect_size = np.nan, np.nan, np.nan

        result = dict(config)
        result.update({
            "router_a": router_a,
            "router_b": router_b,
            "metric": metric,
            f"{router_a}_mean": np.mean(vals_a),
            f"{router_b}_mean": np.mean(vals_b),
            "u_statistic": u_stat,
            "p_value": p_value,
            "effect_size": effect_size,
            "significant": p_value < 0.05 if not np.isnan(p_value) else False,
        })
        results.append(result)

    return pd.DataFrame(results)


def compute_improvement_table(
    df: pd.DataFrame,
    baseline_router: str = "xy",
    metric: str = "avg_latency_ns",
    lower_is_better: bool = True,
) -> pd.DataFrame:
    """Compute percentage improvement of each router over a baseline.

    Args:
        df: Summary stats DataFrame (with _mean columns).
        baseline_router: Router to compare against.
        metric: Metric to compare.
        lower_is_better: If True, negative changes are improvements.

    Returns:
        DataFrame with improvement percentages.
    """
    metric_mean = f"{metric}_mean"
    if metric_mean not in df.columns:
        # Try using raw metric
        metric_mean = metric

    group_cols = [c for c in ["grid_size", "traffic_pattern", "failure_rate"] if c in df.columns]

    results = []
    baseline_df = df[df["router"] == baseline_router]

    for _, baseline_row in baseline_df.iterrows():
        config = {c: baseline_row[c] for c in group_cols}
        baseline_val = baseline_row[metric_mean]

        if pd.isna(baseline_val) or baseline_val == 0:
            continue

        for router in df["router"].unique():
            if router == baseline_router:
                continue

            mask = df["router"] == router
            for col, val in config.items():
                mask = mask & (df[col] == val)
            router_row = df[mask]

            if len(router_row) == 0:
                continue

            router_val = router_row[metric_mean].iloc[0]
            if pd.isna(router_val):
                continue

            if lower_is_better:
                pct_improvement = (baseline_val - router_val) / baseline_val * 100
            else:
                pct_improvement = (router_val - baseline_val) / baseline_val * 100

            result = dict(config)
            result.update({
                "router": router,
                "baseline": baseline_router,
                f"baseline_{metric}": baseline_val,
                f"router_{metric}": router_val,
                "improvement_pct": pct_improvement,
            })
            results.append(result)

    return pd.DataFrame(results)


def generate_latex_table(
    summary_df: pd.DataFrame,
    metric: str = "avg_latency_ns",
    caption: str = "Average Latency (ns)",
    label: str = "tab:latency",
) -> str:
    """Generate a LaTeX table from summary statistics.

    Returns:
        LaTeX table string.
    """
    metric_mean = f"{metric}_mean"
    metric_std = f"{metric}_std"

    if metric_mean not in summary_df.columns:
        return f"% Metric '{metric}' not found in summary"

    routers = summary_df["router"].unique()
    configs = summary_df.groupby(["grid_size", "traffic_pattern"]).first().index

    # Build table
    header = " & ".join(["Config"] + [r.replace("_", "\\_") for r in routers])
    lines = [
        f"\\begin{{table}}[ht]",
        f"\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{l{'c' * len(routers)}}}",
        f"\\toprule",
        f"{header} \\\\",
        f"\\midrule",
    ]

    for grid_size, traffic in configs:
        row = f"{grid_size}×{grid_size}, {traffic}"
        for router in routers:
            mask = (
                (summary_df["grid_size"] == grid_size)
                & (summary_df["traffic_pattern"] == traffic)
                & (summary_df["router"] == router)
            )
            subset = summary_df[mask]
            if len(subset) > 0:
                mean = subset[metric_mean].iloc[0]
                std = subset[metric_std].iloc[0]
                row += f" & {mean:.2f} $\\pm$ {std:.2f}"
            else:
                row += " & ---"
        row += " \\\\"
        lines.append(row)

    lines.extend([
        f"\\bottomrule",
        f"\\end{{tabular}}",
        f"\\end{{table}}",
    ])

    return "\n".join(lines)
