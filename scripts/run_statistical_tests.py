#!/usr/bin/env python
"""
Run statistical significance tests on experiment results.

Computes:
    - Mann-Whitney U test (non-parametric) between GRL and each baseline
    - Welch's t-test (parametric complement)
    - Cohen's d effect sizes
    - Summary table saved as CSV

Usage:
    python scripts/run_statistical_tests.py --csv results/csv/experiment_results.csv
"""

import argparse
import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.experiments.analysis import load_results, pairwise_comparison
from src.utils.logging import setup_logging, get_logger

logger = get_logger("stats")


def welch_t_test(group_a: np.ndarray, group_b: np.ndarray):
    """Welch's t-test (unequal variances)."""
    from scipy.stats import ttest_ind
    stat, pval = ttest_ind(group_a, group_b, equal_var=False)
    return stat, pval


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Cohen's d effect size (pooled standard deviation)."""
    n1, n2 = len(group_a), len(group_b)
    s1, s2 = np.std(group_a, ddof=1), np.std(group_b, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group_a) - np.mean(group_b)) / pooled_std


def run_tests(csv_path: str, output_dir: str = "results/tables"):
    """Run full statistical analysis on experiment results."""
    df = load_results(csv_path)

    if "error" in df.columns:
        df = df[df["error"].isna()].copy()

    routers = sorted(df["router"].unique())
    metrics = ["avg_latency_ns", "throughput_pps", "drop_rate", "avg_hop_count"]
    group_cols = ["grid_size", "traffic_pattern", "failure_rate"]

    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    # Test GRL against every other router (and also pairwise among baselines)
    test_pairs = []
    if "grl" in routers:
        for r in routers:
            if r != "grl":
                test_pairs.append(("grl", r))
    # Also compare gnn vs baselines if available
    if "gnn" in routers:
        for r in ["xy", "odd_even", "valiant"]:
            if r in routers:
                test_pairs.append(("gnn", r))

    for metric in metrics:
        if metric not in df.columns:
            continue

        for router_a, router_b in test_pairs:
            # Mann-Whitney U (already in analysis module)
            mw_results = pairwise_comparison(df, router_a, router_b, metric, group_cols)

            # Add Welch's t-test and Cohen's d
            df_a = df[df["router"] == router_a]
            df_b = df[df["router"] == router_b]

            for _, row in mw_results.iterrows():
                config = {c: row[c] for c in group_cols if c in row.index}

                # Get raw values for this config
                mask_a = df_a["router"] == router_a
                mask_b = df_b["router"] == router_b
                for col, val in config.items():
                    mask_a = mask_a & (df_a[col] == val)
                    mask_b = mask_b & (df_b[col] == val)

                vals_a = df_a[mask_a][metric].dropna().values
                vals_b = df_b[mask_b][metric].dropna().values

                # Welch's t-test
                try:
                    t_stat, t_pval = welch_t_test(vals_a, vals_b)
                except Exception:
                    t_stat, t_pval = np.nan, np.nan

                # Cohen's d
                try:
                    d = cohens_d(vals_a, vals_b)
                except Exception:
                    d = np.nan

                result = dict(row)
                result.update({
                    "welch_t_stat": t_stat,
                    "welch_p_value": t_pval,
                    "cohens_d": d,
                    "effect_magnitude": (
                        "large" if abs(d) >= 0.8 else
                        "medium" if abs(d) >= 0.5 else
                        "small" if abs(d) >= 0.2 else
                        "negligible"
                    ) if not np.isnan(d) else "N/A",
                })
                all_results.append(result)

    results_df = pd.DataFrame(all_results)

    # Save full results
    out_path = os.path.join(output_dir, "statistical_tests.csv")
    results_df.to_csv(out_path, index=False)
    logger.info(f"Statistical tests saved to: {out_path}")

    # Print formatted summary
    print(f"\n{'='*80}")
    print(f"  Statistical Significance Tests")
    print(f"{'='*80}")

    for metric in metrics:
        metric_df = results_df[results_df["metric"] == metric]
        if metric_df.empty:
            continue

        print(f"\n  Metric: {metric}")
        print(f"  {'-'*70}")
        print(f"  {'Comparison':<25} {'Config':<25} {'MW p-val':<10} {'Welch p':<10} {'Cohen d':<10} {'Sig?':<5}")
        print(f"  {'-'*70}")

        for _, row in metric_df.iterrows():
            ra = row.get("router_a", "?")
            rb = row.get("router_b", "?")
            config_str = f"{int(row.get('grid_size', 0))}×{int(row.get('grid_size', 0))} {row.get('traffic_pattern', '')} f={row.get('failure_rate', 0)}"
            mw_p = row.get("p_value", np.nan)
            w_p = row.get("welch_p_value", np.nan)
            cd = row.get("cohens_d", np.nan)
            sig = "✓" if row.get("significant", False) else "✗"

            print(
                f"  {ra} vs {rb:<15} {config_str:<25} "
                f"{mw_p:<10.4f} {w_p:<10.4f} {cd:<10.3f} {sig}"
            )

    print(f"\n  Full results: {out_path}")
    print(f"{'='*80}\n")

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Run statistical significance tests on GRL-Torus experiments"
    )
    parser.add_argument(
        "--csv", type=str, default="results/csv/experiment_results.csv",
        help="Path to experiment results CSV"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/tables",
        help="Output directory for tables"
    )
    args = parser.parse_args()

    setup_logging()
    run_tests(args.csv, args.output_dir)


if __name__ == "__main__":
    main()
