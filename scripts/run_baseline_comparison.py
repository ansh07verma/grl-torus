#!/usr/bin/env python
"""
Run a full baseline comparison grid and generate tables/figures.

Usage:
    python scripts/run_baseline_comparison.py
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.experiments.runner import run_experiment_grid
from src.experiments.analysis import load_results, compute_summary_stats
from src.viz.comparison_table import generate_latex_table, generate_html_table
from src.viz.paper_figures import generate_all_figures
from src.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Run full 4x4 comparison grid")
    parser.add_argument("--csv", type=str, default="results/csv/comparison_results.csv")
    parser.add_argument("--duration", type=float, default=1000000)
    parser.add_argument("--warmup", type=float, default=100000)
    args = parser.parse_args()

    setup_logging()

    # 1. Run Experiments
    run_experiment_grid(
        topologies=[4],
        traffic_patterns=["uniform", "hotspot", "adversarial"],
        routers=["xy", "odd_even", "valiant", "gnn", "grl"],
        failure_rates=[0.0, 0.1],
        seeds=[0, 1, 2, 3, 4],
        output_csv=args.csv,
        duration_ns=args.duration,
        warmup_ns=args.warmup,
    )

    # 2. Generate Tables
    df = load_results(args.csv)
    summary = compute_summary_stats(df)
    
    latex_path = "results/tables/comparison_table.tex"
    html_path = "results/tables/comparison_table.html"
    
    generate_latex_table(summary, latex_path)
    generate_html_table(summary, html_path)
    
    # 3. Generate Figures
    generate_all_figures(args.csv, "results/figures/", formats=["pdf", "png"])
    
    print("\nComparison run complete!")
    print(f"  LaTeX table: {latex_path}")
    print(f"  HTML table:  {html_path}")


if __name__ == "__main__":
    main()
