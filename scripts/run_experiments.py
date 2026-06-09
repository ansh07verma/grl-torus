#!/usr/bin/env python
"""
CLI script for running experiment grids.

Usage:
    # Baseline-only grid (no ML models needed)
    python scripts/run_experiments.py --routers xy odd_even valiant --topologies 4 8

    # Full grid with GRL
    python scripts/run_experiments.py --routers xy odd_even valiant gnn grl --topologies 4 8
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.experiments.runner import run_experiment_grid
from src.experiments.analysis import load_results, compute_summary_stats, compute_improvement_table
from src.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Run GRL-Torus experiment grid"
    )
    parser.add_argument(
        "--topologies", type=int, nargs="+", default=[4, 8],
        help="Torus grid sizes (default: 4 8)"
    )
    parser.add_argument(
        "--traffic", type=str, nargs="+",
        default=["uniform", "hotspot", "adversarial"],
        help="Traffic patterns"
    )
    parser.add_argument(
        "--routers", type=str, nargs="+",
        default=["xy", "odd_even", "valiant", "gnn", "grl"],
        help="Router types (default: all 5)"
    )
    parser.add_argument(
        "--failure-rates", type=float, nargs="+",
        default=[0.0, 0.1],
        help="Link failure rates"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Random seeds"
    )
    parser.add_argument(
        "--n-runs", type=int, default=None,
        help="Number of runs (convenience: generates seeds 0..n-1, overrides --seeds)"
    )
    parser.add_argument(
        "--duration", type=float, default=1000000,
        help="Simulation duration in ns (default: 1,000,000 = 1ms)"
    )
    parser.add_argument(
        "--warmup", type=float, default=100000,
        help="Warmup period in ns (default: 100,000 = 100µs)"
    )
    parser.add_argument(
        "--injection-rate", type=float, default=0.01,
        help="Packet injection rate"
    )
    parser.add_argument(
        "--output-csv", type=str, default="results/csv/experiment_results.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="results/checkpoints",
        help="Model checkpoint directory"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for ML routers"
    )

    args = parser.parse_args()

    # --n-runs overrides --seeds
    if args.n_runs is not None:
        args.seeds = list(range(args.n_runs))

    setup_logging()

    total = (
        len(args.topologies) * len(args.traffic) * len(args.routers)
        * len(args.failure_rates) * len(args.seeds)
    )

    print(f"\n{'='*60}")
    print(f"  GRL-Torus: Experiment Grid")
    print(f"  Topologies: {args.topologies}")
    print(f"  Traffic:    {args.traffic}")
    print(f"  Routers:    {args.routers}")
    print(f"  Failures:   {args.failure_rates}")
    print(f"  Seeds:      {args.seeds}")
    print(f"  Total:      {total} experiments")
    print(f"{'='*60}\n")

    results = run_experiment_grid(
        topologies=args.topologies,
        traffic_patterns=args.traffic,
        routers=args.routers,
        failure_rates=args.failure_rates,
        seeds=args.seeds,
        output_csv=args.output_csv,
        duration_ns=args.duration,
        warmup_ns=args.warmup,
        injection_rate=args.injection_rate,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Results Summary")
    print(f"{'='*60}")

    df = load_results(args.output_csv)
    summary = compute_summary_stats(df)

    for router in args.routers:
        router_df = summary[summary["router"] == router]
        if len(router_df) == 0:
            continue
        print(f"\n  {router}:")
        for _, row in router_df.iterrows():
            latency = row.get("avg_latency_ns_mean", float("nan"))
            drop = row.get("drop_rate_mean", float("nan"))
            print(
                f"    {int(row['grid_size'])}×{int(row['grid_size'])} "
                f"{row.get('traffic_pattern', '?')}: "
                f"latency={latency:.1f}ns, drop={drop:.3f}"
            )

    # Improvement table vs XY
    if "xy" in args.routers:
        print(f"\n  Improvement over XY (avg latency):")
        improvement = compute_improvement_table(summary, baseline_router="xy")
        for _, row in improvement.iterrows():
            print(
                f"    {row['router']}: {row['improvement_pct']:+.1f}% "
                f"({int(row.get('grid_size', 0))}×{int(row.get('grid_size', 0))} "
                f"{row.get('traffic_pattern', '')})"
            )

    print(f"\n  CSV saved to: {args.output_csv}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
