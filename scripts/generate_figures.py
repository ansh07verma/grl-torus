#!/usr/bin/env python
"""
CLI script for generating research paper figures from experiment results.

Usage:
    python scripts/generate_figures.py --csv results/csv/experiment_results.csv --out results/figures/ --format both
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.viz.paper_figures import generate_all_figures
from src.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Generate paper figures from experiment CSV"
    )
    parser.add_argument(
        "--csv", type=str, default="results/csv/experiment_results.csv",
        help="Path to experiment results CSV"
    )
    parser.add_argument(
        "--out", type=str, default="results/figures/",
        help="Output directory for figures"
    )
    parser.add_argument(
        "--format", type=str, choices=["pdf", "png", "both"], default="both",
        help="Output format (pdf, png, or both)"
    )

    args = parser.parse_args()
    setup_logging()

    formats = ["pdf", "png"] if args.format == "both" else [args.format]

    print(f"\n{'='*60}")
    print(f"  GRL-Torus: Figure Generation")
    print(f"  Input CSV:  {args.csv}")
    print(f"  Output Dir: {args.out}")
    print(f"  Formats:    {formats}")
    print(f"{'='*60}\n")

    generate_all_figures(args.csv, args.out, formats)

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
