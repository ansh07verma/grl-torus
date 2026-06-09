#!/usr/bin/env python
"""
CLI script for training the supervised GNN routing model.

Usage:
    python scripts/train_gnn.py --grid-size 4 --epochs 50
    python scripts/train_gnn.py --grid-size 8 --num-snapshots 500 --device cpu
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.training import train_gnn_supervised
from src.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Train supervised GNN routing model for 2D torus"
    )
    parser.add_argument(
        "--grid-size", type=int, default=4,
        help="Torus grid size N for N×N (default: 4)"
    )
    parser.add_argument(
        "--num-snapshots", type=int, default=200,
        help="Number of training data snapshots (default: 200)"
    )
    parser.add_argument(
        "--samples-per-snapshot", type=int, default=20,
        help="Routing samples per snapshot (default: 20)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Maximum training epochs (default: 100)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--patience", type=int, default=10,
        help="Early stopping patience (default: 10)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="Compute device (default: cpu)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="results/checkpoints",
        help="Directory for model checkpoints"
    )
    parser.add_argument(
        "--log-dir", type=str, default="results/logs",
        help="TensorBoard log directory"
    )

    args = parser.parse_args()

    setup_logging()

    print(f"\n{'='*60}")
    print(f"  GRL-Torus: Supervised GNN Training")
    print(f"  Grid: {args.grid_size}x{args.grid_size}")
    print(f"  Snapshots: {args.num_snapshots}")
    print(f"  Device: {args.device}")
    print(f"{'='*60}\n")

    results = train_gnn_supervised(
        grid_size=args.grid_size,
        num_snapshots=args.num_snapshots,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        samples_per_snapshot=args.samples_per_snapshot,
    )

    print(f"\n{'='*60}")
    print(f"  Training Results")
    print(f"{'='*60}")
    print(f"  Best Val Loss:  {results['best_val_loss']:.4f}")
    print(f"  Best Val Acc:   {results['best_val_acc']:.4f}")
    print(f"  Test Accuracy:  {results['test_acc']:.4f}")
    print(f"  Epochs Trained: {results['epochs_trained']}")
    print(f"  Total Params:   {results['total_params']:,}")
    print(f"  Training Time:  {results['training_time_s']:.1f}s")
    print(f"  Checkpoint:     {results['checkpoint_path']}")
    print(f"{'='*60}\n")

    # Save results summary
    summary_path = os.path.join(args.checkpoint_dir, f"gnn_training_summary_{args.grid_size}x{args.grid_size}.json")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    summary = {k: v for k, v in results.items() if k != "history"}
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
