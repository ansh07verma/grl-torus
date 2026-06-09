#!/usr/bin/env python
"""
CLI script for training the DQN routing policy.

Usage:
    # Frozen GNN mode (default)
    python scripts/train_dqn.py --grid-size 4 --episodes 200 --gnn-checkpoint results/checkpoints/gnn_best_4x4.pt

    # Joint training mode
    python scripts/train_dqn.py --grid-size 4 --episodes 200 --joint --gnn-checkpoint results/checkpoints/gnn_best_4x4.pt
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.training import train_dqn
from src.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Train DQN routing policy for 2D torus"
    )
    parser.add_argument(
        "--grid-size", type=int, default=4,
        help="Torus grid size N for N×N (default: 4)"
    )
    parser.add_argument(
        "--gnn-checkpoint", type=str, default=None,
        help="Path to pretrained GNN checkpoint"
    )
    parser.add_argument(
        "--joint", action="store_true",
        help="Joint training mode (train GNN + DQN together)"
    )
    parser.add_argument(
        "--episodes", type=int, default=500,
        help="Number of training episodes (default: 500)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="DQN learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99,
        help="Discount factor (default: 0.99)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Mini-batch size (default: 64)"
    )
    parser.add_argument(
        "--eps-decay-steps", type=int, default=5000,
        help="Epsilon decay steps (default: 5000)"
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

    mode = "Joint Training" if args.joint else "Frozen GNN"

    print(f"\n{'='*60}")
    print(f"  GRL-Torus: DQN Training ({mode})")
    print(f"  Grid: {args.grid_size}x{args.grid_size}")
    print(f"  Episodes: {args.episodes}")
    print(f"  GNN: {args.gnn_checkpoint or 'random init'}")
    print(f"  Device: {args.device}")
    print(f"{'='*60}\n")

    results = train_dqn(
        grid_size=args.grid_size,
        gnn_checkpoint_path=args.gnn_checkpoint,
        freeze_gnn=not args.joint,
        episodes=args.episodes,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        eps_decay_steps=args.eps_decay_steps,
        device=args.device,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )

    print(f"\n{'='*60}")
    print(f"  DQN Training Results ({mode})")
    print(f"{'='*60}")
    print(f"  Best Avg Reward:  {results['best_avg_reward']:.2f}")
    print(f"  Total Episodes:   {results['total_episodes']}")
    print(f"  Total Steps:      {results['total_steps']:,}")
    print(f"  Training Time:    {results['training_time_s']:.1f}s")
    print(f"  Checkpoint:       {results['checkpoint_path']}")
    print(f"{'='*60}\n")

    # Save summary
    tag = "joint" if args.joint else "frozen"
    summary_path = os.path.join(
        args.checkpoint_dir,
        f"dqn_training_summary_{tag}_{args.grid_size}x{args.grid_size}.json"
    )
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    summary = {k: v for k, v in results.items() if k != "history"}
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
