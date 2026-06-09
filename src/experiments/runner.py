"""
Experiment runner — grid search across topologies, traffic, routers, and seeds.

Orchestrates the full experiment matrix defined in conf/experiment/ and
produces a consolidated CSV with all metrics for statistical analysis.
"""

from __future__ import annotations

import csv
import itertools
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.routers.base import BaseRouter
from src.routers.xy_router import XYRouter
from src.routers.odd_even_router import OddEvenRouter
from src.routers.valiant_router import ValiantRouter
from src.sim.simulator import TorusSimulator, SimulationResults
from src.sim.torus_graph import TorusGraph
from src.sim.traffic import TrafficGenerator
from src.utils.logging import get_logger
from src.utils.seeding import set_global_seed

logger = get_logger("experiments.runner")


def _create_router(router_type: str, seed: int = 42, **kwargs) -> BaseRouter:
    """Factory function to create a router by type name.

    Args:
        router_type: One of 'xy', 'odd_even', 'valiant', 'gnn', 'grl'.
        seed: Random seed (used by Valiant).

    Returns:
        Router instance.
    """
    if router_type == "xy":
        return XYRouter()
    elif router_type == "odd_even":
        return OddEvenRouter()
    elif router_type == "valiant":
        return ValiantRouter(seed=seed)
    elif router_type == "gnn":
        from src.routers.gnn_router import GNNRouter
        ckpt = kwargs.get("gnn_checkpoint", "results/checkpoints/gnn_best.pt")
        device = kwargs.get("device", "cpu")
        return GNNRouter(checkpoint_path=ckpt, device=device)
    elif router_type == "grl":
        from src.routers.grl_router import GRLRouter
        gnn_ckpt = kwargs.get("gnn_checkpoint", "results/checkpoints/gnn_best.pt")
        dqn_ckpt = kwargs.get("dqn_checkpoint", "results/checkpoints/dqn_best.pt")
        device = kwargs.get("device", "cpu")
        grid_size = kwargs.get("grid_size", 8)
        return GRLRouter(
            gnn_checkpoint_path=gnn_ckpt,
            dqn_checkpoint_path=dqn_ckpt,
            device=device,
            grid_size=grid_size,
        )
    else:
        raise ValueError(f"Unknown router type: {router_type}")


def run_single_experiment(
    grid_size: int,
    router_type: str,
    traffic_pattern: str,
    failure_rate: float,
    seed: int,
    duration_ns: float = 100_000,
    warmup_ns: float = 10_000,
    injection_rate: float = 0.01,
    buffer_depth: int = 64,
    **router_kwargs,
) -> Dict[str, Any]:
    """Run a single simulation experiment and return metrics.

    Args:
        grid_size: Torus N for N×N.
        router_type: Router algorithm name.
        traffic_pattern: Traffic pattern name.
        failure_rate: Link failure injection rate [0, 1].
        seed: Random seed.
        duration_ns: Simulation duration.
        warmup_ns: Warmup period.
        injection_rate: Packet injection rate.
        buffer_depth: Node buffer depth.
        **router_kwargs: Additional router arguments.

    Returns:
        Dict with experiment config + metrics.
    """
    set_global_seed(seed)

    # Build torus
    vc_count = 2 if router_type == "odd_even" else 1
    torus = TorusGraph(
        n=grid_size,
        buffer_depth=buffer_depth,
        vc_count=vc_count,
    )

    # Create router
    router = _create_router(
        router_type, seed=seed,
        grid_size=grid_size,
        **router_kwargs,
    )

    # Create traffic generator
    traffic_gen = TrafficGenerator(
        pattern=traffic_pattern if failure_rate == 0 else "fault",
        grid_size=grid_size,
        injection_rate=injection_rate,
        seed=seed,
        failure_rate=failure_rate,
    )

    # Override traffic pattern for non-fault patterns with failure injection
    if failure_rate > 0 and traffic_pattern != "fault":
        # Inject failures manually, then use the desired traffic pattern
        rng = np.random.default_rng(seed)
        torus.inject_link_failures(failure_rate, rng)
        traffic_gen = TrafficGenerator(
            pattern=traffic_pattern,
            grid_size=grid_size,
            injection_rate=injection_rate,
            seed=seed,
        )

    # Create simulator
    sim = TorusSimulator(
        torus=torus,
        router=router,
        duration_ns=duration_ns,
        warmup_ns=warmup_ns,
    )

    # Run
    results = sim.run(traffic_gen)
    metrics = results.to_dict()

    # Add experiment config to metrics
    metrics.update({
        "grid_size": grid_size,
        "router": router_type,
        "traffic_pattern": traffic_pattern,
        "failure_rate": failure_rate,
        "seed": seed,
        "injection_rate": injection_rate,
        "duration_ns": duration_ns,
    })

    return metrics


def run_experiment_grid(
    topologies: List[int],
    traffic_patterns: List[str],
    routers: List[str],
    failure_rates: List[float],
    seeds: List[int],
    output_csv: str = "results/csv/all_results.csv",
    duration_ns: float = 100_000,
    warmup_ns: float = 10_000,
    injection_rate: float = 0.01,
    **router_kwargs,
) -> List[Dict[str, Any]]:
    """Run a full experiment grid and save results to CSV.

    Args:
        topologies: List of grid sizes.
        traffic_patterns: List of traffic patterns.
        routers: List of router types.
        failure_rates: List of failure rates.
        seeds: List of random seeds.
        output_csv: Path for output CSV file.
        duration_ns: Simulation duration per experiment.
        warmup_ns: Warmup period per experiment.
        injection_rate: Packet injection rate.
        **router_kwargs: Additional router arguments.

    Returns:
        List of all experiment results.
    """
    # Calculate total experiments
    total = len(topologies) * len(traffic_patterns) * len(routers) * len(failure_rates) * len(seeds)
    logger.info(
        f"Running experiment grid: {total} experiments "
        f"({len(topologies)} topo × {len(traffic_patterns)} traffic × "
        f"{len(routers)} routers × {len(failure_rates)} failure × {len(seeds)} seeds)"
    )

    all_results = []
    completed = 0
    start_time = time.time()

    # Ensure output directory exists
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for grid_size, traffic, router_type, failure_rate, seed in itertools.product(
        topologies, traffic_patterns, routers, failure_rates, seeds
    ):
        try:
            # Adjust router kwargs for current grid size
            current_kwargs = dict(router_kwargs)
            if router_type in ("gnn", "grl"):
                # Use grid-specific checkpoints
                ckpt_dir = current_kwargs.get("checkpoint_dir", "results/checkpoints")
                current_kwargs["gnn_checkpoint"] = os.path.join(
                    ckpt_dir, f"gnn_best_{grid_size}x{grid_size}.pt"
                )
                if router_type == "grl":
                    tag = "frozen"
                    current_kwargs["dqn_checkpoint"] = os.path.join(
                        ckpt_dir, f"dqn_best_{tag}_{grid_size}x{grid_size}.pt"
                    )

            metrics = run_single_experiment(
                grid_size=grid_size,
                router_type=router_type,
                traffic_pattern=traffic,
                failure_rate=failure_rate,
                seed=seed,
                duration_ns=duration_ns,
                warmup_ns=warmup_ns,
                injection_rate=injection_rate,
                **current_kwargs,
            )
            all_results.append(metrics)

        except Exception as e:
            logger.error(
                f"FAILED: grid={grid_size}, router={router_type}, "
                f"traffic={traffic}, failure={failure_rate}, seed={seed}: {e}"
            )
            all_results.append({
                "grid_size": grid_size,
                "router": router_type,
                "traffic_pattern": traffic,
                "failure_rate": failure_rate,
                "seed": seed,
                "error": str(e),
            })

        completed += 1
        if completed % 10 == 0 or completed == total:
            elapsed = time.time() - start_time
            eta = elapsed / completed * (total - completed) if completed > 0 else 0
            logger.info(
                f"Progress: {completed}/{total} "
                f"({100*completed/total:.1f}%), "
                f"elapsed={elapsed:.1f}s, ETA={eta:.1f}s"
            )

    # Save to CSV
    if all_results:
        fieldnames = sorted(set().union(*[r.keys() for r in all_results]))
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_results)
        logger.info(f"Results saved to: {output_csv}")

    elapsed_total = time.time() - start_time
    logger.info(f"Experiment grid complete in {elapsed_total:.1f}s")

    return all_results
