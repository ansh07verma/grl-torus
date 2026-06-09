"""
Traffic generator for 2D torus network simulation.

Supports four traffic patterns with Poisson arrival process:
    1. Uniform Random — src/dst drawn uniformly
    2. Hotspot — 80% of traffic targets 20% of nodes
    3. Adversarial — traffic concentrated along central rows/columns
    4. Fault — uniform traffic with link/node failures injected
"""

from __future__ import annotations

import random
from typing import Any, Generator, List, Optional, Tuple

import numpy as np
import simpy

from src.sim.packet import Packet
from src.utils.logging import get_logger

logger = get_logger("sim.traffic")


class TrafficGenerator:
    """Configurable packet injection engine for the torus simulator.

    Args:
        pattern: Traffic pattern — 'uniform', 'hotspot', 'adversarial', or 'fault'.
        grid_size: N for the N×N torus.
        injection_rate: Mean packet injection rate (packets per ns).
        seed: Random seed for reproducibility.
        hotspot_ratio: Fraction of nodes that are hotspots (default 0.2).
        hotspot_traffic_frac: Fraction of traffic targeting hotspots (default 0.8).
        failure_rate: For 'fault' pattern — fraction of links to fail.
        failure_type: 'link' or 'node' failures.
    """

    def __init__(
        self,
        pattern: str,
        grid_size: int,
        injection_rate: float = 0.01,
        seed: int = 42,
        hotspot_ratio: float = 0.2,
        hotspot_traffic_frac: float = 0.8,
        failure_rate: float = 0.1,
        failure_type: str = "link",
    ):
        self.pattern = pattern.lower()
        self.grid_size = grid_size
        self.injection_rate = injection_rate
        self.seed = seed
        self.hotspot_ratio = hotspot_ratio
        self.hotspot_traffic_frac = hotspot_traffic_frac
        self.failure_rate = failure_rate
        self.failure_type = failure_type

        self._rng = np.random.default_rng(seed)
        self._packet_counter = 0

        # All node coordinates
        self._all_nodes = [
            (x, y) for x in range(grid_size) for y in range(grid_size)
        ]

        # Pre-compute hotspot nodes
        n_hotspots = max(1, int(len(self._all_nodes) * hotspot_ratio))
        hotspot_indices = self._rng.choice(
            len(self._all_nodes), size=n_hotspots, replace=False
        )
        self._hotspot_nodes = [self._all_nodes[i] for i in hotspot_indices]
        self._non_hotspot_nodes = [
            n for n in self._all_nodes if n not in self._hotspot_nodes
        ]

        # Pre-compute adversarial sources/destinations (central rows/columns)
        center = grid_size // 2
        quarter = max(1, grid_size // 4)
        self._adversarial_srcs = [
            (x, y)
            for x in range(grid_size)
            for y in range(grid_size)
            if center - quarter <= x <= center + quarter
            or center - quarter <= y <= center + quarter
        ]

        logger.info(
            f"TrafficGenerator: pattern={pattern}, rate={injection_rate}, "
            f"grid={grid_size}x{grid_size}, seed={seed}"
        )

    def _generate_pair_uniform(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Generate a random (src, dst) pair, uniform distribution."""
        while True:
            src_idx = self._rng.integers(0, len(self._all_nodes))
            dst_idx = self._rng.integers(0, len(self._all_nodes))
            if src_idx != dst_idx:
                return self._all_nodes[src_idx], self._all_nodes[dst_idx]

    def _generate_pair_hotspot(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Generate (src, dst) with hotspot bias — 80% traffic to 20% nodes."""
        # Source is uniform
        src_idx = self._rng.integers(0, len(self._all_nodes))
        src = self._all_nodes[src_idx]

        # Destination biased toward hotspots
        if self._rng.random() < self.hotspot_traffic_frac:
            dst_idx = self._rng.integers(0, len(self._hotspot_nodes))
            dst = self._hotspot_nodes[dst_idx]
        else:
            dst_idx = self._rng.integers(0, len(self._all_nodes))
            dst = self._all_nodes[dst_idx]

        # Ensure src != dst
        while dst == src:
            dst_idx = self._rng.integers(0, len(self._all_nodes))
            dst = self._all_nodes[dst_idx]

        return src, dst

    def _generate_pair_adversarial(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Generate (src, dst) concentrated along central rows/columns.

        This is worst-case for XY routing as central links become bottlenecks.
        """
        # Sources from edge regions, destinations in center
        n = self.grid_size
        src_idx = self._rng.integers(0, len(self._all_nodes))
        src = self._all_nodes[src_idx]

        # Destination biased toward center
        if self._rng.random() < 0.7:  # 70% of traffic targets central region
            dst_idx = self._rng.integers(0, len(self._adversarial_srcs))
            dst = self._adversarial_srcs[dst_idx]
        else:
            dst_idx = self._rng.integers(0, len(self._all_nodes))
            dst = self._all_nodes[dst_idx]

        while dst == src:
            dst_idx = self._rng.integers(0, len(self._all_nodes))
            dst = self._all_nodes[dst_idx]

        return src, dst

    def _generate_pair(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Generate a (src, dst) pair based on the configured pattern."""
        if self.pattern == "uniform" or self.pattern == "fault":
            return self._generate_pair_uniform()
        elif self.pattern == "hotspot":
            return self._generate_pair_hotspot()
        elif self.pattern == "adversarial":
            return self._generate_pair_adversarial()
        else:
            raise ValueError(f"Unknown traffic pattern: {self.pattern}")

    def _inter_arrival_time(self) -> float:
        """Generate Poisson inter-arrival time (exponential distribution)."""
        if self.injection_rate <= 0:
            return float("inf")
        return self._rng.exponential(1.0 / self.injection_rate)

    def generate(self, env: simpy.Environment) -> Generator:
        """SimPy process that generates packets at Poisson-distributed intervals.

        Yields:
            SimPy timeout events between packet creations.
        """
        while True:
            # Wait for next packet arrival
            inter_arrival = self._inter_arrival_time()
            yield env.timeout(inter_arrival)

            # Create packet
            src, dst = self._generate_pair()
            self._packet_counter += 1

            packet = Packet(
                id=self._packet_counter,
                src=src,
                dst=dst,
                creation_time=env.now,
            )

            yield packet  # Yield the packet to the simulator

    def generate_batch(
        self, env: simpy.Environment, callback: Any
    ) -> Generator:
        """SimPy process that generates packets and passes them to a callback.

        Args:
            env: SimPy environment.
            callback: Function to call with each generated packet.
        """
        while True:
            inter_arrival = self._inter_arrival_time()
            yield env.timeout(inter_arrival)

            src, dst = self._generate_pair()
            self._packet_counter += 1

            packet = Packet(
                id=self._packet_counter,
                src=src,
                dst=dst,
                creation_time=env.now,
            )

            callback(packet)

    def inject_failures(self, torus: Any) -> int:
        """Inject failures into the torus for the 'fault' traffic pattern.

        Args:
            torus: TorusGraph instance.

        Returns:
            Number of failures injected.
        """
        if self.pattern != "fault":
            return 0

        if self.failure_type == "node":
            return torus.inject_node_failures(self.failure_rate, self._rng)
        else:
            return torus.inject_link_failures(self.failure_rate, self._rng)

    def reset(self) -> None:
        """Reset generator state for a new run."""
        self._packet_counter = 0
        self._rng = np.random.default_rng(self.seed)

    def __repr__(self) -> str:
        return (
            f"TrafficGenerator(pattern={self.pattern}, rate={self.injection_rate}, "
            f"grid={self.grid_size}x{self.grid_size})"
        )
