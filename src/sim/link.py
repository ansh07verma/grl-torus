"""
Link model with utilisation tracking for the 2D torus network.

Each TorusLink represents a directed link between two adjacent nodes,
with configurable bandwidth, propagation delay, and failure probability.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


class TorusLink:
    """A directed link between two adjacent nodes in the 2D torus.

    Attributes:
        src_node: Source node coordinates (x, y).
        dst_node: Destination node coordinates (x, y).
        direction: Direction from src to dst — one of 'N', 'S', 'E', 'W'.
        bandwidth_gbps: Link bandwidth in Gbps.
        propagation_delay_ns: Propagation delay in nanoseconds.
        failure_prob: Probability of link failure (for fault injection).
        is_failed: Whether this link is currently failed.
        is_wraparound: Whether this link is a torus wraparound edge.
    """

    def __init__(
        self,
        src_node: Tuple[int, int],
        dst_node: Tuple[int, int],
        direction: str,
        bandwidth_gbps: float = 100.0,
        propagation_delay_ns: float = 1.0,
        failure_prob: float = 0.0,
        is_failed: bool = False,
        is_wraparound: bool = False,
    ):
        self.src_node = src_node
        self.dst_node = dst_node
        self.direction = direction
        self.bandwidth_gbps = bandwidth_gbps
        self.propagation_delay_ns = propagation_delay_ns
        self.failure_prob = failure_prob
        self.is_failed = is_failed
        self.is_wraparound = is_wraparound

        # Utilisation tracking
        self._bytes_transmitted: int = 0
        self._utilisation_window: float = 0.0  # Time window for utilisation calc
        self._utilisation: float = 0.0  # Current utilisation [0, 1]

        # Statistics
        self.packets_transmitted: int = 0
        self._utilisation_samples: list[float] = []

    @property
    def utilisation(self) -> float:
        """Current link utilisation as a fraction [0, 1]."""
        return self._utilisation

    @property
    def transmission_delay_ns(self) -> float:
        """Time to transmit a 64-byte packet in nanoseconds.

        transmission_time = packet_size_bits / bandwidth_bps
        For 64 bytes at 100 Gbps: 64*8 / 100e9 = 5.12 ns
        """
        packet_bits = 64 * 8  # 64-byte default packet
        bandwidth_bps = self.bandwidth_gbps * 1e9
        return (packet_bits / bandwidth_bps) * 1e9  # Convert seconds to ns

    @property
    def total_delay_ns(self) -> float:
        """Total per-hop delay: transmission + propagation."""
        return self.transmission_delay_ns + self.propagation_delay_ns

    def transmit(self, packet_size_bytes: int = 64) -> float:
        """Record a packet transmission and return the delay.

        Args:
            packet_size_bytes: Size of the packet payload.

        Returns:
            Total delay in nanoseconds (transmission + propagation).
        """
        self.packets_transmitted += 1
        self._bytes_transmitted += packet_size_bytes

        # Update utilisation estimate (simple running average)
        packet_bits = packet_size_bytes * 8
        bandwidth_bps = self.bandwidth_gbps * 1e9
        tx_time_s = packet_bits / bandwidth_bps
        # Approximate utilisation as fraction of time the link is busy
        self._utilisation = min(1.0, self._utilisation * 0.95 + 0.05)
        self._utilisation_samples.append(self._utilisation)

        return self.total_delay_ns

    def update_utilisation(self, current_time: float, window_ns: float = 1000.0) -> None:
        """Update utilisation based on recent activity.

        Args:
            current_time: Current simulation time in ns.
            window_ns: Time window for utilisation averaging.
        """
        if window_ns <= 0:
            return
        # Approximate: packets transmitted * tx_time / window
        if self.packets_transmitted > 0:
            tx_time_per_pkt = self.transmission_delay_ns
            total_busy_time = self.packets_transmitted * tx_time_per_pkt
            self._utilisation = min(1.0, total_busy_time / window_ns)
        else:
            self._utilisation *= 0.95  # Decay when idle

    def get_features(self) -> np.ndarray:
        """Return 3-dimensional edge feature vector for GNN.

        Features:
            [0] utilisation        — current link utilisation [0, 1]
            [1] failure_prob       — failure probability [0, 1]
            [2] propagation_delay  — propagation delay in ns (normalised by 10)
        """
        return np.array(
            [
                self._utilisation,
                self.failure_prob,
                self.propagation_delay_ns / 10.0,  # Normalise to ~[0, 1]
            ],
            dtype=np.float32,
        )

    def fail(self) -> None:
        """Mark this link as failed."""
        self.is_failed = True

    def repair(self) -> None:
        """Repair this link."""
        self.is_failed = False

    def reset(self) -> None:
        """Reset link state for a new simulation run."""
        self._bytes_transmitted = 0
        self._utilisation = 0.0
        self.packets_transmitted = 0
        self._utilisation_samples.clear()

    def __repr__(self) -> str:
        status = "FAILED" if self.is_failed else f"util={self._utilisation:.2f}"
        wrap = " [wrap]" if self.is_wraparound else ""
        return (
            f"TorusLink({self.src_node}->{self.dst_node}, "
            f"dir={self.direction}, {status}{wrap})"
        )
