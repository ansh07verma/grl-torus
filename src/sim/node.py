"""
Node model with directional queues for the 2D torus network.

Each TorusNode maintains four directional queues (N/S/E/W) with configurable
max buffer depth. Queue occupancy is observable by the GNN at each timestep.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from src.sim.packet import Packet


class NodeQueue:
    """Fixed-capacity FIFO queue for one directional output port.

    Attributes:
        direction: Queue direction — one of 'N', 'S', 'E', 'W'.
        max_depth: Maximum number of packets the queue can hold.
        vc_count: Number of virtual channels (for Odd-Even routing).
    """

    def __init__(self, direction: str, max_depth: int = 64, vc_count: int = 1):
        self.direction = direction
        self.max_depth = max_depth
        self.vc_count = vc_count

        # One deque per virtual channel
        self._queues: List[deque] = [deque(maxlen=max_depth) for _ in range(vc_count)]

        # Occupancy tracking over time
        self._occupancy_history: List[int] = []

    @property
    def occupancy(self) -> int:
        """Total number of packets across all virtual channels."""
        return sum(len(q) for q in self._queues)

    @property
    def is_full(self) -> bool:
        """Whether all virtual channels are full."""
        return all(len(q) >= self.max_depth for q in self._queues)

    def has_space(self, vc: int = 0) -> bool:
        """Check if a specific virtual channel has space."""
        if vc >= self.vc_count:
            return False
        return len(self._queues[vc]) < self.max_depth

    def enqueue(self, packet: "Packet", vc: int = 0) -> bool:
        """Add a packet to the specified virtual channel.

        Returns:
            True if packet was enqueued, False if queue was full.
        """
        if vc >= self.vc_count:
            return False
        if len(self._queues[vc]) >= self.max_depth:
            return False
        self._queues[vc].append(packet)
        self._occupancy_history.append(self.occupancy)
        return True

    def dequeue(self, vc: int = 0) -> Optional["Packet"]:
        """Remove and return the head packet from the specified virtual channel.

        Returns:
            The dequeued Packet, or None if the queue is empty.
        """
        if vc >= self.vc_count or len(self._queues[vc]) == 0:
            return None
        pkt = self._queues[vc].popleft()
        self._occupancy_history.append(self.occupancy)
        return pkt

    def peek(self, vc: int = 0) -> Optional["Packet"]:
        """Peek at the head packet without removing it."""
        if vc >= self.vc_count or len(self._queues[vc]) == 0:
            return None
        return self._queues[vc][0]

    def get_vc_occupancy(self, vc: int) -> int:
        """Get occupancy for a specific virtual channel."""
        if vc >= self.vc_count:
            return 0
        return len(self._queues[vc])

    def clear(self) -> None:
        """Clear all virtual channels."""
        for q in self._queues:
            q.clear()
        self._occupancy_history.clear()

    def __len__(self) -> int:
        return self.occupancy

    def __repr__(self) -> str:
        return f"NodeQueue(dir={self.direction}, occ={self.occupancy}/{self.max_depth})"


# Direction constants
DIRECTIONS = ("N", "S", "E", "W")


class TorusNode:
    """A node in the 2D torus network.

    Each node has coordinates (x, y) and maintains four directional output
    queues. It provides an 8-dimensional feature vector for the GNN:
    [queue_N, queue_S, queue_E, queue_W, x_norm, y_norm, load, is_failed]

    Attributes:
        x: X-coordinate in the torus grid.
        y: Y-coordinate in the torus grid.
        grid_size: Size N of the N×N torus (for coordinate normalisation).
        is_failed: Whether this node has failed.
        buffer_depth: Max queue depth per direction.
        vc_count: Virtual channels per direction (2 for Odd-Even).
    """

    def __init__(
        self,
        x: int,
        y: int,
        grid_size: int,
        buffer_depth: int = 64,
        vc_count: int = 1,
        is_failed: bool = False,
    ):
        self.x = x
        self.y = y
        self.grid_size = grid_size
        self.is_failed = is_failed
        self.buffer_depth = buffer_depth
        self.vc_count = vc_count

        # Directional output queues
        self.queues: Dict[str, NodeQueue] = {
            d: NodeQueue(direction=d, max_depth=buffer_depth, vc_count=vc_count)
            for d in DIRECTIONS
        }

        # Load tracking (exponential moving average of total queue occupancy)
        self._load_ema: float = 0.0
        self._load_alpha: float = 0.1  # EMA smoothing factor

        # Statistics
        self.packets_processed: int = 0
        self.packets_generated: int = 0
        self.packets_dropped: int = 0

    @property
    def coords(self) -> Tuple[int, int]:
        """Return (x, y) coordinates."""
        return (self.x, self.y)

    @property
    def total_queue_occupancy(self) -> int:
        """Total packets across all directional queues."""
        return sum(q.occupancy for q in self.queues.values())

    @property
    def load(self) -> float:
        """Exponential moving average of queue occupancy."""
        return self._load_ema

    def update_load(self) -> None:
        """Update the load EMA based on current queue occupancy."""
        current = self.total_queue_occupancy / (4 * self.buffer_depth)  # Normalise to [0, 1]
        self._load_ema = (
            self._load_alpha * current + (1 - self._load_alpha) * self._load_ema
        )

    def get_features(self) -> np.ndarray:
        """Return 8-dimensional node feature vector for GNN.

        Features:
            [0] queue_N — normalised North queue occupancy
            [1] queue_S — normalised South queue occupancy
            [2] queue_E — normalised East queue occupancy
            [3] queue_W — normalised West queue occupancy
            [4] x_norm  — normalised x-coordinate [0, 1]
            [5] y_norm  — normalised y-coordinate [0, 1]
            [6] load    — EMA of total queue load [0, 1]
            [7] is_failed — 1.0 if node is failed, 0.0 otherwise
        """
        max_q = max(self.buffer_depth, 1)
        return np.array(
            [
                self.queues["N"].occupancy / max_q,
                self.queues["S"].occupancy / max_q,
                self.queues["E"].occupancy / max_q,
                self.queues["W"].occupancy / max_q,
                self.x / max(self.grid_size - 1, 1),
                self.y / max(self.grid_size - 1, 1),
                self._load_ema,
                float(self.is_failed),
            ],
            dtype=np.float32,
        )

    def get_queue(self, direction: str) -> NodeQueue:
        """Get the queue for a specific direction."""
        return self.queues[direction]

    def reset(self) -> None:
        """Reset node state for a new simulation run."""
        for q in self.queues.values():
            q.clear()
        self._load_ema = 0.0
        self.packets_processed = 0
        self.packets_generated = 0
        self.packets_dropped = 0

    def __repr__(self) -> str:
        occ = self.total_queue_occupancy
        return f"TorusNode(({self.x},{self.y}), occ={occ}, failed={self.is_failed})"
