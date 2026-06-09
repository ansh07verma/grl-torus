"""
Packet dataclass with full lifecycle tracking.

Each packet tracks its journey through the torus network, recording
creation time, hop path, queue wait times, and delivery status.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Packet:
    """A network packet traversing the 2D torus.

    Attributes:
        id: Unique packet identifier.
        src: Source node coordinates (x, y).
        dst: Destination node coordinates (x, y).
        creation_time: SimPy time when packet was injected.
        payload_size: Payload size in bytes.
        hops: Ordered list of nodes visited (path trace).
        queue_wait_time: Cumulative time spent waiting in queues (ns).
        transmission_time: Cumulative time spent in transmission (ns).
        delivery_time: SimPy time when packet reached destination.
        delivered: Whether the packet successfully reached its destination.
        dropped: Whether the packet was dropped (buffer overflow / timeout).
        drop_reason: Reason for dropping, if applicable.
        consecutive_holds: Number of consecutive Hold actions (for deadlock prevention).
        current_node: The node where the packet currently resides.
        valiant_intermediate: For Valiant routing — the random intermediate node.
        valiant_phase: For Valiant routing — current phase (1 or 2).
    """

    id: int
    src: Tuple[int, int]
    dst: Tuple[int, int]
    creation_time: float
    payload_size: int = 64  # Default 64 bytes

    # Path tracking
    hops: List[Tuple[int, int]] = field(default_factory=list)

    # Timing
    queue_wait_time: float = 0.0
    transmission_time: float = 0.0
    delivery_time: Optional[float] = None

    # Status
    delivered: bool = False
    dropped: bool = False
    drop_reason: Optional[str] = None

    # Deadlock prevention
    consecutive_holds: int = 0

    # Current position
    current_node: Optional[Tuple[int, int]] = None

    # Valiant routing state
    valiant_intermediate: Optional[Tuple[int, int]] = None
    valiant_phase: int = 1

    @property
    def total_latency(self) -> Optional[float]:
        """End-to-end latency in ns. None if not yet delivered."""
        if self.delivery_time is not None:
            return self.delivery_time - self.creation_time
        return None

    @property
    def hop_count(self) -> int:
        """Number of hops taken so far."""
        return len(self.hops)

    @property
    def hops_so_far(self) -> int:
        """Alias for hop_count, used in DQN state vector."""
        return self.hop_count

    def record_hop(self, node: Tuple[int, int]) -> None:
        """Record a hop to the given node."""
        self.hops.append(node)
        self.current_node = node

    def mark_delivered(self, time: float) -> None:
        """Mark packet as successfully delivered."""
        self.delivered = True
        self.delivery_time = time

    def mark_dropped(self, reason: str = "buffer_overflow") -> None:
        """Mark packet as dropped."""
        self.dropped = True
        self.drop_reason = reason

    def __repr__(self) -> str:
        status = "delivered" if self.delivered else ("dropped" if self.dropped else "in-flight")
        return (
            f"Packet(id={self.id}, src={self.src}, dst={self.dst}, "
            f"hops={self.hop_count}, status={status})"
        )
