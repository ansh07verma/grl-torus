"""
Abstract base class for all routing algorithms.

All routers (XY, Odd-Even, Valiant, GNN, GRL) implement this interface
to ensure interchangeability in the simulator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from src.sim.packet import Packet


class BaseRouter(ABC):
    """Abstract router interface.

    Every routing algorithm must implement route() and name.
    The simulator calls route() at every hop for every packet.
    """

    @abstractmethod
    def route(
        self,
        packet: Packet,
        current_node: Tuple[int, int],
        graph_state: Dict[str, Any],
        torus: Any,  # TorusGraph — avoid circular import
    ) -> Optional[Tuple[int, int]]:
        """Determine the next hop for a packet.

        Args:
            packet: The packet being routed.
            current_node: Current (x, y) position of the packet.
            graph_state: Snapshot of graph features (node/edge features, edge_index).
            torus: Reference to the TorusGraph for topology queries.

        Returns:
            Next-hop (x, y) coordinates, or None to hold the packet in buffer.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the routing algorithm."""
        ...

    def reset(self) -> None:
        """Reset any internal state between simulation runs."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
