"""
Valiant Load Balancing — two-phase routing via random intermediate node.

Phase 1: Route from source to a randomly chosen intermediate node using XY routing.
Phase 2: Route from intermediate to destination using XY routing.

Provably optimal for adversarial traffic load balance at the cost of
approximately 2× path length under benign traffic.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.routers.base import BaseRouter
from src.routers.xy_router import XYRouter
from src.sim.packet import Packet


class ValiantRouter(BaseRouter):
    """Valiant Load Balancing for 2D torus.

    Uses XY routing for each phase. The intermediate node is selected
    uniformly at random from all non-failed nodes.

    Args:
        seed: Random seed for intermediate node selection.
    """

    def __init__(self, seed: int = 42):
        self._xy_router = XYRouter()
        self._rng = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        return "Valiant"

    def route(
        self,
        packet: Packet,
        current_node: Tuple[int, int],
        graph_state: Dict[str, Any],
        torus: Any,
    ) -> Optional[Tuple[int, int]]:
        """Route using Valiant two-phase load balancing.

        On first call for a packet, selects a random intermediate node.
        Phase 1: XY-route toward intermediate.
        Phase 2: After reaching intermediate, XY-route toward destination.
        """
        # Already at destination
        if current_node == packet.dst:
            return None

        # Assign random intermediate node on first routing decision
        if packet.valiant_intermediate is None:
            active_nodes = torus.get_active_nodes()
            # Exclude source and destination for meaningful load balance
            candidates = [
                n for n in active_nodes
                if n != packet.src and n != packet.dst
            ]
            if not candidates:
                candidates = active_nodes
            idx = self._rng.integers(0, len(candidates))
            packet.valiant_intermediate = candidates[idx]
            packet.valiant_phase = 1

        intermediate = packet.valiant_intermediate

        # Phase 1: Route toward intermediate
        if packet.valiant_phase == 1:
            if current_node == intermediate:
                # Reached intermediate — switch to phase 2
                packet.valiant_phase = 2
            else:
                # Create a temporary packet targeting intermediate
                temp_packet = Packet(
                    id=packet.id,
                    src=packet.src,
                    dst=intermediate,
                    creation_time=packet.creation_time,
                )
                return self._xy_router.route(temp_packet, current_node, graph_state, torus)

        # Phase 2: Route toward destination
        return self._xy_router.route(packet, current_node, graph_state, torus)

    def reset(self) -> None:
        """Reset internal state."""
        self._xy_router.reset()
