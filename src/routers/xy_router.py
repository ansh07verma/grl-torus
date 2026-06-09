"""
XY Routing — deterministic, deadlock-free baseline.

Routes packets first along the X dimension (East/West), then along the
Y dimension (North/South), using the shortest direction on each torus ring.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from src.routers.base import BaseRouter
from src.sim.packet import Packet


class XYRouter(BaseRouter):
    """XY dimension-order routing for 2D torus.

    Algorithm:
        1. If current_x != dst_x: move in shortest X-direction (E or W)
        2. Else if current_y != dst_y: move in shortest Y-direction (N or S)
        3. Else: arrived at destination

    Properties:
        - Deterministic: same (src, dst) always produces same path
        - Deadlock-free: no cyclic dependencies in channel usage
        - Not congestion-aware: cannot avoid hotspots
    """

    @property
    def name(self) -> str:
        return "XY"

    def route(
        self,
        packet: Packet,
        current_node: Tuple[int, int],
        graph_state: Dict[str, Any],
        torus: Any,
    ) -> Optional[Tuple[int, int]]:
        """Route using XY dimension-order routing.

        Returns:
            Next-hop coordinates, or None if already at destination.
        """
        dst = packet.dst
        cx, cy = current_node
        dx, dy = dst

        # Already at destination
        if current_node == dst:
            return None

        # Phase 1: Route along X dimension
        if cx != dx:
            direction = torus.shortest_direction_x(current_node, dst)
            if direction:
                next_hop = torus.get_neighbor(current_node, direction)
                # Check if the link is available
                link = torus.get_link(current_node, next_hop)
                if link and not link.is_failed and not torus.nodes[next_hop].is_failed:
                    return next_hop
                # Try the other X direction as fallback
                alt_dir = "W" if direction == "E" else "E"
                alt_hop = torus.get_neighbor(current_node, alt_dir)
                link = torus.get_link(current_node, alt_hop)
                if link and not link.is_failed and not torus.nodes[alt_hop].is_failed:
                    return alt_hop

        # Phase 2: Route along Y dimension
        if cy != dy:
            direction = torus.shortest_direction_y(current_node, dst)
            if direction:
                next_hop = torus.get_neighbor(current_node, direction)
                link = torus.get_link(current_node, next_hop)
                if link and not link.is_failed and not torus.nodes[next_hop].is_failed:
                    return next_hop
                # Try the other Y direction as fallback
                alt_dir = "N" if direction == "S" else "S"
                alt_hop = torus.get_neighbor(current_node, alt_dir)
                link = torus.get_link(current_node, alt_hop)
                if link and not link.is_failed and not torus.nodes[alt_hop].is_failed:
                    return alt_hop

        # Fallback: try any non-failed neighbor
        active = torus.get_active_neighbors(current_node)
        if active:
            return next(iter(active.values()))

        # No path available — hold
        return None
