"""
Odd-Even Routing — adaptive, deadlock-free with virtual channels.

Implements the Odd-Even turn model (Chiu 2000) adapted for 2D torus with
2 virtual channels per physical link to handle wraparound cycle breaking.

Turn Restrictions (mesh-based):
    - Even columns: prohibit East→North (EN), East→South (ES)
    - Odd columns:  prohibit North→West (NW), South→West (SW)

Virtual Channel Strategy (for torus):
    - VC-0: Used for wraparound link traversals
    - VC-1: Used for non-wraparound link traversals
    This breaks the cycles introduced by wraparound edges.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.routers.base import BaseRouter
from src.sim.packet import Packet


class OddEvenRouter(BaseRouter):
    """Odd-Even adaptive routing for 2D torus with virtual channels.

    The router selects among legal next-hops (those not violating turn
    restrictions) and picks the one with the lowest queue occupancy,
    providing congestion-aware adaptive routing.
    """

    @property
    def name(self) -> str:
        return "Odd-Even"

    def _is_turn_legal(
        self,
        incoming_dir: Optional[str],
        outgoing_dir: str,
        current_col: int,
    ) -> bool:
        """Check if a turn from incoming_dir to outgoing_dir is legal.

        Args:
            incoming_dir: Direction from which the packet arrived (None if at source).
            outgoing_dir: Proposed outgoing direction.
            current_col: X-coordinate (column) of current node.

        Returns:
            True if the turn is allowed under Odd-Even restrictions.
        """
        if incoming_dir is None:
            # At source — no turn restriction
            return True

        is_even_col = (current_col % 2 == 0)

        if is_even_col:
            # Even columns: prohibit E→N and E→S
            if incoming_dir == "E" and outgoing_dir == "N":
                return False
            if incoming_dir == "E" and outgoing_dir == "S":
                return False
        else:
            # Odd columns: prohibit N→W and S→W
            if incoming_dir == "N" and outgoing_dir == "W":
                return False
            if incoming_dir == "S" and outgoing_dir == "W":
                return False

        return True

    def _get_incoming_direction(self, packet: Packet) -> Optional[str]:
        """Determine the incoming direction based on packet's last hop.

        Returns:
            Direction the packet arrived from, or None if at source.
        """
        if len(packet.hops) < 2:
            return None

        prev_node = packet.hops[-2]
        curr_node = packet.hops[-1]

        # Determine direction prev -> curr
        dx = curr_node[0] - prev_node[0]
        dy = curr_node[1] - prev_node[1]

        # Handle wraparound — the displacement might be +(N-1) or -(N-1)
        # We need to find the actual direction taken
        if dx == 1 or dx < -1:  # Moved East (including wraparound)
            return "E"
        elif dx == -1 or dx > 1:  # Moved West (including wraparound)
            return "W"
        elif dy == 1 or dy < -1:  # Moved South
            return "S"
        elif dy == -1 or dy > 1:  # Moved North
            return "N"

        return None

    def _get_productive_directions(
        self,
        current: Tuple[int, int],
        dst: Tuple[int, int],
        torus: Any,
    ) -> List[str]:
        """Get directions that move the packet closer to the destination.

        Returns:
            List of productive direction strings.
        """
        directions = []

        # X dimension
        x_dir = torus.shortest_direction_x(current, dst)
        if x_dir:
            directions.append(x_dir)

        # Y dimension
        y_dir = torus.shortest_direction_y(current, dst)
        if y_dir:
            directions.append(y_dir)

        return directions

    def _select_vc(self, current: Tuple[int, int], direction: str, torus: Any) -> int:
        """Select the virtual channel for the given hop.

        VC-0 for wraparound links, VC-1 for non-wraparound.

        Returns:
            Virtual channel index (0 or 1).
        """
        neighbor = torus.get_neighbor(current, direction)
        link = torus.get_link(current, neighbor)
        if link and link.is_wraparound:
            return 0
        return 1

    def route(
        self,
        packet: Packet,
        current_node: Tuple[int, int],
        graph_state: Dict[str, Any],
        torus: Any,
    ) -> Optional[Tuple[int, int]]:
        """Route using Odd-Even adaptive routing with virtual channels.

        Algorithm:
            1. Get productive directions (those reducing distance to dst)
            2. Filter by turn legality (Odd-Even restrictions)
            3. Filter by link availability (not failed)
            4. Among remaining, pick direction with lowest queue occupancy
            5. If no productive legal direction exists, try any legal direction
            6. If no legal direction exists, hold (return None)
        """
        dst = packet.dst

        if current_node == dst:
            return None

        incoming_dir = self._get_incoming_direction(packet)
        current_col = current_node[0]
        current_node_obj = torus.get_node(current_node)

        # Step 1: Get productive directions
        productive = self._get_productive_directions(current_node, dst, torus)

        # Step 2 & 3: Filter by turn legality and link availability
        candidates: List[Tuple[str, Tuple[int, int], int]] = []
        for direction in productive:
            if not self._is_turn_legal(incoming_dir, direction, current_col):
                continue
            neighbor = torus.get_neighbor(current_node, direction)
            link = torus.get_link(current_node, neighbor)
            if link and not link.is_failed and not torus.nodes[neighbor].is_failed:
                vc = self._select_vc(current_node, direction, torus)
                queue_occ = current_node_obj.queues[direction].get_vc_occupancy(vc)
                candidates.append((direction, neighbor, queue_occ))

        # Step 4: Pick lowest queue occupancy
        if candidates:
            candidates.sort(key=lambda x: x[2])  # Sort by queue occupancy
            return candidates[0][1]

        # Step 5: Try any legal non-productive direction
        all_directions = ["N", "S", "E", "W"]
        for direction in all_directions:
            if direction in productive:
                continue  # Already tried
            if not self._is_turn_legal(incoming_dir, direction, current_col):
                continue
            neighbor = torus.get_neighbor(current_node, direction)
            link = torus.get_link(current_node, neighbor)
            if link and not link.is_failed and not torus.nodes[neighbor].is_failed:
                return neighbor

        # Step 6: No legal direction — hold
        return None
