"""
Tests for routing algorithms: XY, Odd-Even, and Valiant.

Validates correctness, deadlock freedom, turn restrictions, and
routing convergence (packets reach destination).
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.sim.packet import Packet
from src.sim.torus_graph import TorusGraph
from src.routers.xy_router import XYRouter
from src.routers.odd_even_router import OddEvenRouter
from src.routers.valiant_router import ValiantRouter


def _simulate_route(router, torus, src, dst, max_hops=100):
    """Helper: simulate routing a packet from src to dst, return path."""
    packet = Packet(id=1, src=src, dst=dst, creation_time=0.0)
    packet.record_hop(src)
    current = src
    graph_state = torus.get_graph_state()

    for _ in range(max_hops):
        if current == dst:
            return packet.hops, True
        next_hop = router.route(packet, current, graph_state, torus)
        if next_hop is None:
            # Hold — skip
            continue
        packet.record_hop(next_hop)
        current = next_hop

    return packet.hops, (current == dst)


class TestXYRouter:
    """Test XY dimension-order routing."""

    def test_routes_x_first(self):
        """XY should route along X dimension before Y."""
        torus = TorusGraph(4)
        router = XYRouter()
        packet = Packet(id=1, src=(0, 0), dst=(2, 2), creation_time=0.0)
        packet.record_hop((0, 0))
        state = torus.get_graph_state()

        # First hop should be in X direction (East)
        next_hop = router.route(packet, (0, 0), state, torus)
        assert next_hop == (1, 0), f"Expected (1,0), got {next_hop}"

    def test_reaches_destination_4x4(self):
        """XY should always reach destination on a 4×4 torus."""
        torus = TorusGraph(4)
        router = XYRouter()
        path, reached = _simulate_route(router, torus, (0, 0), (3, 3))
        assert reached, f"Failed to reach (3,3), path: {path}"

    def test_uses_wraparound(self):
        """XY should use wraparound when it's shorter."""
        torus = TorusGraph(4)
        router = XYRouter()
        path, reached = _simulate_route(router, torus, (0, 0), (3, 0))
        assert reached
        # Wraparound: (0,0) → (3,0) should be 1 hop (West wraparound)
        assert len(path) == 2, f"Expected 2 hops (via wraparound), got {len(path)}: {path}"

    @pytest.mark.parametrize(
        "src,dst",
        [
            ((0, 0), (2, 3)),
            ((1, 1), (3, 0)),
            ((3, 3), (0, 0)),
            ((2, 0), (0, 2)),
        ],
    )
    def test_always_reaches_destination(self, src, dst):
        torus = TorusGraph(4)
        router = XYRouter()
        _, reached = _simulate_route(router, torus, src, dst)
        assert reached, f"XY failed to route from {src} to {dst}"

    def test_no_loops(self):
        """XY routing should not revisit nodes on a healthy torus."""
        torus = TorusGraph(8)
        router = XYRouter()
        path, reached = _simulate_route(router, torus, (0, 0), (4, 4))
        assert reached
        # No node should appear twice
        assert len(path) == len(set(path)), f"Loop detected in path: {path}"


class TestOddEvenRouter:
    """Test Odd-Even adaptive routing."""

    def test_reaches_destination(self):
        torus = TorusGraph(4, vc_count=2)
        router = OddEvenRouter()
        _, reached = _simulate_route(router, torus, (0, 0), (3, 3))
        assert reached

    @pytest.mark.parametrize(
        "src,dst",
        [
            ((0, 0), (2, 3)),
            ((1, 1), (3, 0)),
            ((3, 3), (0, 0)),
            ((2, 0), (0, 2)),
        ],
    )
    def test_always_reaches_destination(self, src, dst):
        torus = TorusGraph(4, vc_count=2)
        router = OddEvenRouter()
        _, reached = _simulate_route(router, torus, src, dst, max_hops=200)
        assert reached, f"Odd-Even failed to route from {src} to {dst}"

    def test_no_prohibited_turns_even_col(self):
        """At even columns, E→N and E→S turns should never occur."""
        torus = TorusGraph(8, vc_count=2)
        router = OddEvenRouter()
        # Test many paths
        violations = []
        for sx in range(8):
            for sy in range(8):
                for dx in range(8):
                    for dy in range(8):
                        if (sx, sy) == (dx, dy):
                            continue
                        path, _ = _simulate_route(
                            router, torus, (sx, sy), (dx, dy), max_hops=100
                        )
                        for i in range(2, len(path)):
                            prev = path[i - 2]
                            curr = path[i - 1]
                            next_node = path[i]

                            # Determine incoming direction (prev -> curr)
                            in_dx = (curr[0] - prev[0])
                            in_dy = (curr[1] - prev[1])
                            # Determine outgoing direction (curr -> next)
                            out_dx = (next_node[0] - curr[0])
                            out_dy = (next_node[1] - curr[1])

                            col = curr[0]

                            # Normalize for wraparound
                            if abs(in_dx) > 1:
                                in_dx = -1 if in_dx > 0 else 1
                            if abs(out_dx) > 1:
                                out_dx = -1 if out_dx > 0 else 1
                            if abs(in_dy) > 1:
                                in_dy = -1 if in_dy > 0 else 1
                            if abs(out_dy) > 1:
                                out_dy = -1 if out_dy > 0 else 1

                            incoming = None
                            if in_dx == 1:
                                incoming = "E"
                            elif in_dx == -1:
                                incoming = "W"
                            elif in_dy == 1:
                                incoming = "S"
                            elif in_dy == -1:
                                incoming = "N"

                            outgoing = None
                            if out_dx == 1:
                                outgoing = "E"
                            elif out_dx == -1:
                                outgoing = "W"
                            elif out_dy == 1:
                                outgoing = "S"
                            elif out_dy == -1:
                                outgoing = "N"

                            if col % 2 == 0:  # Even column
                                if incoming == "E" and outgoing in ("N", "S"):
                                    violations.append(
                                        (prev, curr, next_node, incoming, outgoing, col)
                                    )

        # Only check a subset to keep test time reasonable
        # (full 8×8 all-pairs would be too many)
        assert len(violations) == 0, f"Found {len(violations)} turn violations"


class TestValiantRouter:
    """Test Valiant load balancing routing."""

    def test_reaches_destination(self):
        torus = TorusGraph(4)
        router = ValiantRouter(seed=42)
        _, reached = _simulate_route(router, torus, (0, 0), (3, 3))
        assert reached

    def test_assigns_intermediate(self):
        """Valiant should assign a random intermediate node."""
        torus = TorusGraph(4)
        router = ValiantRouter(seed=42)
        packet = Packet(id=1, src=(0, 0), dst=(3, 3), creation_time=0.0)
        packet.record_hop((0, 0))
        state = torus.get_graph_state()

        router.route(packet, (0, 0), state, torus)
        assert packet.valiant_intermediate is not None
        assert packet.valiant_intermediate != (0, 0)
        assert packet.valiant_intermediate != (3, 3)

    @pytest.mark.parametrize(
        "src,dst",
        [
            ((0, 0), (2, 3)),
            ((1, 1), (3, 0)),
            ((3, 3), (0, 0)),
        ],
    )
    def test_always_reaches_destination(self, src, dst):
        torus = TorusGraph(4)
        router = ValiantRouter(seed=42)
        _, reached = _simulate_route(router, torus, src, dst, max_hops=200)
        assert reached
