"""
Tests for the TorusGraph builder.

Validates topology correctness, wraparound edges, PyG export,
failure injection, and shortest path computation.
"""

import sys
import os
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.sim.torus_graph import TorusGraph


class TestTorusTopology:
    """Test basic torus topology properties."""

    @pytest.mark.parametrize("n", [4, 8, 16])
    def test_node_count(self, n: int):
        """N×N torus should have exactly N² nodes."""
        torus = TorusGraph(n)
        assert torus.num_nodes == n * n

    @pytest.mark.parametrize("n", [4, 8, 16])
    def test_link_count(self, n: int):
        """N×N torus should have exactly 4×N² directed links."""
        torus = TorusGraph(n)
        assert torus.num_links == 4 * n * n

    def test_all_nodes_degree_4(self):
        """Every node in a torus has exactly 4 neighbors."""
        torus = TorusGraph(4)
        for coords in torus.all_node_coords():
            neighbors = torus.get_neighbors(coords)
            assert len(neighbors) == 4, f"Node {coords} has {len(neighbors)} neighbors"

    def test_wraparound_north(self):
        """Node (0,0) should connect to (0, N-1) going North."""
        torus = TorusGraph(4)
        north = torus.get_neighbor((0, 0), "N")
        assert north == (0, 3), f"North of (0,0) should be (0,3), got {north}"

    def test_wraparound_west(self):
        """Node (0,0) should connect to (N-1, 0) going West."""
        torus = TorusGraph(4)
        west = torus.get_neighbor((0, 0), "W")
        assert west == (3, 0), f"West of (0,0) should be (3,0), got {west}"

    def test_wraparound_south(self):
        """Node (0, N-1) should connect to (0, 0) going South."""
        torus = TorusGraph(4)
        south = torus.get_neighbor((0, 3), "S")
        assert south == (0, 0)

    def test_wraparound_east(self):
        """Node (N-1, 0) should connect to (0, 0) going East."""
        torus = TorusGraph(4)
        east = torus.get_neighbor((3, 0), "E")
        assert east == (0, 0)

    def test_non_wraparound(self):
        """Interior nodes should have standard neighbors."""
        torus = TorusGraph(4)
        assert torus.get_neighbor((1, 1), "N") == (1, 0)
        assert torus.get_neighbor((1, 1), "S") == (1, 2)
        assert torus.get_neighbor((1, 1), "E") == (2, 1)
        assert torus.get_neighbor((1, 1), "W") == (0, 1)

    def test_direction_detection(self):
        """get_direction should correctly identify direction between adjacent nodes."""
        torus = TorusGraph(4)
        assert torus.get_direction((1, 1), (2, 1)) == "E"
        assert torus.get_direction((1, 1), (0, 1)) == "W"
        assert torus.get_direction((1, 1), (1, 0)) == "N"
        assert torus.get_direction((1, 1), (1, 2)) == "S"

    def test_direction_non_adjacent_returns_none(self):
        """get_direction should return None for non-adjacent nodes."""
        torus = TorusGraph(4)
        assert torus.get_direction((0, 0), (2, 2)) is None


class TestTorusDistance:
    """Test torus distance computation."""

    def test_same_node(self):
        torus = TorusGraph(4)
        assert torus.torus_distance((0, 0), (0, 0)) == 0

    def test_adjacent(self):
        torus = TorusGraph(4)
        assert torus.torus_distance((0, 0), (1, 0)) == 1

    def test_diagonal(self):
        torus = TorusGraph(4)
        assert torus.torus_distance((0, 0), (1, 1)) == 2

    def test_wraparound_shorter(self):
        """On a 4×4 torus, (0,0) to (3,0) should be distance 1 via wraparound."""
        torus = TorusGraph(4)
        assert torus.torus_distance((0, 0), (3, 0)) == 1

    def test_max_distance(self):
        """Max distance on N×N torus is N (2 × N/2)."""
        torus = TorusGraph(8)
        # (0,0) to (4,4) = 4+4 = 8 via direct, but 4+4=8 via either direction
        assert torus.torus_distance((0, 0), (4, 4)) == 8

    def test_shortest_direction_x(self):
        torus = TorusGraph(8)
        assert torus.shortest_direction_x((0, 0), (3, 0)) == "E"
        assert torus.shortest_direction_x((0, 0), (5, 0)) == "W"
        assert torus.shortest_direction_x((0, 0), (0, 0)) is None

    def test_shortest_direction_y(self):
        torus = TorusGraph(8)
        assert torus.shortest_direction_y((0, 0), (0, 3)) == "S"
        assert torus.shortest_direction_y((0, 0), (0, 5)) == "N"


class TestGraphState:
    """Test graph state extraction."""

    def test_node_features_shape(self):
        torus = TorusGraph(4)
        state = torus.get_graph_state()
        assert state["node_features"].shape == (16, 8)

    def test_edge_index_shape(self):
        torus = TorusGraph(4)
        state = torus.get_graph_state()
        assert state["edge_index"].shape == (2, 64)  # 4 * 16 = 64 directed edges

    def test_edge_features_shape(self):
        torus = TorusGraph(4)
        state = torus.get_graph_state()
        assert state["edge_features"].shape == (64, 3)

    def test_adjacency_matrix_shape(self):
        torus = TorusGraph(4)
        adj = torus.to_adjacency_matrix()
        assert adj.shape == (16, 16)
        # Each row should have exactly 4 ones
        for i in range(16):
            assert adj[i].sum() == 4, f"Row {i} has {adj[i].sum()} connections"


class TestFailureInjection:
    """Test failure injection functionality."""

    def test_link_failure_count(self):
        torus = TorusGraph(4)
        rng = np.random.default_rng(42)
        count = torus.inject_link_failures(0.1, rng)
        assert count > 0
        assert count <= int(64 * 0.1) + 1  # ~10% of 64 links

    def test_node_failure(self):
        torus = TorusGraph(4)
        rng = np.random.default_rng(42)
        count = torus.inject_node_failures(0.1, rng)
        assert count > 0
        # Failed node's links should also be failed
        for coords, node in torus.nodes.items():
            if node.is_failed:
                for d in ["N", "S", "E", "W"]:
                    neighbor = torus.get_neighbor(coords, d)
                    link = torus.get_link(coords, neighbor)
                    assert link.is_failed

    def test_active_nodes_after_failure(self):
        torus = TorusGraph(4)
        rng = np.random.default_rng(42)
        torus.inject_node_failures(0.25, rng)
        active = torus.get_active_nodes()
        assert len(active) < 16
        assert len(active) > 0

    def test_reset_clears_failures(self):
        torus = TorusGraph(4)
        rng = np.random.default_rng(42)
        torus.inject_link_failures(0.5, rng)
        torus.reset()
        failed_links = [l for l in torus.links.values() if l.is_failed]
        assert len(failed_links) == 0


class TestShortestPath:
    """Test Dijkstra shortest path on torus."""

    def test_adjacent_path(self):
        torus = TorusGraph(4)
        path, cost = torus.shortest_path_weighted((0, 0), (1, 0))
        assert len(path) == 2
        assert path[0] == (0, 0)
        assert path[1] == (1, 0)

    def test_path_exists(self):
        torus = TorusGraph(4)
        path, cost = torus.shortest_path_weighted((0, 0), (3, 3))
        assert len(path) >= 2
        assert path[0] == (0, 0)
        assert path[-1] == (3, 3)

    def test_unreachable_path(self):
        """Path should be empty if destination is unreachable."""
        torus = TorusGraph(4)
        # Fail all links around node (1,1)
        for d in ["N", "S", "E", "W"]:
            neighbor = torus.get_neighbor((1, 1), d)
            link = torus.get_link(neighbor, (1, 1))
            if link:
                link.fail()
            link = torus.get_link((1, 1), neighbor)
            if link:
                link.fail()
        torus.nodes[(1, 1)].is_failed = True
        path, cost = torus.shortest_path_weighted((0, 0), (1, 1))
        assert len(path) == 0
