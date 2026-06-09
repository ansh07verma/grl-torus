"""
2D Torus graph builder with NetworkX backend and PyG export.

Constructs N×N torus topology with wraparound edges, coordinate labelling,
failure injection, and conversion to PyTorch Geometric Data objects for
GNN consumption.
"""

from __future__ import annotations

import heapq
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from src.sim.link import TorusLink
from src.sim.node import TorusNode
from src.utils.logging import get_logger

logger = get_logger("sim.torus_graph")

# Direction vectors: (dx, dy) for each direction
DIRECTION_VECTORS = {
    "E": (1, 0),
    "W": (-1, 0),
    "N": (0, -1),  # North = decreasing y
    "S": (0, 1),   # South = increasing y
}

OPPOSITE_DIRECTION = {"N": "S", "S": "N", "E": "W", "W": "E"}


class TorusGraph:
    """2D Torus network graph.

    Builds an N×N torus via NetworkX with wraparound edges. Manages
    TorusNode and TorusLink objects and provides conversion to PyG Data.

    Args:
        n: Grid size — produces an N×N torus with N² nodes.
        buffer_depth: Max queue depth per directional queue per node.
        bandwidth_gbps: Link bandwidth in Gbps.
        propagation_delay_ns: Link propagation delay in ns.
        vc_count: Virtual channels per link (2 for Odd-Even routing).
    """

    def __init__(
        self,
        n: int,
        buffer_depth: int = 64,
        bandwidth_gbps: float = 100.0,
        propagation_delay_ns: float = 1.0,
        vc_count: int = 1,
    ):
        if n < 2:
            raise ValueError(f"Torus size must be >= 2, got {n}")

        self.n = n
        self.buffer_depth = buffer_depth
        self.bandwidth_gbps = bandwidth_gbps
        self.propagation_delay_ns = propagation_delay_ns
        self.vc_count = vc_count

        # Build NetworkX graph
        self._graph = nx.grid_2d_graph(n, n, periodic=True)

        # Create node and link objects
        self.nodes: Dict[Tuple[int, int], TorusNode] = {}
        self.links: Dict[Tuple[Tuple[int, int], Tuple[int, int]], TorusLink] = {}

        self._build_nodes()
        self._build_links()

        logger.info(
            f"Built {n}x{n} torus: {len(self.nodes)} nodes, "
            f"{len(self.links)} directed links"
        )

    def _build_nodes(self) -> None:
        """Create TorusNode objects for every grid position."""
        for x in range(self.n):
            for y in range(self.n):
                self.nodes[(x, y)] = TorusNode(
                    x=x,
                    y=y,
                    grid_size=self.n,
                    buffer_depth=self.buffer_depth,
                    vc_count=self.vc_count,
                )

    def _build_links(self) -> None:
        """Create TorusLink objects for every directed edge."""
        for x in range(self.n):
            for y in range(self.n):
                src = (x, y)
                for direction, (dx, dy) in DIRECTION_VECTORS.items():
                    dst_x = (x + dx) % self.n
                    dst_y = (y + dy) % self.n
                    dst = (dst_x, dst_y)

                    # Determine if this is a wraparound edge
                    is_wraparound = (
                        (direction == "E" and x == self.n - 1)
                        or (direction == "W" and x == 0)
                        or (direction == "S" and y == self.n - 1)
                        or (direction == "N" and y == 0)
                    )

                    self.links[(src, dst)] = TorusLink(
                        src_node=src,
                        dst_node=dst,
                        direction=direction,
                        bandwidth_gbps=self.bandwidth_gbps,
                        propagation_delay_ns=self.propagation_delay_ns,
                        is_wraparound=is_wraparound,
                    )

    # ----------------------------------------------------------------
    # Topology queries
    # ----------------------------------------------------------------

    def get_neighbor(self, node: Tuple[int, int], direction: str) -> Tuple[int, int]:
        """Get the neighbor of a node in a given direction, accounting for wraparound.

        Args:
            node: (x, y) coordinates.
            direction: One of 'N', 'S', 'E', 'W'.

        Returns:
            (x, y) of the neighbor.
        """
        dx, dy = DIRECTION_VECTORS[direction]
        return ((node[0] + dx) % self.n, (node[1] + dy) % self.n)

    def get_neighbors(self, node: Tuple[int, int]) -> Dict[str, Tuple[int, int]]:
        """Get all four neighbors of a node.

        Returns:
            Dict mapping direction -> neighbor coordinates.
        """
        return {d: self.get_neighbor(node, d) for d in DIRECTION_VECTORS}

    def get_direction(self, src: Tuple[int, int], dst: Tuple[int, int]) -> Optional[str]:
        """Determine the direction from src to dst (must be adjacent).

        Returns:
            Direction string, or None if not adjacent.
        """
        for direction, neighbor in self.get_neighbors(src).items():
            if neighbor == dst:
                return direction
        return None

    def get_link(
        self, src: Tuple[int, int], dst: Tuple[int, int]
    ) -> Optional[TorusLink]:
        """Get the link object between two adjacent nodes."""
        return self.links.get((src, dst))

    def get_node(self, coords: Tuple[int, int]) -> TorusNode:
        """Get the node object at given coordinates."""
        return self.nodes[coords]

    def all_node_coords(self) -> List[Tuple[int, int]]:
        """Return all node coordinates in row-major order."""
        return [(x, y) for y in range(self.n) for x in range(self.n)]

    @property
    def num_nodes(self) -> int:
        """Total number of nodes (N²)."""
        return self.n * self.n

    @property
    def num_links(self) -> int:
        """Total number of directed links (4 × N²)."""
        return len(self.links)

    # ----------------------------------------------------------------
    # Distance computation
    # ----------------------------------------------------------------

    def torus_distance(
        self, src: Tuple[int, int], dst: Tuple[int, int]
    ) -> int:
        """Manhattan distance on the torus (accounting for wraparound).

        Args:
            src: Source (x, y).
            dst: Destination (x, y).

        Returns:
            Minimum hop count.
        """
        dx = abs(dst[0] - src[0])
        dy = abs(dst[1] - src[1])
        # On a torus ring of size N, min distance is min(d, N-d)
        dx = min(dx, self.n - dx)
        dy = min(dy, self.n - dy)
        return dx + dy

    def shortest_direction_x(
        self, src: Tuple[int, int], dst: Tuple[int, int]
    ) -> Optional[str]:
        """Determine the shortest X-direction from src to dst.

        Returns:
            'E', 'W', or None if same column.
        """
        diff = (dst[0] - src[0]) % self.n
        if diff == 0:
            return None
        return "E" if diff <= self.n // 2 else "W"

    def shortest_direction_y(
        self, src: Tuple[int, int], dst: Tuple[int, int]
    ) -> Optional[str]:
        """Determine the shortest Y-direction from src to dst.

        Returns:
            'S', 'N', or None if same row.
        """
        diff = (dst[1] - src[1]) % self.n
        if diff == 0:
            return None
        return "S" if diff <= self.n // 2 else "N"

    # ----------------------------------------------------------------
    # Failure injection
    # ----------------------------------------------------------------

    def inject_link_failures(
        self, failure_rate: float, rng: Optional[np.random.Generator] = None
    ) -> int:
        """Randomly fail a fraction of links.

        Args:
            failure_rate: Fraction of links to fail [0, 1].
            rng: NumPy random generator for determinism.

        Returns:
            Number of links failed.
        """
        if rng is None:
            rng = np.random.default_rng()

        link_keys = list(self.links.keys())
        num_to_fail = int(len(link_keys) * failure_rate)
        failed_indices = rng.choice(len(link_keys), size=num_to_fail, replace=False)

        count = 0
        for idx in failed_indices:
            link = self.links[link_keys[idx]]
            if not link.is_failed:
                link.fail()
                count += 1

        # Mark nodes with all links failed
        for coords, node in self.nodes.items():
            all_failed = all(
                self.links.get((coords, self.get_neighbor(coords, d)), TorusLink(coords, coords, d)).is_failed
                for d in DIRECTION_VECTORS
            )
            if all_failed:
                node.is_failed = True

        logger.info(f"Injected {count} link failures ({failure_rate*100:.1f}% rate)")
        return count

    def inject_node_failures(
        self, failure_rate: float, rng: Optional[np.random.Generator] = None
    ) -> int:
        """Randomly fail a fraction of nodes (and all their links).

        Args:
            failure_rate: Fraction of nodes to fail [0, 1].
            rng: NumPy random generator.

        Returns:
            Number of nodes failed.
        """
        if rng is None:
            rng = np.random.default_rng()

        node_keys = list(self.nodes.keys())
        num_to_fail = int(len(node_keys) * failure_rate)
        failed_indices = rng.choice(len(node_keys), size=num_to_fail, replace=False)

        count = 0
        for idx in failed_indices:
            coords = node_keys[idx]
            node = self.nodes[coords]
            if not node.is_failed:
                node.is_failed = True
                count += 1
                # Fail all incident links
                for d in DIRECTION_VECTORS:
                    neighbor = self.get_neighbor(coords, d)
                    link_out = self.links.get((coords, neighbor))
                    link_in = self.links.get((neighbor, coords))
                    if link_out:
                        link_out.fail()
                    if link_in:
                        link_in.fail()

        logger.info(f"Injected {count} node failures ({failure_rate*100:.1f}% rate)")
        return count

    def get_active_nodes(self) -> List[Tuple[int, int]]:
        """Return coordinates of all non-failed nodes."""
        return [c for c, n in self.nodes.items() if not n.is_failed]

    def get_active_neighbors(
        self, node: Tuple[int, int]
    ) -> Dict[str, Tuple[int, int]]:
        """Get non-failed neighbors reachable via non-failed links."""
        result = {}
        for direction, neighbor in self.get_neighbors(node).items():
            link = self.links.get((node, neighbor))
            if link and not link.is_failed and not self.nodes[neighbor].is_failed:
                result[direction] = neighbor
        return result

    # ----------------------------------------------------------------
    # Shortest path (congestion-weighted)
    # ----------------------------------------------------------------

    def shortest_path_weighted(
        self,
        src: Tuple[int, int],
        dst: Tuple[int, int],
        weight_fn: Optional[Callable[[TorusLink], float]] = None,
    ) -> Tuple[List[Tuple[int, int]], float]:
        """Dijkstra shortest path on congestion-weighted torus.

        Args:
            src: Source node coordinates.
            dst: Destination node coordinates.
            weight_fn: Function mapping TorusLink -> edge weight.
                       Default: queue_length × link_utilisation + 1.

        Returns:
            (path, total_weight) — list of node coordinates and path cost.
        """
        if weight_fn is None:
            def weight_fn(link: TorusLink) -> float:
                src_node = self.nodes[link.src_node]
                direction = link.direction
                queue_occ = src_node.queues[direction].occupancy
                return (queue_occ + 1) * (link.utilisation + 0.1)

        # Dijkstra
        dist: Dict[Tuple[int, int], float] = {src: 0.0}
        prev: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {src: None}
        visited: Set[Tuple[int, int]] = set()
        heap = [(0.0, src)]

        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)

            if u == dst:
                break

            for direction, neighbor in self.get_active_neighbors(u).items():
                link = self.links[(u, neighbor)]
                w = weight_fn(link)
                new_dist = d + w
                if new_dist < dist.get(neighbor, float("inf")):
                    dist[neighbor] = new_dist
                    prev[neighbor] = u
                    heapq.heappush(heap, (new_dist, neighbor))

        # Reconstruct path
        if dst not in prev:
            return [], float("inf")

        path = []
        current: Optional[Tuple[int, int]] = dst
        while current is not None:
            path.append(current)
            current = prev[current]
        path.reverse()

        return path, dist.get(dst, float("inf"))

    def get_optimal_next_hop(
        self,
        src: Tuple[int, int],
        dst: Tuple[int, int],
        weight_fn: Optional[Callable[[TorusLink], float]] = None,
    ) -> Optional[str]:
        """Get optimal next-hop direction from src toward dst using Dijkstra.

        Returns:
            Direction string ('N'/'S'/'E'/'W') or None if unreachable.
        """
        path, _ = self.shortest_path_weighted(src, dst, weight_fn)
        if len(path) < 2:
            return None
        next_node = path[1]
        return self.get_direction(src, next_node)

    # ----------------------------------------------------------------
    # Graph state extraction
    # ----------------------------------------------------------------

    def get_graph_state(self) -> Dict[str, Any]:
        """Snapshot of current graph state for routing decisions.

        Returns:
            Dict with 'node_features', 'edge_features', 'edge_index',
            and 'node_coords' arrays.
        """
        node_coords = self.all_node_coords()
        coord_to_idx = {c: i for i, c in enumerate(node_coords)}

        # Node features: [N², 8]
        node_features = np.stack(
            [self.nodes[c].get_features() for c in node_coords], axis=0
        )

        # Edge index and features
        edge_src_list = []
        edge_dst_list = []
        edge_features_list = []

        for (src, dst), link in self.links.items():
            if src in coord_to_idx and dst in coord_to_idx:
                edge_src_list.append(coord_to_idx[src])
                edge_dst_list.append(coord_to_idx[dst])
                edge_features_list.append(link.get_features())

        edge_index = np.array([edge_src_list, edge_dst_list], dtype=np.int64)
        edge_features = (
            np.stack(edge_features_list, axis=0)
            if edge_features_list
            else np.zeros((0, 3), dtype=np.float32)
        )

        return {
            "node_features": node_features,     # [N², 8]
            "edge_index": edge_index,           # [2, num_edges]
            "edge_features": edge_features,     # [num_edges, 3]
            "node_coords": node_coords,         # List[(x,y)]
            "coord_to_idx": coord_to_idx,       # Dict[(x,y) -> int]
        }

    def to_pyg_data(self) -> Any:
        """Convert current graph state to PyTorch Geometric Data object.

        Returns:
            torch_geometric.data.Data with x, edge_index, edge_attr.
        """
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError:
            raise ImportError(
                "PyTorch Geometric is required for to_pyg_data(). "
                "Install with: pip install torch-geometric"
            )

        state = self.get_graph_state()

        return Data(
            x=torch.tensor(state["node_features"], dtype=torch.float32),
            edge_index=torch.tensor(state["edge_index"], dtype=torch.long),
            edge_attr=torch.tensor(state["edge_features"], dtype=torch.float32),
            num_nodes=self.num_nodes,
        )

    # ----------------------------------------------------------------
    # Adjacency matrix export
    # ----------------------------------------------------------------

    def to_adjacency_matrix(self) -> np.ndarray:
        """Export adjacency matrix as NumPy array.

        Returns:
            [N², N²] binary adjacency matrix.
        """
        node_coords = self.all_node_coords()
        coord_to_idx = {c: i for i, c in enumerate(node_coords)}
        n_nodes = len(node_coords)

        adj = np.zeros((n_nodes, n_nodes), dtype=np.int32)
        for (src, dst) in self.links:
            if src in coord_to_idx and dst in coord_to_idx:
                adj[coord_to_idx[src], coord_to_idx[dst]] = 1

        return adj

    # ----------------------------------------------------------------
    # Reset
    # ----------------------------------------------------------------

    def reset(self) -> None:
        """Reset all node and link state for a new simulation run."""
        for node in self.nodes.values():
            node.reset()
            node.is_failed = False
        for link in self.links.values():
            link.reset()
            link.is_failed = False
            link.failure_prob = 0.0

    def __repr__(self) -> str:
        active = len(self.get_active_nodes())
        return f"TorusGraph({self.n}x{self.n}, active_nodes={active}/{self.num_nodes})"
