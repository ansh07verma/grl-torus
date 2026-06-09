"""
GRL Router — Graph Reinforcement Learning (GNN encoder + DQN policy).

Combines pretrained GraphSAGE embeddings with a trained Dueling DQN
Q-network for adaptive routing. The GNN provides graph-aware state
representation; the DQN selects next-hop actions.

Supports two variants:
    - Frozen GNN: encoder weights fixed, only DQN is trained
    - Joint training: both GNN and DQN trained together
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from src.models.dqn import DuelingDQN
from src.models.graphsage import GraphSAGEEncoder, GNNRoutingModel
from src.routers.base import BaseRouter
from src.sim.packet import Packet
from src.utils.logging import get_logger

logger = get_logger("routers.grl_router")

# Action mapping
ACTION_TO_DIRECTION = {0: "N", 1: "S", 2: "E", 3: "W", 4: "HOLD"}
DIRECTION_TO_ACTION = {v: k for k, v in ACTION_TO_DIRECTION.items()}


class GRLRouter(BaseRouter):
    """GRL Router combining GNN encoder + DQN policy.

    Args:
        gnn_checkpoint_path: Path to pretrained GNN encoder checkpoint.
        dqn_checkpoint_path: Path to trained DQN checkpoint.
        device: Compute device.
        gnn_encoder: Optional pre-loaded GNN encoder.
        dqn_network: Optional pre-loaded DQN network.
        grid_size: Torus N (for feature normalisation).
    """

    def __init__(
        self,
        gnn_checkpoint_path: Optional[str] = None,
        dqn_checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        gnn_encoder: Optional[GraphSAGEEncoder] = None,
        dqn_network: Optional[DuelingDQN] = None,
        grid_size: int = 8,
    ):
        self.device = device
        self.grid_size = grid_size

        # Load GNN encoder
        if gnn_encoder is not None:
            self.gnn_encoder = gnn_encoder.to(device)
        elif gnn_checkpoint_path is not None:
            self.gnn_encoder = self._load_gnn(gnn_checkpoint_path)
        else:
            self.gnn_encoder = GraphSAGEEncoder().to(device)

        # Load DQN
        if dqn_network is not None:
            self.dqn = dqn_network.to(device)
        elif dqn_checkpoint_path is not None:
            self.dqn = self._load_dqn(dqn_checkpoint_path)
        else:
            self.dqn = DuelingDQN().to(device)

        self.gnn_encoder.eval()
        self.dqn.eval()

        # Embedding cache
        self._cached_embeddings: Optional[torch.Tensor] = None
        self._cached_time: float = -1.0

        # Hold counter for deadlock prevention
        self._hold_counters: Dict[int, int] = {}

    def _load_gnn(self, path: str) -> GraphSAGEEncoder:
        """Load GNN encoder from a GNNRoutingModel checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        config = checkpoint.get("config", {})
        model = GNNRoutingModel(
            in_channels=config.get("in_channels", 8),
            hidden_channels=config.get("hidden_channels", 128),
            embedding_dim=config.get("embedding_dim", 64),
            num_layers=config.get("num_layers", 3),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        encoder = model.encoder.to(self.device)

        logger.info(f"Loaded GNN encoder from {path}")
        return encoder

    def _load_dqn(self, path: str) -> DuelingDQN:
        """Load DQN network from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        config = checkpoint.get("config", {})

        dqn = DuelingDQN(
            state_dim=config.get("state_dim", 69),
            action_dim=config.get("action_dim", 5),
        )
        dqn.load_state_dict(checkpoint["model_state_dict"])
        dqn = dqn.to(self.device)

        logger.info(f"Loaded DQN from {path}")
        return dqn

    @property
    def name(self) -> str:
        return "GRL"

    def _get_embeddings(
        self, graph_state: Dict[str, Any], sim_time: float = 0
    ) -> torch.Tensor:
        """Get graph embeddings cached per simulation tick."""
        if self._cached_time != sim_time or self._cached_embeddings is None:
            with torch.no_grad():
                x = torch.tensor(
                    graph_state["node_features"], dtype=torch.float32, device=self.device
                )
                ei = torch.tensor(
                    graph_state["edge_index"], dtype=torch.long, device=self.device
                )
                self._cached_embeddings = self.gnn_encoder(x, ei)
            self._cached_time = sim_time

        return self._cached_embeddings

    def _build_state_vector(
        self,
        packet: Packet,
        current_node: Tuple[int, int],
        graph_state: Dict[str, Any],
        sim_time: float = 0,
    ) -> torch.Tensor:
        """Build the 69-dim DQN state vector.

        State = [64-dim GNN embedding of current node] + [5-dim packet features]
        Packet features: [src_x_norm, src_y_norm, dst_x_norm, dst_y_norm, hops_norm]
        """
        embeddings = self._get_embeddings(graph_state, sim_time)
        coord_to_idx = graph_state["coord_to_idx"]
        node_idx = coord_to_idx[current_node]

        # Current node embedding: [64]
        node_emb = embeddings[node_idx]

        # Packet features: [5]
        n = max(self.grid_size - 1, 1)
        packet_features = torch.tensor(
            [
                packet.src[0] / n,
                packet.src[1] / n,
                packet.dst[0] / n,
                packet.dst[1] / n,
                min(packet.hops_so_far / (4 * self.grid_size), 1.0),
            ],
            dtype=torch.float32,
            device=self.device,
        )

        # Concatenate: [69]
        state = torch.cat([node_emb, packet_features])
        return state

    def route(
        self,
        packet: Packet,
        current_node: Tuple[int, int],
        graph_state: Dict[str, Any],
        torus: Any,
    ) -> Optional[Tuple[int, int]]:
        """Route using GRL (GNN embedding + DQN action selection).

        1. Build state vector from GNN embeddings + packet features
        2. DQN forward pass → Q-values for 5 actions
        3. Select argmax Q (greedy at inference)
        4. If Hold: check deadlock counter, force deflection after 3
        5. Return next-hop for selected direction
        """
        if current_node == packet.dst:
            return None

        # Build state
        state = self._build_state_vector(packet, current_node, graph_state)

        # DQN decision (greedy — no exploration at inference)
        action = self.dqn.select_action(state, epsilon=0.0)
        direction = ACTION_TO_DIRECTION[action]

        # Handle Hold action with deadlock prevention
        if direction == "HOLD":
            hold_count = self._hold_counters.get(packet.id, 0) + 1
            self._hold_counters[packet.id] = hold_count

            if hold_count > 3:
                # Force deflection
                active = torus.get_active_neighbors(current_node)
                if active:
                    directions = list(active.keys())
                    idx = hash(packet.id) % len(directions)
                    self._hold_counters[packet.id] = 0
                    return active[directions[idx]]
                return None

            return None  # Hold — caller will retry next tick

        # Reset hold counter on non-Hold action
        self._hold_counters[packet.id] = 0

        # Try the DQN-selected direction
        neighbor = torus.get_neighbor(current_node, direction)
        link = torus.get_link(current_node, neighbor)
        if link and not link.is_failed and not torus.nodes[neighbor].is_failed:
            return neighbor

        # If DQN direction is blocked, try other Q-values in order
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            q_values = self.dqn(state).squeeze()

        sorted_actions = q_values.argsort(descending=True)
        for alt_action in sorted_actions:
            alt_dir = ACTION_TO_DIRECTION[alt_action.item()]
            if alt_dir == "HOLD":
                continue
            alt_neighbor = torus.get_neighbor(current_node, alt_dir)
            alt_link = torus.get_link(current_node, alt_neighbor)
            if alt_link and not alt_link.is_failed and not torus.nodes[alt_neighbor].is_failed:
                return alt_neighbor

        # No available direction
        return None

    def reset(self) -> None:
        """Reset state between runs."""
        self._cached_embeddings = None
        self._cached_time = -1.0
        self._hold_counters.clear()
