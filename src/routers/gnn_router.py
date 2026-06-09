"""
GNN supervised router — routes packets using a pretrained GraphSAGE model.

Loads a trained GNNRoutingModel checkpoint and uses it to predict the
optimal next-hop direction at each hop. Caches the GNN forward pass
per simulation tick for efficiency.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from src.models.graphsage import GNNRoutingModel
from src.routers.base import BaseRouter
from src.sim.packet import Packet
from src.utils.logging import get_logger

logger = get_logger("routers.gnn_router")

# Direction mapping
IDX_TO_DIRECTION = {0: "N", 1: "S", 2: "E", 3: "W"}


class GNNRouter(BaseRouter):
    """GNN-based supervised router.

    Uses a pretrained GraphSAGE + routing head to predict the optimal
    next-hop direction for each packet at each hop.

    Args:
        checkpoint_path: Path to the saved model checkpoint.
        device: Compute device ('cpu' or 'cuda').
        model: Optional pre-loaded model (overrides checkpoint_path).
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        model: Optional[GNNRoutingModel] = None,
    ):
        self.device = device
        self._cached_embeddings: Optional[torch.Tensor] = None
        self._cached_time: float = -1.0
        self._cached_node_features: Optional[torch.Tensor] = None
        self._cached_edge_index: Optional[torch.Tensor] = None

        if model is not None:
            self.model = model.to(device)
        elif checkpoint_path is not None:
            self.model = self._load_checkpoint(checkpoint_path)
        else:
            raise ValueError("Either checkpoint_path or model must be provided")

        self.model.eval()

    def _load_checkpoint(self, path: str) -> GNNRoutingModel:
        """Load model from checkpoint file."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        config = checkpoint.get("config", {})
        model = GNNRoutingModel(
            in_channels=config.get("in_channels", 8),
            hidden_channels=config.get("hidden_channels", 128),
            embedding_dim=config.get("embedding_dim", 64),
            num_layers=config.get("num_layers", 3),
            num_directions=4,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)

        logger.info(
            f"Loaded GNN checkpoint from {path} "
            f"(epoch {checkpoint.get('epoch', '?')}, "
            f"val_acc={checkpoint.get('val_acc', '?'):.4f})"
        )
        return model

    @property
    def name(self) -> str:
        return "GNN"

    def _get_cached_data(
        self, graph_state: Dict[str, Any], sim_time: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get or compute cached node features and edge index tensors."""
        if self._cached_time != sim_time or self._cached_node_features is None:
            self._cached_node_features = torch.tensor(
                graph_state["node_features"], dtype=torch.float32, device=self.device
            )
            self._cached_edge_index = torch.tensor(
                graph_state["edge_index"], dtype=torch.long, device=self.device
            )
            self._cached_embeddings = None  # Invalidate embedding cache
            self._cached_time = sim_time

        return self._cached_node_features, self._cached_edge_index

    def _get_embeddings(
        self, graph_state: Dict[str, Any], sim_time: float
    ) -> torch.Tensor:
        """Get node embeddings, cached per simulation tick."""
        nf, ei = self._get_cached_data(graph_state, sim_time)

        if self._cached_embeddings is None:
            with torch.no_grad():
                self._cached_embeddings = self.model.get_embeddings(nf, ei)

        return self._cached_embeddings

    def route(
        self,
        packet: Packet,
        current_node: Tuple[int, int],
        graph_state: Dict[str, Any],
        torus: Any,
    ) -> Optional[Tuple[int, int]]:
        """Route using GNN-predicted optimal direction.

        1. Get cached graph embeddings (one forward pass per tick)
        2. Look up current node and destination node embeddings
        3. Feed through routing head → direction logits
        4. Select argmax direction → return next-hop
        5. Fallback to any active neighbor if predicted direction is blocked
        """
        if current_node == packet.dst:
            return None

        coord_to_idx = graph_state["coord_to_idx"]

        # Get indices
        current_idx = coord_to_idx.get(current_node)
        dst_idx = coord_to_idx.get(packet.dst)

        if current_idx is None or dst_idx is None:
            return None

        # Get embeddings (cached per tick)
        # For the GNN router, we use the full model including routing head
        nf, ei = self._get_cached_data(graph_state, 0)  # sim_time not tracked here

        with torch.no_grad():
            current_idx_t = torch.tensor([current_idx], dtype=torch.long, device=self.device)
            dst_idx_t = torch.tensor([dst_idx], dtype=torch.long, device=self.device)

            logits = self.model(nf, ei, current_idx_t, dst_idx_t)  # [1, 4]
            # Sort by confidence (highest first)
            sorted_actions = logits.argsort(dim=-1, descending=True).squeeze()

        # Try directions in order of GNN confidence
        for action_idx in sorted_actions:
            direction = IDX_TO_DIRECTION[action_idx.item()]
            neighbor = torus.get_neighbor(current_node, direction)
            link = torus.get_link(current_node, neighbor)

            if link and not link.is_failed and not torus.nodes[neighbor].is_failed:
                return neighbor

        # Fallback: any active neighbor
        active = torus.get_active_neighbors(current_node)
        if active:
            return next(iter(active.values()))

        return None

    def reset(self) -> None:
        """Reset caches between simulation runs."""
        self._cached_embeddings = None
        self._cached_time = -1.0
        self._cached_node_features = None
        self._cached_edge_index = None
