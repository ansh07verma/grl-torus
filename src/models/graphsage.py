"""
GraphSAGE encoder for 2D torus network state embedding.

Implements a 3-layer GraphSAGE (Hamilton et al., 2017) using PyTorch Geometric's
SAGEConv. Takes 8-dim node features and produces 64-dim per-node embeddings
that capture local neighbourhood structure and congestion patterns.

Architecture:
    Layer 1: SAGEConv(8 → 128, mean aggregation, project=True) + BatchNorm + ReLU + Dropout
    Layer 2: SAGEConv(128 → 128, mean aggregation) + BatchNorm + ReLU + Dropout
    Layer 3: SAGEConv(128 → 64, mean aggregation) — no activation on output

The output embeddings are fed to either:
    - A classification head (supervised training: next-hop prediction)
    - The DQN Q-network (reinforcement learning: action selection)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm


class GraphSAGEEncoder(nn.Module):
    """GraphSAGE encoder producing per-node embeddings from torus graph state.

    Args:
        in_channels: Input node feature dimension (default 8).
        hidden_channels: Hidden layer dimension (default 128).
        out_channels: Output embedding dimension (default 64).
        num_layers: Number of GraphSAGE layers (default 3).
        dropout: Dropout rate between layers (default 0.1).
        aggr: Aggregation scheme — 'mean', 'max', or 'lstm' (default 'mean').
    """

    def __init__(
        self,
        in_channels: int = 8,
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        aggr: str = "mean",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Layer 1: in_channels → hidden_channels (with input projection)
        self.convs.append(
            SAGEConv(in_channels, hidden_channels, aggr=aggr, project=True)
        )
        self.norms.append(BatchNorm(hidden_channels))

        # Middle layers: hidden_channels → hidden_channels
        for i in range(1, num_layers - 1):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels, aggr=aggr)
            )
            self.norms.append(BatchNorm(hidden_channels))

        # Final layer: hidden_channels → out_channels (no norm/activation)
        self.convs.append(
            SAGEConv(hidden_channels, out_channels, aggr=aggr)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through GraphSAGE layers.

        Args:
            x: Node features [num_nodes, in_channels].
            edge_index: Edge indices [2, num_edges].

        Returns:
            Node embeddings [num_nodes, out_channels].
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                # Apply BatchNorm, ReLU, Dropout for all layers except the last
                x = self.norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x  # [num_nodes, out_channels]

    def get_embedding_dim(self) -> int:
        """Return the output embedding dimension."""
        return self.out_channels


class GNNRoutingHead(nn.Module):
    """Classification head for supervised next-hop prediction.

    Takes GraphSAGE node embeddings and predicts the optimal next-hop
    direction (N/S/E/W) for each node given a specific destination.

    Architecture:
        Input: node_embedding (64) + destination_embedding (64) = 128
        Linear(128, 64) → ReLU → Linear(64, 4)
        Output: logits over 4 directions

    Args:
        embedding_dim: Dimension of input node embeddings (default 64).
        num_directions: Number of output directions (default 4: N/S/E/W).
    """

    def __init__(self, embedding_dim: int = 64, num_directions: int = 4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, num_directions),
        )

    def forward(
        self,
        node_embedding: torch.Tensor,
        dst_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Predict next-hop direction logits.

        Args:
            node_embedding: Current node's embedding [batch, embedding_dim].
            dst_embedding: Destination node's embedding [batch, embedding_dim].

        Returns:
            Logits over directions [batch, num_directions].
        """
        combined = torch.cat([node_embedding, dst_embedding], dim=-1)
        return self.head(combined)


class GNNRoutingModel(nn.Module):
    """Complete GNN model: GraphSAGE encoder + routing classification head.

    This model encodes the entire torus graph and predicts next-hop directions
    for (current_node, destination) pairs.

    Args:
        in_channels: Node feature dimension (default 8).
        hidden_channels: GraphSAGE hidden dim (default 128).
        embedding_dim: Output embedding dim (default 64).
        num_layers: GraphSAGE layers (default 3).
        num_directions: Output directions (default 4).
        dropout: Dropout rate (default 0.1).
    """

    def __init__(
        self,
        in_channels: int = 8,
        hidden_channels: int = 128,
        embedding_dim: int = 64,
        num_layers: int = 3,
        num_directions: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = GraphSAGEEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=embedding_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.routing_head = GNNRoutingHead(
            embedding_dim=embedding_dim,
            num_directions=num_directions,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        current_node_indices: torch.Tensor,
        dst_node_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Full forward pass: encode graph → predict routing directions.

        Args:
            x: Node features [num_nodes, in_channels].
            edge_index: Edge indices [2, num_edges].
            current_node_indices: Indices of current nodes [batch].
            dst_node_indices: Indices of destination nodes [batch].

        Returns:
            Direction logits [batch, num_directions].
        """
        # Encode full graph
        embeddings = self.encoder(x, edge_index)  # [num_nodes, embedding_dim]

        # Extract embeddings for specific nodes
        current_emb = embeddings[current_node_indices]  # [batch, embedding_dim]
        dst_emb = embeddings[dst_node_indices]  # [batch, embedding_dim]

        # Predict directions
        logits = self.routing_head(current_emb, dst_emb)  # [batch, num_directions]

        return logits

    def get_embeddings(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Get node embeddings without the routing head (for DQN input).

        Args:
            x: Node features [num_nodes, in_channels].
            edge_index: Edge indices [2, num_edges].

        Returns:
            Node embeddings [num_nodes, embedding_dim].
        """
        return self.encoder(x, edge_index)
