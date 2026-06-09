"""
Dueling DQN Q-network for routing action selection.

Implements the Dueling architecture (Wang et al., 2016) that decouples
state-value V(s) from action-advantage A(s,a):
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))

Input: 69-dim state vector = 64-dim GNN embedding + 5-dim packet features
    [src_x_norm, src_y_norm, dst_x_norm, dst_y_norm, hops_so_far_norm]

Output: 5 Q-values for actions [North, South, East, West, Hold]
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class DuelingDQN(nn.Module):
    """Dueling DQN Q-network for torus routing.

    Architecture:
        Shared feature extraction:
            Linear(state_dim, 256) → ReLU → Linear(256, 128) → ReLU

        Value stream:
            Linear(128, 1) → V(s)

        Advantage stream:
            Linear(128, action_dim) → A(s,a)

        Output:
            Q(s,a) = V(s) + A(s,a) - mean(A(s,:))

    Args:
        state_dim: Input state dimension (default 69 = 64 GNN + 5 packet).
        action_dim: Number of actions (default 5: N/S/E/W/Hold).
        hidden_dims: Hidden layer sizes (default [256, 128]).
    """

    def __init__(
        self,
        state_dim: int = 69,
        action_dim: int = 5,
        hidden_dims: Tuple[int, ...] = (256, 128),
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature extraction
        layers = []
        prev_dim = state_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU()])
            prev_dim = dim
        self.feature_net = nn.Sequential(*layers)

        # Value stream: V(s) — single scalar
        self.value_stream = nn.Linear(prev_dim, 1)

        # Advantage stream: A(s, a) — one value per action
        self.advantage_stream = nn.Linear(prev_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for all actions.

        Args:
            state: State vector [batch, state_dim].

        Returns:
            Q-values [batch, action_dim].
        """
        features = self.feature_net(state)

        value = self.value_stream(features)  # [batch, 1]
        advantage = self.advantage_stream(features)  # [batch, action_dim]

        # Dueling combination: Q = V + (A - mean(A))
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)

        return q_values

    def select_action(
        self,
        state: torch.Tensor,
        epsilon: float = 0.0,
    ) -> int:
        """Select action using epsilon-greedy policy.

        Args:
            state: State vector [1, state_dim] or [state_dim].
            epsilon: Exploration rate [0, 1].

        Returns:
            Selected action index.
        """
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.action_dim, (1,)).item()

        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            q_values = self.forward(state)
            return q_values.argmax(dim=-1).item()
