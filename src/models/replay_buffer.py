"""
Prioritised Experience Replay (PER) with Sum-Tree.

Implements priority-weighted sampling for DQN training where transitions
with higher TD-errors are sampled more frequently, improving learning
efficiency. Uses importance-sampling weights to correct the bias.

References:
    Schaul et al., 2015 — "Prioritized Experience Replay"
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import torch


class Transition(NamedTuple):
    """A single experience transition."""

    state: np.ndarray       # [state_dim]
    action: int
    reward: float
    next_state: np.ndarray  # [state_dim]
    done: bool


class SumTree:
    """Binary sum-tree for O(log N) priority-weighted sampling.

    Each leaf stores a priority value. Internal nodes store the sum
    of their children. This allows:
        - O(log N) priority update
        - O(log N) proportional sampling
        - O(1) total priority sum
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write_idx = 0
        self.size = 0

    def _propagate(self, idx: int, change: float) -> None:
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, value: float) -> int:
        """Find the leaf index for a given cumulative value."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])

    @property
    def total_priority(self) -> float:
        """Total sum of all priorities."""
        return self.tree[0]

    @property
    def min_priority(self) -> float:
        """Minimum priority among stored transitions."""
        if self.size == 0:
            return 0.0
        leaf_start = self.capacity - 1
        priorities = self.tree[leaf_start : leaf_start + self.size]
        non_zero = priorities[priorities > 0]
        return float(non_zero.min()) if len(non_zero) > 0 else 1e-6

    def add(self, priority: float, data: Transition) -> None:
        """Add a transition with given priority."""
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data

        self.update(idx, priority)

        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx: int, priority: float) -> None:
        """Update the priority of a leaf node."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, value: float) -> Tuple[int, float, Transition]:
        """Sample a leaf by cumulative value.

        Returns:
            (tree_index, priority, transition_data)
        """
        idx = self._retrieve(0, value)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """Prioritised Experience Replay buffer.

    Samples transitions proportional to their TD-error priority, with
    importance-sampling weights to correct gradient bias.

    Args:
        capacity: Maximum number of transitions to store.
        alpha: Prioritisation exponent [0, 1]. 0 = uniform, 1 = full prioritisation.
        beta_start: Initial importance-sampling exponent.
        beta_end: Final importance-sampling exponent (annealed over training).
        beta_anneal_steps: Number of steps to anneal beta from start to end.
        epsilon: Small constant added to priorities to prevent zero-priority.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_anneal_steps: int = 50_000,
        epsilon: float = 1e-6,
    ):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal_steps = beta_anneal_steps
        self.epsilon = epsilon

        self._max_priority = 1.0
        self._step = 0

    @property
    def beta(self) -> float:
        """Current importance-sampling exponent (annealed linearly)."""
        frac = min(1.0, self._step / max(1, self.beta_anneal_steps))
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    def __len__(self) -> int:
        return self.tree.size

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition with maximum priority.

        New transitions get max priority to ensure they are sampled at
        least once before their priority is updated.
        """
        transition = Transition(state, action, reward, next_state, done)
        priority = self._max_priority ** self.alpha
        self.tree.add(priority, transition)

    def sample(
        self, batch_size: int
    ) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """Sample a batch of transitions proportional to priority.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            (transitions, tree_indices, importance_sampling_weights)
        """
        self._step += 1

        transitions: List[Transition] = []
        indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)

        # Divide the total priority range into equal segments
        segment = self.tree.total_priority / batch_size

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = np.random.uniform(low, high)

            idx, priority, data = self.tree.get(value)
            if data is None:
                # Fallback: re-sample
                value = np.random.uniform(0, self.tree.total_priority)
                idx, priority, data = self.tree.get(value)

            transitions.append(data)
            indices[i] = idx
            priorities[i] = priority

        # Compute importance-sampling weights
        total = self.tree.total_priority
        min_prob = self.tree.min_priority / total if total > 0 else 1e-6
        min_prob = max(min_prob, 1e-8)

        probs = priorities / total
        probs = np.clip(probs, 1e-8, None)

        beta = self.beta
        weights = (self.tree.size * probs) ** (-beta)
        max_weight = (self.tree.size * min_prob) ** (-beta)
        weights = weights / max_weight  # Normalise to [0, 1]

        return transitions, indices, weights.astype(np.float32)

    def update_priorities(
        self, indices: np.ndarray, td_errors: np.ndarray
    ) -> None:
        """Update priorities based on new TD-errors.

        Args:
            indices: Tree indices from sample().
            td_errors: Absolute TD-errors for each transition.
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self._max_priority = max(self._max_priority, priority)
            self.tree.update(int(idx), priority)

    def sample_tensors(
        self, batch_size: int, device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, torch.Tensor]:
        """Sample and return as PyTorch tensors ready for training.

        Returns:
            (states, actions, rewards, next_states, dones, indices, weights)
        """
        transitions, indices, weights = self.sample(batch_size)

        states = torch.tensor(
            np.stack([t.state for t in transitions]), dtype=torch.float32, device=device
        )
        actions = torch.tensor(
            [t.action for t in transitions], dtype=torch.long, device=device
        )
        rewards = torch.tensor(
            [t.reward for t in transitions], dtype=torch.float32, device=device
        )
        next_states = torch.tensor(
            np.stack([t.next_state for t in transitions]), dtype=torch.float32, device=device
        )
        dones = torch.tensor(
            [float(t.done) for t in transitions], dtype=torch.float32, device=device
        )
        is_weights = torch.tensor(weights, dtype=torch.float32, device=device)

        return states, actions, rewards, next_states, dones, indices, is_weights
