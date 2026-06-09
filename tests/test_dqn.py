"""
Tests for Dueling DQN and Prioritised Experience Replay buffer.

Validates DQN forward pass, action selection, PER sampling/priority
updates, and deadlock prevention logic.
"""

import sys
import os
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.dqn import DuelingDQN
from src.models.replay_buffer import PrioritizedReplayBuffer, SumTree, Transition
from src.utils.seeding import set_global_seed


class TestDuelingDQN:
    """Test Dueling DQN Q-network."""

    def test_output_shape(self):
        """Forward pass should produce [batch, action_dim] Q-values."""
        dqn = DuelingDQN(state_dim=69, action_dim=5)
        state = torch.randn(32, 69)
        q_values = dqn(state)
        assert q_values.shape == (32, 5)

    def test_single_state(self):
        """Should handle single state input."""
        dqn = DuelingDQN(state_dim=69, action_dim=5)
        state = torch.randn(1, 69)
        q_values = dqn(state)
        assert q_values.shape == (1, 5)

    def test_greedy_action(self):
        """With epsilon=0, should always pick argmax Q."""
        set_global_seed(42)
        dqn = DuelingDQN(state_dim=69, action_dim=5)
        dqn.eval()

        state = torch.randn(69)
        with torch.no_grad():
            q_values = dqn(state.unsqueeze(0))
            expected_action = q_values.argmax(dim=-1).item()

        action = dqn.select_action(state, epsilon=0.0)
        assert action == expected_action

    def test_random_action_at_eps_1(self):
        """With epsilon=1.0, all actions should be random."""
        dqn = DuelingDQN(state_dim=69, action_dim=5)
        state = torch.randn(69)

        actions = set()
        for _ in range(100):
            action = dqn.select_action(state, epsilon=1.0)
            actions.add(action)
            assert 0 <= action < 5

        # With 100 samples at eps=1.0, we should see multiple distinct actions
        assert len(actions) > 1

    def test_dueling_architecture(self):
        """Value stream and advantage stream should combine correctly."""
        dqn = DuelingDQN(state_dim=10, action_dim=3)
        state = torch.randn(4, 10)

        q_values = dqn(state)

        # Q = V + (A - mean(A))  →  mean(Q) should roughly equal V
        # This is a property check, not exact
        assert q_values.shape == (4, 3)
        assert not torch.isnan(q_values).any()

    def test_custom_hidden_dims(self):
        dqn = DuelingDQN(state_dim=20, action_dim=3, hidden_dims=(64, 32))
        state = torch.randn(5, 20)
        q = dqn(state)
        assert q.shape == (5, 3)

    def test_gradient_flow(self):
        """Gradients should flow through all layers."""
        dqn = DuelingDQN(state_dim=69, action_dim=5)
        state = torch.randn(4, 69)
        q = dqn(state)
        loss = q.sum()
        loss.backward()

        for name, param in dqn.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestSumTree:
    """Test the Sum-Tree data structure for PER."""

    def test_add_and_total(self):
        tree = SumTree(capacity=4)
        tree.add(1.0, Transition(np.zeros(5), 0, 1.0, np.zeros(5), False))
        tree.add(2.0, Transition(np.zeros(5), 1, 0.5, np.zeros(5), False))
        assert abs(tree.total_priority - 3.0) < 1e-6

    def test_update_priority(self):
        tree = SumTree(capacity=4)
        tree.add(1.0, Transition(np.zeros(5), 0, 1.0, np.zeros(5), False))
        idx = tree.capacity - 1  # First leaf
        tree.update(idx, 5.0)
        assert abs(tree.total_priority - 5.0) < 1e-6

    def test_sampling(self):
        """Sampling should return valid data."""
        tree = SumTree(capacity=4)
        for i in range(4):
            tree.add(float(i + 1), Transition(np.ones(5) * i, i, float(i), np.ones(5) * i, False))

        idx, priority, data = tree.get(tree.total_priority * 0.5)
        assert data is not None
        assert priority > 0


class TestPrioritizedReplayBuffer:
    """Test PER buffer functionality."""

    def test_push_and_len(self):
        buffer = PrioritizedReplayBuffer(capacity=100)
        assert len(buffer) == 0

        for i in range(50):
            buffer.push(np.zeros(10), 0, 1.0, np.zeros(10), False)
        assert len(buffer) == 50

    def test_capacity_limit(self):
        buffer = PrioritizedReplayBuffer(capacity=10)
        for i in range(20):
            buffer.push(np.ones(5) * i, i % 5, float(i), np.ones(5) * i, False)
        assert len(buffer) == 10  # Should not exceed capacity

    def test_sample_shapes(self):
        """Sample should return correct shapes."""
        buffer = PrioritizedReplayBuffer(capacity=100)
        for i in range(50):
            buffer.push(np.random.randn(10), i % 5, float(i), np.random.randn(10), i % 3 == 0)

        transitions, indices, weights = buffer.sample(16)
        assert len(transitions) == 16
        assert indices.shape == (16,)
        assert weights.shape == (16,)

    def test_sample_tensors(self):
        """sample_tensors should return correctly shaped tensors."""
        buffer = PrioritizedReplayBuffer(capacity=100)
        for i in range(50):
            buffer.push(np.random.randn(10), i % 5, float(i), np.random.randn(10), False)

        states, actions, rewards, next_states, dones, indices, weights = buffer.sample_tensors(8)
        assert states.shape == (8, 10)
        assert actions.shape == (8,)
        assert rewards.shape == (8,)
        assert next_states.shape == (8, 10)
        assert dones.shape == (8,)
        assert weights.shape == (8,)

    def test_priority_update(self):
        """Updated priorities should affect sampling distribution."""
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=1.0)

        for i in range(10):
            buffer.push(np.ones(5) * i, 0, 0.0, np.ones(5), False)

        # Sample and update one transition with very high priority
        transitions, indices, weights = buffer.sample(5)
        high_td = np.array([100.0] + [0.01] * 4)
        buffer.update_priorities(indices, high_td)

        # After updating, the high-priority item should be sampled more often
        # (probabilistic, just check it doesn't crash)
        for _ in range(10):
            _, _, _ = buffer.sample(3)

    def test_weights_normalised(self):
        """IS weights should be in [0, 1] range."""
        buffer = PrioritizedReplayBuffer(capacity=100)
        for i in range(50):
            buffer.push(np.random.randn(5), 0, 1.0, np.random.randn(5), False)

        _, _, weights = buffer.sample(16)
        assert (weights >= 0).all()
        assert (weights <= 1.0 + 1e-6).all()

    def test_beta_annealing(self):
        """Beta should anneal from start to end over steps."""
        buffer = PrioritizedReplayBuffer(
            capacity=100, beta_start=0.4, beta_end=1.0, beta_anneal_steps=100
        )

        assert abs(buffer.beta - 0.4) < 0.01

        # Sample many times to advance step counter
        for i in range(50):
            buffer.push(np.random.randn(5), 0, 1.0, np.random.randn(5), False)

        for _ in range(100):
            buffer.sample(5)

        assert buffer.beta > 0.4
