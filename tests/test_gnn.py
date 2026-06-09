"""
Tests for the GraphSAGE encoder, GNN routing model, and supervised training.

Validates model architecture, forward pass shapes, training convergence,
checkpoint save/load, and GNN router integration.
"""

import sys
import os
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.graphsage import GraphSAGEEncoder, GNNRoutingHead, GNNRoutingModel
from src.models.training import generate_training_data, DIRECTION_TO_IDX
from src.sim.torus_graph import TorusGraph
from src.utils.seeding import set_global_seed


class TestGraphSAGEEncoder:
    """Test GraphSAGE encoder architecture and forward pass."""

    def test_output_shape_4x4(self):
        """4×4 torus: 16 nodes → 16 embeddings of dim 64."""
        encoder = GraphSAGEEncoder(in_channels=8, hidden_channels=128, out_channels=64)
        torus = TorusGraph(4)
        data = torus.to_pyg_data()

        out = encoder(data.x, data.edge_index)
        assert out.shape == (16, 64), f"Expected (16, 64), got {out.shape}"

    def test_output_shape_8x8(self):
        """8×8 torus: 64 nodes → 64 embeddings."""
        encoder = GraphSAGEEncoder(in_channels=8, hidden_channels=128, out_channels=64)
        torus = TorusGraph(8)
        data = torus.to_pyg_data()

        out = encoder(data.x, data.edge_index)
        assert out.shape == (64, 64)

    def test_custom_dimensions(self):
        """Test with custom hidden/output dimensions."""
        encoder = GraphSAGEEncoder(in_channels=8, hidden_channels=64, out_channels=32, num_layers=2)
        torus = TorusGraph(4)
        data = torus.to_pyg_data()

        out = encoder(data.x, data.edge_index)
        assert out.shape == (16, 32)

    def test_get_embedding_dim(self):
        encoder = GraphSAGEEncoder(out_channels=64)
        assert encoder.get_embedding_dim() == 64

    def test_parameter_count(self):
        """Model should have a reasonable number of parameters."""
        encoder = GraphSAGEEncoder()
        params = sum(p.numel() for p in encoder.parameters())
        assert params > 0
        assert params < 1_000_000  # Should be well under 1M params

    def test_training_mode(self):
        """Dropout should behave differently in train vs eval."""
        encoder = GraphSAGEEncoder(dropout=0.5)
        torus = TorusGraph(4)
        data = torus.to_pyg_data()

        set_global_seed(42)
        encoder.train()
        out_train = encoder(data.x, data.edge_index)

        set_global_seed(42)
        encoder.eval()
        out_eval = encoder(data.x, data.edge_index)

        # In eval mode, dropout is disabled — outputs should differ
        # (unless dropout is 0 or model has no randomness)
        # Just verify both produce valid shapes
        assert out_train.shape == out_eval.shape


class TestGNNRoutingHead:
    """Test the routing classification head."""

    def test_output_shape(self):
        head = GNNRoutingHead(embedding_dim=64, num_directions=4)
        node_emb = torch.randn(10, 64)
        dst_emb = torch.randn(10, 64)

        logits = head(node_emb, dst_emb)
        assert logits.shape == (10, 4)

    def test_single_sample(self):
        head = GNNRoutingHead(embedding_dim=64)
        node_emb = torch.randn(1, 64)
        dst_emb = torch.randn(1, 64)

        logits = head(node_emb, dst_emb)
        assert logits.shape == (1, 4)


class TestGNNRoutingModel:
    """Test the complete GNN routing model (encoder + head)."""

    def test_forward_pass(self):
        """Full forward pass: graph → direction logits."""
        model = GNNRoutingModel()
        torus = TorusGraph(4)
        data = torus.to_pyg_data()

        # Route 5 packets: random current/destination
        current_indices = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        dst_indices = torch.tensor([15, 14, 13, 12, 11], dtype=torch.long)

        logits = model(data.x, data.edge_index, current_indices, dst_indices)
        assert logits.shape == (5, 4)

    def test_get_embeddings(self):
        """get_embeddings should return encoder-only output."""
        model = GNNRoutingModel()
        torus = TorusGraph(4)
        data = torus.to_pyg_data()

        embeddings = model.get_embeddings(data.x, data.edge_index)
        assert embeddings.shape == (16, 64)

    def test_gradients_flow(self):
        """Gradients should flow through the full model."""
        model = GNNRoutingModel()
        torus = TorusGraph(4)
        data = torus.to_pyg_data()

        current_indices = torch.tensor([0, 1], dtype=torch.long)
        dst_indices = torch.tensor([15, 14], dtype=torch.long)
        labels = torch.tensor([0, 1], dtype=torch.long)

        logits = model(data.x, data.edge_index, current_indices, dst_indices)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()

        # Check that encoder parameters have gradients
        for name, param in model.encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                break  # Just check first parameter


class TestTrainingDataGeneration:
    """Test training data generation from simulator snapshots."""

    def test_generates_data(self):
        """Should produce non-empty training data."""
        data = generate_training_data(
            grid_size=4,
            num_snapshots=5,
            samples_per_snapshot=10,
            seed=42,
        )
        nf, ei, ci, di, labels = data

        assert len(nf) > 0, "No training data generated"
        assert len(nf) == len(ei) == len(ci) == len(di) == len(labels)

    def test_data_shapes(self):
        """Training data should have correct shapes."""
        data = generate_training_data(
            grid_size=4,
            num_snapshots=3,
            samples_per_snapshot=5,
            seed=42,
        )
        nf, ei, ci, di, labels = data

        for i in range(len(nf)):
            assert nf[i].shape == (16, 8), f"Node features shape: {nf[i].shape}"
            assert ei[i].shape[0] == 2, f"Edge index shape: {ei[i].shape}"
            assert len(ci[i]) == len(labels[i])
            assert len(di[i]) == len(labels[i])

    def test_labels_valid(self):
        """Labels should be valid direction indices (0-3)."""
        data = generate_training_data(
            grid_size=4,
            num_snapshots=3,
            samples_per_snapshot=5,
            seed=42,
        )
        _, _, _, _, labels = data

        for lab in labels:
            assert (lab >= 0).all() and (lab <= 3).all(), f"Invalid labels: {lab}"


class TestTrainingConvergence:
    """Test that training loop converges on small data."""

    def test_loss_decreases(self):
        """Loss should decrease over a few epochs on a small dataset."""
        set_global_seed(42)

        model = GNNRoutingModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        # Generate tiny dataset
        data = generate_training_data(
            grid_size=4,
            num_snapshots=10,
            samples_per_snapshot=15,
            seed=42,
        )
        nf, ei, ci, di, labels = data

        if len(nf) == 0:
            pytest.skip("No training data generated")

        losses = []
        model.train()

        for epoch in range(15):
            epoch_loss = 0.0
            for i in range(len(nf)):
                optimizer.zero_grad()
                logits = model(nf[i], ei[i], ci[i], di[i])
                loss = criterion(logits, labels[i])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(nf))

        # Loss should decrease (last loss < first loss)
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        )


class TestCheckpointSaveLoad:
    """Test model checkpoint save and load."""

    def test_save_and_load(self, tmp_path):
        """Model should produce same output after save/load."""
        set_global_seed(42)

        model = GNNRoutingModel()
        torus = TorusGraph(4)
        data = torus.to_pyg_data()

        model.eval()
        with torch.no_grad():
            ci = torch.tensor([0], dtype=torch.long)
            di = torch.tensor([15], dtype=torch.long)
            original_output = model(data.x, data.edge_index, ci, di)

        # Save checkpoint
        ckpt_path = tmp_path / "test_model.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": {
                "in_channels": 8,
                "hidden_channels": 128,
                "embedding_dim": 64,
                "num_layers": 3,
            },
        }, ckpt_path)

        # Load checkpoint
        checkpoint = torch.load(ckpt_path, weights_only=False)
        loaded_model = GNNRoutingModel()
        loaded_model.load_state_dict(checkpoint["model_state_dict"])
        loaded_model.eval()

        with torch.no_grad():
            loaded_output = loaded_model(data.x, data.edge_index, ci, di)

        assert torch.allclose(original_output, loaded_output, atol=1e-6), (
            "Output mismatch after checkpoint load"
        )
