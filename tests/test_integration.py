"""
Integration tests for ML routers (GNN and GRL) within the SimPy simulator.
"""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.graphsage import GNNRoutingModel
from src.models.dqn import DuelingDQN
from src.routers.gnn_router import GNNRouter
from src.routers.grl_router import GRLRouter
from src.sim.torus_graph import TorusGraph
from src.sim.traffic import TrafficGenerator
from src.sim.simulator import TorusSimulator
from src.utils.seeding import set_global_seed


def test_gnn_router_simulation():
    """Test that GNN router runs successfully in the simulator."""
    set_global_seed(42)
    grid_size = 4
    
    # Create dummy untrained GNN model
    model = GNNRoutingModel(
        in_channels=8,
        hidden_channels=16,
        embedding_dim=16,
        num_layers=2,
        num_directions=4,
    )
    
    # Initialize router with dummy model
    router = GNNRouter(model=model, device="cpu")
    
    torus = TorusGraph(n=grid_size)
    traffic = TrafficGenerator(pattern="uniform", grid_size=grid_size, injection_rate=0.01, seed=42)
    sim = TorusSimulator(torus=torus, router=router, duration_ns=1000)
    
    results = sim.run(traffic)
    
    # Check that simulation completes and produces metrics
    assert results.duration_ns == 1000
    assert len(results.packets) > 0
    # Since model is untrained, packets might drop or loop, but the simulator shouldn't crash
    metrics = results.to_dict()
    assert "avg_latency_ns" in metrics


def test_grl_router_simulation():
    """Test that GRL router runs successfully in the simulator."""
    set_global_seed(42)
    grid_size = 4
    
    # Create dummy untrained GNN encoder
    gnn_model = GNNRoutingModel(
        in_channels=8,
        hidden_channels=16,
        embedding_dim=16,
        num_layers=2,
    )
    encoder = gnn_model.encoder
    
    # Create dummy untrained DQN network
    dqn_net = DuelingDQN(
        state_dim=16 + 5, # embedding_dim + 5 packet features
        action_dim=5,
        hidden_dims=(32, 16),
    )
    
    # Initialize router with dummy models
    router = GRLRouter(gnn_encoder=encoder, dqn_network=dqn_net, device="cpu", grid_size=grid_size)
    
    torus = TorusGraph(n=grid_size)
    traffic = TrafficGenerator(pattern="hotspot", grid_size=grid_size, injection_rate=0.01, seed=42)
    sim = TorusSimulator(torus=torus, router=router, duration_ns=1000)
    
    results = sim.run(traffic)
    
    assert results.duration_ns == 1000
    assert len(results.packets) > 0
    metrics = results.to_dict()
    assert "throughput_pps" in metrics
