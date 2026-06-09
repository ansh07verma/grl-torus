# Model Architecture Specification

> This document provides the exact architecture details of every ML component in GRL-Torus,
> verified line-by-line against the source code. All hyperparameters, layer dimensions,
> and design decisions are cross-referenced with their implementation files and config YAMLs.

---

## 1. Node Feature Vector (8 Dimensions)

Each torus node produces an 8-dimensional feature vector via [`TorusNode.get_features()`](file:///c:/Projects/Optical%20Project/src/sim/node.py#L179-L205):

| Index | Name | Range | Description | Source |
|-------|------|-------|-------------|--------|
| 0 | `queue_N_norm` | [0, 1] | North queue occupancy / `buffer_depth` | [node.py:L195](file:///c:/Projects/Optical%20Project/src/sim/node.py#L195) |
| 1 | `queue_S_norm` | [0, 1] | South queue occupancy / `buffer_depth` | [node.py:L196](file:///c:/Projects/Optical%20Project/src/sim/node.py#L196) |
| 2 | `queue_E_norm` | [0, 1] | East queue occupancy / `buffer_depth` | [node.py:L197](file:///c:/Projects/Optical%20Project/src/sim/node.py#L197) |
| 3 | `queue_W_norm` | [0, 1] | West queue occupancy / `buffer_depth` | [node.py:L198](file:///c:/Projects/Optical%20Project/src/sim/node.py#L198) |
| 4 | `x_norm` | [0, 1] | x / (grid_size − 1) | [node.py:L199](file:///c:/Projects/Optical%20Project/src/sim/node.py#L199) |
| 5 | `y_norm` | [0, 1] | y / (grid_size − 1) | [node.py:L200](file:///c:/Projects/Optical%20Project/src/sim/node.py#L200) |
| 6 | `load_ema` | [0, 1] | Exponential moving average of total queue load (α = 0.1) | [node.py:L201](file:///c:/Projects/Optical%20Project/src/sim/node.py#L201), [L174-L177](file:///c:/Projects/Optical%20Project/src/sim/node.py#L174-L177) |
| 7 | `is_failed` | {0, 1} | 1.0 if node is failed, 0.0 otherwise | [node.py:L202](file:///c:/Projects/Optical%20Project/src/sim/node.py#L202) |

> [!WARNING]
> **Correction**: Earlier documentation described the feature vector as `[queue_length, link_utilisation_N/S/E/W, is_destination, x, y]`. This is **incorrect**. The actual features are per-direction queue occupancies, normalised grid coordinates, EMA load, and failure flag as listed above.

---

## 2. Edge Feature Vector (3 Dimensions)

Each directed link produces a 3-dimensional feature via [`TorusLink.get_features()`](file:///c:/Projects/Optical%20Project/src/sim/link.py#L118-L133):

| Index | Name | Range | Description |
|-------|------|-------|-------------|
| 0 | `utilisation` | [0, 1] | Current link utilisation (EMA-smoothed) |
| 1 | `failure_prob` | [0, 1] | Link failure probability |
| 2 | `propagation_delay_norm` | [0, ~1] | Propagation delay / 10 ns |

---

## 3. GraphSAGE Encoder

**Reference**: [graphsage.py](file:///c:/Projects/Optical%20Project/src/models/graphsage.py), [gnn_supervised.yaml](file:///c:/Projects/Optical%20Project/conf/training/gnn_supervised.yaml)

| Parameter | Value | Source |
|-----------|-------|--------|
| Architecture | GraphSAGE (Hamilton et al., 2017) | [graphsage.py:L1-L16](file:///c:/Projects/Optical%20Project/src/models/graphsage.py#L1-L16) |
| Implementation | PyTorch Geometric `SAGEConv` | [graphsage.py:L25](file:///c:/Projects/Optical%20Project/src/models/graphsage.py#L25) |
| Number of layers | 3 | [graphsage.py:L45](file:///c:/Projects/Optical%20Project/src/models/graphsage.py#L45), [gnn_supervised.yaml:L6](file:///c:/Projects/Optical%20Project/conf/training/gnn_supervised.yaml#L6) |
| Input dimension | 8 (node features) | [graphsage.py:L42](file:///c:/Projects/Optical%20Project/src/models/graphsage.py#L42), [gnn_supervised.yaml:L3](file:///c:/Projects/Optical%20Project/conf/training/gnn_supervised.yaml#L3) |
| Hidden dimension | 128 | [graphsage.py:L43](file:///c:/Projects/Optical%20Project/src/models/graphsage.py#L43), [gnn_supervised.yaml:L4](file:///c:/Projects/Optical%20Project/conf/training/gnn_supervised.yaml#L4) |
| Output embedding dim | 64 | [graphsage.py:L44](file:///c:/Projects/Optical%20Project/src/models/graphsage.py#L44), [gnn_supervised.yaml:L5](file:///c:/Projects/Optical%20Project/conf/training/gnn_supervised.yaml#L5) |
| Aggregation | Mean | [graphsage.py:L47](file:///c:/Projects/Optical%20Project/src/models/graphsage.py#L47) |
| Dropout | 0.1 (between layers, not on final) | [graphsage.py:L46](file:///c:/Projects/Optical%20Project/src/models/graphsage.py#L46), [gnn_supervised.yaml:L7](file:///c:/Projects/Optical%20Project/conf/training/gnn_supervised.yaml#L7) |
| BatchNorm | After each layer except final | [graphsage.py:L64,L71](file:///c:/Projects/Optical%20Project/src/models/graphsage.py#L64) |
| Input projection | Layer 1 only (`project=True`) | [graphsage.py:L62](file:///c:/Projects/Optical%20Project/src/models/graphsage.py#L62) |
| Activation | ReLU (between layers, not on final) | [graphsage.py:L97](file:///c:/Projects/Optical%20Project/src/models/graphsage.py#L97) |

**Layer-by-layer architecture**:

```
Layer 1: SAGEConv(8 → 128, mean, project=True) → BatchNorm(128) → ReLU → Dropout(0.1)
Layer 2: SAGEConv(128 → 128, mean)             → BatchNorm(128) → ReLU → Dropout(0.1)
Layer 3: SAGEConv(128 → 64, mean)              [no norm/activation on output]
```

### Supervised Training (GNN Pretraining)

| Parameter | Value | Source |
|-----------|-------|--------|
| Loss function | CrossEntropyLoss | [training.py:L303](file:///c:/Projects/Optical%20Project/src/models/training.py#L303) |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-4) | [gnn_supervised.yaml:L8-L9](file:///c:/Projects/Optical%20Project/conf/training/gnn_supervised.yaml#L8-L9) |
| LR schedule | CosineAnnealingLR | [training.py:L301](file:///c:/Projects/Optical%20Project/src/models/training.py#L301) |
| Epochs | 100 (early stopping, patience=10) | [gnn_supervised.yaml:L10,L12](file:///c:/Projects/Optical%20Project/conf/training/gnn_supervised.yaml#L10) |
| Batch size | 32 | [gnn_supervised.yaml:L11](file:///c:/Projects/Optical%20Project/conf/training/gnn_supervised.yaml#L11) |
| Train/Val/Test split | 80% / 10% / 10% | [gnn_supervised.yaml:L14-L16](file:///c:/Projects/Optical%20Project/conf/training/gnn_supervised.yaml#L14-L16) |
| Labels | Dijkstra shortest-path optimal next-hop direction | [training.py data generation](file:///c:/Projects/Optical%20Project/src/models/training.py) |
| Training episodes | 1000 per topology | [gnn_supervised.yaml:L13](file:///c:/Projects/Optical%20Project/conf/training/gnn_supervised.yaml#L13) |

### Routing Head (Supervised)

| Parameter | Value | Source |
|-----------|-------|--------|
| Input | `[current_node_emb ∥ dst_node_emb]` = 128-dim | [graphsage.py:L126](file:///c:/Projects/Optical%20Project/src/models/graphsage.py#L126) |
| Architecture | Linear(128→64) → ReLU → Dropout(0.1) → Linear(64→4) | [graphsage.py:L125-L130](file:///c:/Projects/Optical%20Project/src/models/graphsage.py#L125-L130) |
| Output | 4 logits (N, S, E, W) | [graphsage.py:L129](file:///c:/Projects/Optical%20Project/src/models/graphsage.py#L129) |

---

## 4. Dueling DQN Q-Network

**Reference**: [dqn.py](file:///c:/Projects/Optical%20Project/src/models/dqn.py), [dqn.yaml](file:///c:/Projects/Optical%20Project/conf/training/dqn.yaml)

| Parameter | Value | Source |
|-----------|-------|--------|
| Architecture | Dueling DQN (Wang et al., 2016) | [dqn.py:L1-L11](file:///c:/Projects/Optical%20Project/src/models/dqn.py#L1-L11) |
| State dimension | 69 = 64 (GNN embedding) + 5 (packet features) | [dqn.py:L46](file:///c:/Projects/Optical%20Project/src/models/dqn.py#L46), [dqn.yaml:L3](file:///c:/Projects/Optical%20Project/conf/training/dqn.yaml#L3) |
| Action dimension | 5 (North, South, East, West, Hold) | [dqn.py:L47](file:///c:/Projects/Optical%20Project/src/models/dqn.py#L47), [dqn.yaml:L4](file:///c:/Projects/Optical%20Project/conf/training/dqn.yaml#L4) |
| Hidden layers | [256, 128] | [dqn.py:L48](file:///c:/Projects/Optical%20Project/src/models/dqn.py#L48), [dqn.yaml:L5](file:///c:/Projects/Optical%20Project/conf/training/dqn.yaml#L5) |
| Activation | ReLU | [dqn.py:L59](file:///c:/Projects/Optical%20Project/src/models/dqn.py#L59) |

**Packet features** (5 dimensions, appended to GNN embedding):

| Index | Name | Range | Source |
|-------|------|-------|--------|
| 0 | `src_x_norm` | [0, 1] | [grl_router.py:L159](file:///c:/Projects/Optical%20Project/src/routers/grl_router.py#L159) |
| 1 | `src_y_norm` | [0, 1] | [grl_router.py:L160](file:///c:/Projects/Optical%20Project/src/routers/grl_router.py#L160) |
| 2 | `dst_x_norm` | [0, 1] | [grl_router.py:L161](file:///c:/Projects/Optical%20Project/src/routers/grl_router.py#L161) |
| 3 | `dst_y_norm` | [0, 1] | [grl_router.py:L162](file:///c:/Projects/Optical%20Project/src/routers/grl_router.py#L162) |
| 4 | `hops_norm` | [0, 1] | [grl_router.py:L163](file:///c:/Projects/Optical%20Project/src/routers/grl_router.py#L163) — `min(hops / 4N, 1.0)` |

**Dueling architecture**:

```
Shared:       Linear(69→256) → ReLU → Linear(256→128) → ReLU
Value stream: Linear(128→1)           → V(s)
Adv. stream:  Linear(128→5)           → A(s,a)
Output:       Q(s,a) = V(s) + A(s,a) − mean_a(A(s,:))
```

### DQN Training (Reinforcement Learning)

| Parameter | Value | Source |
|-----------|-------|--------|
| Algorithm | Double DQN + Dueling + PER | [training.py:L870-L877](file:///c:/Projects/Optical%20Project/src/models/training.py#L870-L877) |
| Learning rate | 1e-3 (DQN), 1e-4 (GNN in joint mode) | [dqn.yaml:L6,L25](file:///c:/Projects/Optical%20Project/conf/training/dqn.yaml#L6) |
| Discount factor γ | 0.99 | [dqn.yaml:L7](file:///c:/Projects/Optical%20Project/conf/training/dqn.yaml#L7) |
| Loss function | Huber loss (weighted by IS weights) | [training.py:L879](file:///c:/Projects/Optical%20Project/src/models/training.py#L879) |
| Target network | Hard update every 500 steps | [dqn.yaml:L10](file:///c:/Projects/Optical%20Project/conf/training/dqn.yaml#L10) |
| Exploration | ε-greedy, linear decay 1.0 → 0.05 over 10K steps | [dqn.yaml:L11-L13](file:///c:/Projects/Optical%20Project/conf/training/dqn.yaml#L11-L13) |
| Replay buffer | Prioritized (Sum-Tree), capacity 100K | [dqn.yaml:L8](file:///c:/Projects/Optical%20Project/conf/training/dqn.yaml#L8) |
| PER α | 0.6 | [dqn.yaml:L20](file:///c:/Projects/Optical%20Project/conf/training/dqn.yaml#L20) |
| PER β anneal | 0.4 → 1.0 over 50K steps | [dqn.yaml:L21-L23](file:///c:/Projects/Optical%20Project/conf/training/dqn.yaml#L21-L23) |
| Gradient clipping | max_norm = 1.0 (DQN), 0.5 (GNN joint) | [training.py:L884-L886](file:///c:/Projects/Optical%20Project/src/models/training.py#L884-L886) |
| Episodes | 2000 | [dqn.yaml:L14](file:///c:/Projects/Optical%20Project/conf/training/dqn.yaml#L14) |
| Max consecutive holds | 3 (then forced deflection) | [dqn.yaml:L15](file:///c:/Projects/Optical%20Project/conf/training/dqn.yaml#L15) |
| Batch size | 64 | [dqn.yaml:L9](file:///c:/Projects/Optical%20Project/conf/training/dqn.yaml#L9) |

### Reward Function

Defined in [`_compute_reward()`](file:///c:/Projects/Optical%20Project/src/models/training.py#L484-L524):

```
R(s, a) = +1.0 · 𝟙[delivered] − α · 𝟙[dropped] − delay_ns/100 + β · (util_prev − util_curr)
```

| Coefficient | Value | Description | Source |
|-------------|-------|-------------|--------|
| Delivery bonus | +1.0 | Reward for reaching destination | [training.py:L512](file:///c:/Projects/Optical%20Project/src/models/training.py#L512) |
| α (drop penalty) | 10.0 | Penalty for packet drop | [dqn.yaml:L17](file:///c:/Projects/Optical%20Project/conf/training/dqn.yaml#L17) |
| β (utilisation) | 0.5 | Reward for reducing link congestion | [dqn.yaml:L18](file:///c:/Projects/Optical%20Project/conf/training/dqn.yaml#L18) |
| Latency penalty | delay_ns / 100 | Normalised hop delay penalty | [training.py:L518](file:///c:/Projects/Optical%20Project/src/models/training.py#L518) |

---

## 5. GRL Router (Inference Pipeline)

**Reference**: [grl_router.py](file:///c:/Projects/Optical%20Project/src/routers/grl_router.py)

The GRL router combines the GNN encoder and DQN agent at inference time:

1. **Embed graph**: Run GraphSAGE forward pass on all node features → 64-dim embeddings per node. Cached per simulation tick. ([L119-L134](file:///c:/Projects/Optical%20Project/src/routers/grl_router.py#L119-L134))
2. **Build state**: Concatenate current node's embedding (64) with packet features (5) → 69-dim state vector. ([L136-L171](file:///c:/Projects/Optical%20Project/src/routers/grl_router.py#L136-L171))
3. **Select action**: DQN forward pass → Q-values for 5 actions → greedy argmax (ε=0 at inference). ([L195-L196](file:///c:/Projects/Optical%20Project/src/routers/grl_router.py#L195-L196))
4. **Deadlock prevention**: If action is HOLD and hold counter > 3, force deflection to a random active neighbour. ([L198-L213](file:///c:/Projects/Optical%20Project/src/routers/grl_router.py#L198-L213))
5. **Fallback**: If selected direction is blocked (failed link/node), iterate Q-values in descending order to find the best available direction. ([L224-L238](file:///c:/Projects/Optical%20Project/src/routers/grl_router.py#L224-L238))

---

## 6. Model Parameter Counts

For a 4×4 torus (16 nodes, 64 directed edges):

| Component | Parameters | Computed from |
|-----------|-----------|---------------|
| GraphSAGE (3-layer) | ~83K | SAGEConv(8→128) + SAGEConv(128→128) + SAGEConv(128→64) + BatchNorms |
| Routing Head | ~8.5K | Linear(128→64) + Linear(64→4) |
| Dueling DQN | ~51K | Linear(69→256) + Linear(256→128) + Linear(128→1) + Linear(128→5) |
| **Total GRL system** | **~142K** | GNN encoder + DQN (head not used at RL inference) |

---

## 7. Training Modes

### Mode A: Frozen GNN (Default)

1. Pre-train GNN encoder with supervised Dijkstra labels (Phase 2).
2. Freeze all GNN parameters (`requires_grad = False`).
3. Train only DQN weights with RL + PER.

### Mode B: Joint Training

1. Pre-train GNN encoder with supervised Dijkstra labels (Phase 2).
2. Train both GNN and DQN together.
3. GNN uses a lower learning rate (1e-4 vs 1e-3) to preserve pre-trained features.
4. Separate gradient clipping: 0.5 for GNN, 1.0 for DQN.
5. Re-compute GNN embeddings every 50 steps to track encoder updates.
