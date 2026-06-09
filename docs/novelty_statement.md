# Novelty Statement — GRL-Torus

> This document articulates the concrete novelty claims of GRL-Torus with
> evidence from the codebase. Intended for the "Key Contributions" subsection
> of the paper's Introduction and for reference during reviewer response.

---

## Key Contributions

### 1. Torus-Aware GraphSAGE with Positional Node Features

Unlike generic GCN approaches that treat the network as an abstract graph, our GraphSAGE encoder incorporates **positional node features** that encode the 2D torus topology natively:

- Features `[4]` and `[5]` are the normalised grid coordinates `(x/(N-1), y/(N-1))`, enabling the GNN to learn wrap-around distance relationships directly.
- Features `[0]–[3]` provide **per-direction** queue occupancies (N/S/E/W), not just an aggregate — giving the encoder fine-grained congestion visibility per output port.
- Feature `[6]` is an EMA-smoothed load signal that captures temporal congestion trends rather than instantaneous snapshots.

**Evidence**: [node.py:L179-L205](file:///c:/Projects/Optical%20Project/src/sim/node.py#L179-L205) — `get_features()` returns `[queue_N, queue_S, queue_E, queue_W, x_norm, y_norm, load_ema, is_failed]`.

**Differentiation from prior work**:
- Li et al. (2019) use aggregate buffer occupancy (1-dim) — we use 4 directional queues.
- Almasan et al. (2022) do not encode positional information — we embed grid coordinates.
- Standard GCN routing papers (e.g., Rusek et al., 2020) use link-level features but lack node spatial encoding.

---

### 2. Dueling DQN + Prioritized Experience Replay for Routing

Our RL agent uses **Dueling DQN** (Wang et al., 2016) architecture that decouples state-value V(s) from per-action advantage A(s,a). This is critical for routing because:

- Many states have similar value (most nodes aren't congested) — the value stream captures this.
- The advantage stream focuses on discriminating between routing directions, which is the actionable decision.

Combined with **Prioritized Experience Replay** (Schaul et al., 2015) using a Sum-Tree data structure:
- High TD-error transitions (surprising routing outcomes) are replayed more frequently.
- Importance sampling weights prevent bias from non-uniform sampling.

**Evidence**: 
- Dueling architecture: [dqn.py:L22-L86](file:///c:/Projects/Optical%20Project/src/models/dqn.py#L22-L86) — `Q(s,a) = V(s) + A(s,a) - mean(A)`.
- Sum-Tree PER: [replay_buffer.py](file:///c:/Projects/Optical%20Project/src/models/replay_buffer.py) — full priority tree implementation.

**Differentiation**:
- Most GNN-routing papers use vanilla DQN or policy gradient — we combine Dueling + Double DQN + PER.
- The Hold action (action 4) is a novel addition to routing RL, allowing the agent to defer routing when all links are congested.

---

### 3. Two-Stage Joint GNN-DQN Training Pipeline

Our training pipeline addresses the **cold-start problem** in GNN-RL for routing:

1. **Stage 1 (Supervised)**: Pre-train GraphSAGE on Dijkstra shortest-path labels → provides warm-start embeddings that already encode graph structure and distance.
2. **Stage 2 (RL)**: Fine-tune with DQN on simulated traffic → adapts embeddings and policy to dynamic congestion patterns.

We support two variants for ablation:
- **Frozen GNN**: Only DQN weights update (faster, more stable).
- **Joint training**: Both GNN and DQN update (potentially better adaptation, separate learning rates).

**Evidence**: [training.py:L567-L972](file:///c:/Projects/Optical%20Project/src/models/training.py#L567-L972) — `train_dqn()` with `freeze_gnn` parameter.

**Differentiation**:
- Pure RL routing (e.g., Mukhutdinov et al., 2019) starts from random embeddings — our supervised pretraining provides 64% baseline accuracy before RL even begins.
- Pure supervised GNN routing cannot adapt to dynamic congestion — our RL stage adds online adaptation.

---

### 4. Systematic Fault-Resilient Evaluation

We conduct experiments under both healthy and degraded network conditions:

- **Healthy**: 0% link failures — tests routing efficiency under nominal operation.
- **Degraded**: 10% random link failures — tests resilience and adaptive rerouting.

This is a significant departure from most GNN routing papers that assume perfect topology and do not evaluate fault tolerance.

**Evidence**: 
- Failure injection: [torus_graph.py `inject_link_failures()`](file:///c:/Projects/Optical%20Project/src/sim/torus_graph.py) — random uniform link failure.
- Experiment grid: [config.yaml:L42](file:///c:/Projects/Optical%20Project/conf/config.yaml#L42) — `failure_rates: [0.0, 0.1]`.
- Runner handles failures: [runner.py:L125-L134](file:///c:/Projects/Optical%20Project/src/experiments/runner.py#L125-L134).
- GRL router gracefully handles blocked links via Q-value fallback: [grl_router.py:L224-L238](file:///c:/Projects/Optical%20Project/src/routers/grl_router.py#L224-L238).

---

## Claims We Do NOT Make

To maintain scientific integrity, the following claims should be **removed** from the manuscript if present:

| ❌ Unsupported Claim | Reason |
|----------------------|--------|
| "Energy-efficient routing" | No energy model is implemented |
| "Scales to arbitrary topologies" | Only 2D torus is tested |
| "Real-time routing" | Inference timing not benchmarked |
| "Outperforms all baselines" | Must be qualified per metric and traffic pattern |

---

## Suggested Introduction Bullet Points

For the revised manuscript's "Key Contributions" section:

1. We propose **GRL-Torus**, a novel routing framework for 2D torus optical interconnects that combines GraphSAGE spatial encoding with Dueling DQN policy learning.

2. We design a **torus-aware 8-dimensional node feature vector** that encodes per-direction queue occupancies, normalised grid coordinates, and temporal load trends, enabling the GNN to reason about wrap-around topology and directional congestion.

3. We implement a **two-stage training pipeline** with supervised Dijkstra pretraining followed by reinforcement learning fine-tuning, supporting both frozen-GNN and joint-training ablation variants.

4. We conduct **systematic evaluation** across 3 traffic patterns, 2 failure scenarios, and 5 routing algorithms with 5 independent seeds per configuration (300 total experiments), including Mann-Whitney U statistical significance tests.
