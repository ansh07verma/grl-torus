# Simulation Setup — Parameter Reference

> All simulation parameters used in GRL-Torus experiments, cross-referenced
> against source code and configuration files. This document is designed for
> direct inclusion in the paper's "Experimental Setup" section.

---

## 1. Simulator Engine

| Parameter | Value | Source |
|-----------|-------|--------|
| Engine | SimPy discrete-event simulation | [simulator.py](file:///c:/Projects/Optical%20Project/src/sim/simulator.py) |
| Time unit | Nanoseconds (ns) | Throughout codebase |
| Simulation duration | 1,000,000 ns (1 ms) | [config.yaml:L8](file:///c:/Projects/Optical%20Project/conf/config.yaml#L8) |
| Warmup period | 100,000 ns (100 µs) | [config.yaml:L9](file:///c:/Projects/Optical%20Project/conf/config.yaml#L9) |
| Max hops per packet | 4 × N (auto) | [config.yaml:L10](file:///c:/Projects/Optical%20Project/conf/config.yaml#L10) |
| Queue timeout | 50,000 ns | [config.yaml:L11](file:///c:/Projects/Optical%20Project/conf/config.yaml#L11) |

> [!NOTE]
> The warmup period ensures that metrics are collected only after the network reaches steady-state. Packets injected during warmup are simulated but their latency/throughput are excluded from final metrics.

---

## 2. Network Topology

| Parameter | Value | Source |
|-----------|-------|--------|
| Topology | 2D Torus (wrap-around mesh) | [torus_graph.py](file:///c:/Projects/Optical%20Project/src/sim/torus_graph.py) |
| Grid sizes | 4×4 (16 nodes), 8×8 (64 nodes) | [config.yaml:L39](file:///c:/Projects/Optical%20Project/conf/config.yaml#L39) |
| Node degree | 4 (bidirectional: N, S, E, W) | [torus_graph.py](file:///c:/Projects/Optical%20Project/src/sim/torus_graph.py) |
| Directed links per grid | 4N² (e.g., 64 for 4×4, 256 for 8×8) | Computed |
| Diameter (4×4) | 4 hops | Manhattan distance with wrap-around |
| Diameter (8×8) | 8 hops | Manhattan distance with wrap-around |
| Bisection bandwidth (4×4) | 800 Gbps | 2×4 links × 100 Gbps |
| Bisection bandwidth (8×8) | 1600 Gbps | 2×8 links × 100 Gbps |

---

## 3. Node Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| Buffer depth | 64 packets per directional queue | [config.yaml:L14](file:///c:/Projects/Optical%20Project/conf/config.yaml#L14), [node.py:L28](file:///c:/Projects/Optical%20Project/src/sim/node.py#L28) |
| Virtual channels (default) | 1 | [config.yaml:L15](file:///c:/Projects/Optical%20Project/conf/config.yaml#L15) |
| Virtual channels (Odd-Even) | 2 | Overridden by router |
| Total buffer capacity per node | 256 packets (4 dirs × 64) | Computed |
| Load tracking | EMA with α = 0.1 | [node.py:L150](file:///c:/Projects/Optical%20Project/src/sim/node.py#L150) |

---

## 4. Link Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| Bandwidth | 100 Gbps per link | [config.yaml:L18](file:///c:/Projects/Optical%20Project/conf/config.yaml#L18), [link.py:L34](file:///c:/Projects/Optical%20Project/src/sim/link.py#L34) |
| Propagation delay | 1 ns | [config.yaml:L19](file:///c:/Projects/Optical%20Project/conf/config.yaml#L19), [link.py:L35](file:///c:/Projects/Optical%20Project/src/sim/link.py#L35) |
| Transmission delay | 5.12 ns | [link.py:L64-L72](file:///c:/Projects/Optical%20Project/src/sim/link.py#L64-L72): `64 × 8 / 100×10⁹ × 10⁹ = 5.12 ns` |
| Total per-hop delay | 6.12 ns (transmission + propagation) | [link.py:L74-L77](file:///c:/Projects/Optical%20Project/src/sim/link.py#L74-L77) |
| Utilisation tracking | EMA-smoothed, updated per transmission | [link.py:L96](file:///c:/Projects/Optical%20Project/src/sim/link.py#L96) |

---

## 5. Packet Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| Payload size | 64 bytes | [config.yaml:L22](file:///c:/Projects/Optical%20Project/conf/config.yaml#L22), [packet.py](file:///c:/Projects/Optical%20Project/src/sim/packet.py) |
| Injection rate | 0.01 packets/ns (per source node) | [config.yaml:L44](file:///c:/Projects/Optical%20Project/conf/config.yaml#L44) |
| Effective aggregate rate (4×4) | 0.16 packets/ns (16 nodes × 0.01) | Computed |
| Effective aggregate rate (8×8) | 0.64 packets/ns (64 nodes × 0.01) | Computed |

---

## 6. Traffic Patterns

**Source**: [traffic.py](file:///c:/Projects/Optical%20Project/src/sim/traffic.py)

| Pattern | Description | Configuration | Source Lines |
|---------|-------------|---------------|-------------|
| **Uniform** | Each source sends to a uniformly random destination | Default | [traffic.py:L80-L100](file:///c:/Projects/Optical%20Project/src/sim/traffic.py) |
| **Hotspot** | 80% of traffic targets 20% of nodes | `hotspot_frac=0.8`, `hotspot_nodes_frac=0.2` | [traffic.py:L102-L120](file:///c:/Projects/Optical%20Project/src/sim/traffic.py) |
| **Adversarial** | Each node (x,y) sends to ((N-1-x), (N-1-y)) — maximum distance | Deterministic destinations | [traffic.py:L122-L144](file:///c:/Projects/Optical%20Project/src/sim/traffic.py) |

---

## 7. Failure Injection

| Parameter | Value | Source |
|-----------|-------|--------|
| Failure rates tested | 0.0 (healthy), 0.1 (10% links failed) | [config.yaml:L42](file:///c:/Projects/Optical%20Project/conf/config.yaml#L42) |
| Failure model | Random uniform link selection | [torus_graph.py `inject_link_failures()`](file:///c:/Projects/Optical%20Project/src/sim/torus_graph.py) |
| Number of failed links (4×4, 10%) | 6 links (out of 64) | 10% × 64 rounded |
| Number of failed links (8×8, 10%) | 25 links (out of 256) | 10% × 256 rounded |
| Failure persistence | Static per simulation run | Applied once before run |

---

## 8. Experiment Grid

**Source**: [config.yaml:L38-L44](file:///c:/Projects/Optical%20Project/conf/config.yaml#L38-L44)

| Dimension | Values | Count |
|-----------|--------|-------|
| Topologies | 4×4, 8×8 | 2 |
| Traffic patterns | Uniform, Hotspot, Adversarial | 3 |
| Routing algorithms | XY, Odd-Even, Valiant, Supervised GNN, GRL (Ours) | 5 |
| Failure rates | 0.0, 0.1 | 2 |
| Random seeds | 0, 1, 2, 3, 4 | 5 |
| **Total experiments** | | **300** |

Each experiment is an independent simulation run with a unique seed controlling: traffic generation, failure injection, and any stochastic routing decisions.

---

## 9. Metrics Collected

| Metric | Unit | Description |
|--------|------|-------------|
| `avg_latency_ns` | ns | Mean packet end-to-end latency (src→dst) |
| `p95_latency_ns` | ns | 95th percentile tail latency |
| `throughput_pps` | packets/ns | Delivered packets / simulation time |
| `drop_rate` | fraction [0,1] | Dropped packets / total injected packets |
| `mean_utilisation` | fraction [0,1] | Average link utilisation across all links |
| `avg_hop_count` | hops | Mean number of hops per delivered packet |

All metrics are computed over the steady-state window (after warmup period).

---

## 10. Reproducibility

| Item | Detail |
|------|--------|
| Python | 3.10 |
| PyTorch | 2.0+ |
| PyTorch Geometric | Latest stable |
| Hardware | CPU-only (Intel, user preference) |
| Seed management | `set_global_seed()` sets `random`, `numpy`, `torch` seeds | 
| Determinism | `torch.use_deterministic_algorithms(True)` when available |

All experiments are fully reproducible with the provided seeds. Results CSV files include the seed used for each run.
