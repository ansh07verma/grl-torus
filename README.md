# GRL-Torus: Graph Reinforcement Learning for Adaptive Routing

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![React](https://img.shields.io/badge/React-18.0+-61dafb.svg)

**GRL-Torus** is a research framework for optimising routing in 2D torus optical interconnects (used in HPC and data centres) using a combination of Graph Neural Networks (GNNs) and Deep Q-Networks (DQNs).

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
6. [Demo Visualiser](#demo-visualiser)

## Overview

Next-generation data centres require scalable, high-speed, low-latency optical interconnects. The 2D torus topology is highly effective but traditional deterministic routing (like XY) fails under non-uniform traffic and link failures.

GRL-Torus combines two powerful ML paradigms:
- **GraphSAGE Encoder**: Learns rich, structural node embeddings that capture the global topology and congestion state.
- **Dueling DQN Agent**: Uses these embeddings to make optimal, hop-by-hop routing decisions that minimise latency and packet drops.

## Features

- **SimPy Discrete-Event Simulator**: Highly accurate cycle-level simulator for 2D Torus networks with buffer tracking, link propagation delays, and virtual channels.
- **Traffic Generation**: Supports Uniform, Hotspot, and Adversarial traffic patterns.
- **Baseline Routers**: Compare GRL against standard XY, Odd-Even (turn models), and Valiant Load Balancing.
- **Fault Injection**: Train and evaluate under transient link/node failure scenarios.
- **React Visualiser**: An interactive, glassmorphic dashboard built with Vite + React to visualise packets routing across the Torus grid.

## Installation

```bash
# Clone the repository
git clone https://github.com/Ansh/grl-torus.git
cd grl-torus

# Install Python dependencies (Backend)
pip install -r requirements.txt

# Install Node dependencies (Frontend Demo)
cd demo
npm install
```

## Usage

### 1. Train the GNN Encoder (Supervised Pretraining)
First, we train a GraphSAGE model using Dijkstra-optimal routing labels on varying topologies.
```bash
python scripts/train_gnn.py --grid-size 4 --epochs 50 --batch-size 32
```

### 2. Train the DQN Policy (Reinforcement Learning)
Next, train the RL agent using Prioritized Experience Replay. You can freeze the GNN or joint-train both networks.
```bash
python scripts/train_dqn.py --grid-size 4 --episodes 500 --gnn-checkpoint results/checkpoints/gnn_best_4x4.pt
```

### 3. Run the Experiment Grid
Evaluate the models across different traffic patterns and failure rates, generating a consolidated CSV of metrics.
```bash
python scripts/run_experiments.py --topologies 4 8 --routers xy odd_even valiant grl
```

### 4. Generate Figures
Produce publication-ready Matplotlib/Seaborn charts.
```bash
python scripts/generate_figures.py
```

## Evaluation Results

The GRL-Torus framework has been thoroughly evaluated across multiple grid sizes (4x4, 8x8), traffic patterns (Uniform, Hotspot, Adversarial), and failure scenarios. Below are some of the key results generated from our automated experiment pipeline.

### 1. Average End-to-End Latency
GRL dynamically adapts to congestion, achieving lower average latency than Valiant load balancing and the Odd-Even turn model, especially under non-uniform traffic.
![Latency Comparison](results/figures/fig1_latency_4x4.png)

### 2. Network Throughput
By avoiding congested links and minimising packet drops, GRL sustains higher throughput than deterministic baselines like XY routing.
![Throughput Comparison](results/figures/fig2_throughput_4x4.png)

### 3. Packet Drop Rate
GRL significantly reduces packet loss under Hotspot and Adversarial traffic patterns compared to baseline routers.
![Drop Rate Comparison](results/figures/fig3_drop_rate_4x4.png)

### 4. Fault Resilience
When subjected to a 10% random link failure rate, the GRL router seamlessly reroutes packets around dead links, maintaining lower latency degradation than standard adaptive algorithms.
![Failure Resilience](results/figures/fig6_failure_resilience_4x4.png)

For a complete breakdown of the evaluation methodology, model architecture, and statistical significance tests, refer to the documents in the `docs/` folder.

## Demo Visualiser

To launch the animated React dashboard:
```bash
cd demo
npm run dev
```
Open `http://localhost:5173` in your browser to interact with the Torus grid!

## Project Structure

```
├── conf/                 # Hydra configuration files (YAML)
├── demo/                 # React UI Dashboard (Vite + React + TS)
├── results/              # Checkpoints, Logs, CSVs, Figures
├── scripts/              # CLI entry points (train, evaluate, viz)
├── src/                  # Core Python Packages
│   ├── experiments/      # Metric collection and Runner
│   ├── models/           # GNN, DQN, Replay Buffer, Training loops
│   ├── routers/          # Baselines (XY) and ML Routers (GNN, GRL)
│   ├── sim/              # SimPy Engine, Packets, Torus Graph
│   └── viz/              # Matplotlib plotting scripts
├── tests/                # Pytest Unit & Integration tests
└── requirements.txt      # Python dependencies
```

## Author
**Ansh** - Vellore Institute of Technology (Chennai)
School of Electronics Engineering
Target Venue: IEEE Networking Letters 2026
