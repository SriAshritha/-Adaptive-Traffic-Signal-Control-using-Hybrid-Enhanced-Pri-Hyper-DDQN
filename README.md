# Hybrid-Enhanced Adaptive Traffic Signal Control using Pri-Hyper DDQN

## Overview

This repository presents a Reinforcement Learning–based adaptive traffic signal control framework for isolated urban intersections. The project investigates multiple classical and deep RL baselines and introduces a final enhanced method:

**Hybrid-Enhanced Pri-Hyper DDQN**

The proposed framework combines:

* Prioritized Double Dueling Deep Q-Networks (Pri-DDQN)
* Hypergraph-inspired movement interaction modeling
* Pressure-aware traffic prioritization
* Conservative signal phase switching
* Multi-objective reward optimization

The system is designed to reduce:

* Vehicle waiting time
* Queue congestion
* Unnecessary phase switching
* Traffic imbalance across directions

while improving:

* Throughput
* Intersection efficiency
* Adaptive control behavior under dynamic traffic conditions

---

# Project Objectives

The primary goal of this work is to design an intelligent adaptive traffic signal controller capable of learning efficient signal policies from traffic states using Reinforcement Learning.

The project compares:

* Classical RL methods
* Deep RL methods
* Enhanced hybrid architectures

under a movement-based traffic benchmark.

---

# Final Proposed Method

## Hybrid-Enhanced Pri-Hyper DDQN

The final architecture integrates:

### 1. Prioritized Experience Replay

Improves learning efficiency by replaying more informative transitions.

### 2. Double DQN

Reduces Q-value overestimation.

### 3. Dueling Network Architecture

Separates:

* State value estimation
* Action advantage estimation

for more stable learning.

### 4. Hypergraph-inspired Movement Encoding

Captures relationships between traffic movements:

* Straight traffic
* Turning traffic
* Cross-direction interactions

### 5. Pressure-aware Action Biasing

Prioritizes highly congested directions dynamically.

### 6. Conservative Phase Switching

Avoids excessive signal switching penalties and unstable policies.

---

# Repository Structure

```text
traffic_dqn/
│
├── agents/
│   ├── dqn_agent.py
│   └── classical_agents.py
│
├── environments/
│   ├── traffic_env.py
│   └── sumo_env.py
│
├── Pri-DDQN/
│   ├── train.py
│   ├── model.py
│   ├── replay_buffer.py
│   └── outputs/
│
├── MA-SAC/
│   ├── train.py
│   ├── model.py
│   ├── replay_buffer.py
│   └── outputs/
│
├── main_experiment_6action/
│   ├── agents/
│   ├── environments/
│   └── train_classical.py
│
├── hybrid_enhanced_benchmark.py
├── benchmark_improved_agents.py
├── evaluate.py
├── deploy_sumo.py
├── train_dqn.py
├── train_classical.py
├── requirements.txt
│
├── our_paper/
│   ├── main.tex
│   └── architecture diagrams
│
└── sumo_config/
    ├── simulation.sumocfg
    ├── routes.rou.xml
    └── tls.add.xml
```

---

# Environment Description

The environment models a four-way urban intersection with movement-level traffic control.

## Action Space

The controller selects one among four traffic phases:

| Action | Traffic Phase           |
| ------ | ----------------------- |
| 0      | North-South Straight    |
| 1      | North-South Left/U-turn |
| 2      | East-West Straight      |
| 3      | East-West Left/U-turn   |

---

## State Representation

The state includes:

* Queue lengths
* Waiting times
* Lane pressures
* Throughput information
* Movement-wise traffic densities
* Previous signal state information

---

## Reward Function

The reward is multi-objective and balances:

* Queue reduction
* Waiting-time minimization
* Throughput maximization
* Fairness across lanes
* Switching stability

---

# Technologies Used

## Programming Language

* Python

## Libraries

* PyTorch
* NumPy
* Pillow
* Matplotlib

## Traffic Simulator

* SUMO (Simulation of Urban Mobility)

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/your-username/Adaptive-Traffic-Signal-Control.git
cd Adaptive-Traffic-Signal-Control
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Install SUMO

Download SUMO from:

[https://www.eclipse.org/sumo/](https://www.eclipse.org/sumo/)

Add SUMO to your system PATH.

Verify installation:

```bash
sumo --version
```

---

# Running Experiments

## Train Classical RL Agents

```bash
python train_classical.py
```

---

## Train DQN Agent

```bash
python train_dqn.py
```

---

## Run Pri-DDQN Benchmark

```bash
python Pri-DDQN/train.py
```

---

## Run MA-SAC Benchmark

```bash
python MA-SAC/train.py
```

---

## Run Final Hybrid-Enhanced Benchmark

```bash
python hybrid_enhanced_benchmark.py
```

---

## Evaluate Trained Models

```bash
python evaluate.py
```

---

# SUMO Deployment

To visualize the learned traffic control policy inside SUMO:

```bash
python deploy_sumo.py
```

---

# Research Contributions

This project contributes:

* A hybrid RL architecture for adaptive traffic control
* Hypergraph-inspired movement interaction encoding
* Pressure-aware signal prioritization
* Comparative benchmarking across multiple RL methods
* Multi-objective traffic optimization framework

---

# Benchmark Methods Included

| Method                         | Type            |
| ------------------------------ | --------------- |
| TD(0)                          | Classical RL    |
| SARSA                          | Classical RL    |
| Q-Learning                     | Classical RL    |
| Pri-DDQN                       | Deep RL         |
| MA-SAC                         | Deep RL         |
| Hybrid-Enhanced Pri-Hyper DDQN | Proposed Method |

---

# Outputs

The repository includes:

* Trained models
* Benchmark results
* Evaluation metrics
* Research paper assets
* SUMO simulation configuration
* Experiment outputs

---

# Research Paper

The repository also contains:

* LaTeX paper source
* Architecture diagrams
* Benchmark tables
* Methodology documentation

Located in:

```text
our_paper/
```

---

# Future Improvements

Potential future extensions include:

* Multi-intersection coordination
* Graph Neural Networks
* Transformer-based traffic forecasting
* Real-world sensor integration
* Multi-agent reinforcement learning
* Dynamic traffic demand prediction

---

# Author

Ashritha
Harshitha
Srija
Raviteja
B.Tech Computer Science Student

Focused on:

* Reinforcement Learning
* Deep Learning
* Intelligent Transportation Systems
* AI Research

---

# License

This project is intended for academic and research purposes.

---

# Acknowledgements

* SUMO Traffic Simulator
* PyTorch
* Open-source RL research community
* Prior work on adaptive traffic signal optimization
