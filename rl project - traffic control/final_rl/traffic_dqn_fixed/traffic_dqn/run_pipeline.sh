#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  Traffic Light Control – DQN  |  Complete Pipeline Runner
# ═══════════════════════════════════════════════════════════════
#
# STEP 0 – Prerequisites
#   pip install numpy torch matplotlib
#   pip install traci sumolib        # only needed for SUMO deployment
#   sudo apt install sumo sumo-tools # SUMO simulator
#
# STEP 1 – Generate SUMO network (run once)
#   cd sumo_config
#   netconvert --node-files intersection.nod.xml \
#              --edge-files intersection.edg.xml \
#              -o intersection.net.xml
#   cd ..
#
# STEP 2 – Train classical RL (TD0, SARSA, Q-Learning)
#   python train_classical.py --episodes 500 --mode normal
#
# STEP 3 – Train DQN
#   python train_dqn.py --episodes 1000 --mode normal
#
# STEP 4 – Deploy in SUMO
#   python deploy_sumo.py --agent dqn           # DQN agent
#   python deploy_sumo.py --agent fixed          # fixed-time baseline
#   python deploy_sumo.py --agent compare        # all controllers
#   python deploy_sumo.py --agent dqn --gui      # with SUMO-GUI
#
# STEP 5 – Evaluate and plot
#   python evaluate.py

set -e

echo "═══════════════════════════════════════════════"
echo "  Traffic DQN – Pipeline"
echo "═══════════════════════════════════════════════"

mkdir -p models outputs

# Train classical RL
echo "[1/4] Training classical RL agents..."
python train_classical.py --episodes 500 --mode normal

# Train DQN
echo "[2/4] Training DQN..."
python train_dqn.py --episodes 1000 --mode normal

# Evaluate without SUMO
echo "[3/4] Generating learning curves..."
python evaluate.py

echo "[4/4] Done! Models saved in ./models/, plots in ./outputs/"
echo ""
echo "To deploy in SUMO (requires SUMO installation):"
echo "  python deploy_sumo.py --agent compare"
