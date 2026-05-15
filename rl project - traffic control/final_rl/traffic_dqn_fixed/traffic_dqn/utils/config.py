"""
Traffic Light Control - Configuration File
==========================================
Central configuration for all RL agents, environment, and SUMO simulation.
"""

# ─────────────────────────────────────────────────────────────────
# ENVIRONMENT SETTINGS
# ─────────────────────────────────────────────────────────────────
ENV_CONFIG = {
    "num_lanes": 4,              # N, S, E, W – one lane each
    "max_queue": 20,             # max vehicles per lane (used for normalization)
    "max_wait_time": 300,        # seconds, used for normalization
    "phase_duration": 10,        # seconds per phase before agent can act
    "yellow_duration": 3,        # yellow light seconds (applied between phase changes)
    "simulation_seconds": 3600,  # 1-hour simulation horizon
    "step_length": 1.0,          # SUMO step length (seconds)
}

# Traffic signal phases (SUMO tls phase indices)
# Phase 0: N-S Green, E-W Red
# Phase 1: N-S Yellow
# Phase 2: E-W Green, N-S Red
# Phase 3: E-W Yellow
PHASES = {
    0: "NS_GREEN",
    1: "NS_YELLOW",
    2: "EW_GREEN",
    3: "EW_YELLOW",
}
ACTION_PHASES = [0, 2]   # Only green phases are selectable actions
NUM_ACTIONS = len(ACTION_PHASES)

# ─────────────────────────────────────────────────────────────────
# TABULAR RL (TD0, SARSA, Q-LEARNING)
# ─────────────────────────────────────────────────────────────────
TABULAR_CONFIG = {
    "alpha": 0.08,         # learning rate
    "gamma": 0.97,         # discount factor
    "epsilon_start": 1.0,  # initial exploration
    "epsilon_end": 0.02,   # final exploration
    "epsilon_decay": 0.997,
    "num_episodes": 800,
    "max_steps": 3600,
    # State discretisation bins
    "queue_bins": 41,      # exact axis queue totals from 0 to 40 vehicles
    "wait_bins": 8,        # coarse wait-pressure buckets per axis
    "pressure_bins": 17,   # directional pressure imbalance bins
    "phase_age_bins": 6,   # coarse phase-age buckets to stabilize switching
    "trace_lambda": 0.8,
    "optimistic_init": 0.15,
    "heuristic_prior": 0.12,
}

# ─────────────────────────────────────────────────────────────────
# DEEP Q-NETWORK (DQN)
# ─────────────────────────────────────────────────────────────────
DQN_CONFIG = {
    # Architecture
    "state_dim": 14,         # raw state plus pressure-aware engineered features
    "action_dim": 2,         # NS_GREEN or EW_GREEN
    "hidden_layers": [256, 256, 128],
    "dueling": True,         # Enable Dueling DQN
    "double_dqn": True,      # Enable Double DQN

    # Training hyperparameters
    "alpha": 1e-3,           # Adam learning rate
    "gamma": 0.95,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.9995,
    "batch_size": 64,
    "target_update_freq": 500,  # steps between target-net sync
    "soft_target_tau": 0.01,

    # Replay buffer
    "buffer_capacity": 50_000,
    "min_replay_size": 1_000,   # warm-up before training starts
    "prioritized_replay": True,
    "priority_alpha": 0.6,
    "priority_beta_start": 0.4,
    "priority_beta_frames": 200000,

    # Training schedule
    "num_episodes": 1000,
    "max_steps_per_episode": 3600,
    "save_freq": 100,           # save model every N episodes
}

# ─────────────────────────────────────────────────────────────────
# SUMO SIMULATION
# ─────────────────────────────────────────────────────────────────
SUMO_CONFIG = {
    "sumo_binary": "sumo",          # use "sumo-gui" for visual mode
    "config_file": "sumo_config/simulation.sumocfg",
    "tls_id": "intersection",       # traffic light ID in SUMO
    "lane_ids": [                   # TraCI lane IDs to monitor
        "north_in_0",
        "south_in_0",
        "east_in_0",
        "west_in_0",
    ],
    "port": 8813,
    "step_length": 1.0,
    "seed": 42,
}

# ─────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────
EVAL_CONFIG = {
    "num_eval_episodes": 20,
    "metrics": [
        "avg_waiting_time",
        "avg_queue_length",
        "throughput",
        "total_delay",
    ],
}

# ─────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────
PATHS = {
    "models_dir": "models/",
    "results_dir": "outputs/",
    "td0_model": "models/td0_qtable.pkl",
    "sarsa_model": "models/sarsa_qtable.pkl",
    "qlearning_model": "models/qlearning_qtable.pkl",
    "dqn_model": "models/dqn_agent.pkl",
    "dqn_weights": "models/dqn_weights.pth",
}
