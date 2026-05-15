"""
Configuration for the main movement-based 4-action traffic RL experiment.

The user-provided table defines four controllable green actions:
0 -> NS Straight
1 -> NS Left / U-turn
2 -> EW Straight
3 -> EW Left / U-turn

The "SUMO phase index" mapping is kept here only as the canonical movement
definition for future benchmarking. This experiment trains only in the local
stochastic RL environment and does not run SUMO evaluation.
"""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"


ACTION_DEFINITIONS = {
    0: {
        "name": "NS Straight",
        "movement": "North-South straight with right-turn traffic grouped in the same stage",
        "sumo_phase_index": 0,
        "served_groups": ("n_through", "s_through"),
    },
    1: {
        "name": "NS Left / U-turn",
        "movement": "North-South protected left-turn and U-turn stage",
        "sumo_phase_index": 2,
        "served_groups": ("n_turn", "s_turn"),
    },
    2: {
        "name": "EW Straight",
        "movement": "East-West straight with right-turn traffic grouped in the same stage",
        "sumo_phase_index": 4,
        "served_groups": ("e_through", "w_through"),
    },
    3: {
        "name": "EW Left / U-turn",
        "movement": "East-West protected left-turn and U-turn stage",
        "sumo_phase_index": 6,
        "served_groups": ("e_turn", "w_turn"),
    },
}


NUM_ACTIONS = len(ACTION_DEFINITIONS)
MOVEMENT_KEYS = (
    "n_through",
    "n_turn",
    "s_through",
    "s_turn",
    "e_through",
    "e_turn",
    "w_through",
    "w_turn",
)


ENV_CONFIG = {
    "max_queue": 30,
    "max_wait_time": 300.0,
    "decision_interval": 10,
    "yellow_duration": 3,
    "simulation_seconds": 3600,
    "movement_rates": {
        "normal": {
            "n_through": 0.10,
            "n_turn": 0.02,
            "s_through": 0.10,
            "s_turn": 0.02,
            "e_through": 0.10,
            "e_turn": 0.02,
            "w_through": 0.10,
            "w_turn": 0.02,
        },
        "peak": {
            "n_through": 0.14,
            "n_turn": 0.03,
            "s_through": 0.14,
            "s_turn": 0.03,
            "e_through": 0.18,
            "e_turn": 0.04,
            "w_through": 0.18,
            "w_turn": 0.04,
        },
        "asymmetric": {
            "n_through": 0.07,
            "n_turn": 0.02,
            "s_through": 0.14,
            "s_turn": 0.03,
            "e_through": 0.18,
            "e_turn": 0.04,
            "w_through": 0.06,
            "w_turn": 0.01,
        },
    },
    "service_rates": {
        "through": 0.80,
        "turn": 0.55,
    },
}


TABULAR_CONFIG = {
    "alpha": 0.10,
    "gamma": 0.95,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay": 0.995,
    "num_episodes": 500,
    "eval_episodes": 30,
    "queue_bins": 5,
}


PATHS = {
    "td0_model": MODELS_DIR / "td0_qtable.pkl",
    "sarsa_model": MODELS_DIR / "sarsa_qtable.pkl",
    "qlearning_model": MODELS_DIR / "qlearning_qtable.pkl",
    "metrics": MODELS_DIR / "classical_metrics.pkl",
    "results_json": OUTPUTS_DIR / "final_results.json",
    "results_csv": OUTPUTS_DIR / "final_results.csv",
}
