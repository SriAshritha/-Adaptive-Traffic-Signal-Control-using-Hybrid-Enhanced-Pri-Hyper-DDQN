from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"


COMMON_CONFIG = {
    "seed": 42,
    "episodes": 500,
    "eval_episodes": 30,
    "gamma": 0.95,
    "lr": 5e-4,
    "batch_size": 64,
    "buffer_capacity": 50_000,
    "warmup_steps": 1_000,
    "target_update_interval": 8,
    "tau": 0.05,
    "hidden_dim": 64,
    "update_interval": 4,
    "priority_alpha": 0.70,
    "priority_beta_start": 0.40,
    "priority_beta_end": 1.00,
    "priority_reward_scale": 0.20,
    "priority_state_scale": 0.10,
    "eps_start": 1.0,
    "eps_end": 0.02,
}


BENCHMARKS = {
    "single_2action": {
        "label": "Single Intersection (2-action)",
        "mode": "normal",
        "state_dim": 9,
        "action_dim": 2,
        "import_root": "root",
    },
    "fourway_4action": {
        "label": "4-Way Movement Benchmark (4-action)",
        "mode": "normal",
        "state_dim": 17,
        "action_dim": 4,
        "import_root": "main_experiment_6action",
    },
}
