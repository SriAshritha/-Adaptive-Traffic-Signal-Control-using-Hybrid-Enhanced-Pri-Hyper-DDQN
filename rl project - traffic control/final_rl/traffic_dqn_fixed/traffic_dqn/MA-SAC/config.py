from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"


TRAIN_CONFIG = {
    "seed": 42,
    "mode": "normal",
    "episodes": 500,
    "eval_episodes": 30,
    "batch_size": 64,
    "buffer_capacity": 100_000,
    "warmup_steps": 1_000,
    "updates_per_step": 1,
    "update_interval": 4,
    "gamma": 0.99,
    "tau": 0.01,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "alpha_lr": 1e-4,
    "hidden_dim": 64,
    "embedding_dim": 32,
    "temporal_window": 3,
    "target_entropy_scale": 0.8,
}


BENCHMARK_ACTIONS = {
    0: {"name": "NS Straight", "sumo_phase_index": 0},
    1: {"name": "NS Left / U-turn", "sumo_phase_index": 2},
    2: {"name": "EW Straight", "sumo_phase_index": 4},
    3: {"name": "EW Left / U-turn", "sumo_phase_index": 6},
}


RESULT_PATHS = {
    "checkpoint": MODEL_DIR / "ma_sac_model.pt",
    "train_metrics": OUTPUT_DIR / "train_metrics.json",
    "eval_results": OUTPUT_DIR / "eval_results.json",
    "summary_csv": OUTPUT_DIR / "summary.csv",
    "methodology_note": OUTPUT_DIR / "methodology_note.md",
}
