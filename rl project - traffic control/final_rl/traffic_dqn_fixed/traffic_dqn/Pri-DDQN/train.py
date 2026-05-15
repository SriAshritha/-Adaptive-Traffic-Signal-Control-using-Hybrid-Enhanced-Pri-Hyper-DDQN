import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent import PriDDQNAgent
from config import BENCHMARKS, COMMON_CONFIG, MODEL_DIR, OUTPUT_DIR


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(benchmark_name: str, seed: int, mode: str):
    if benchmark_name == "single_2action":
        from environments.traffic_env import TrafficEnv

        return TrafficEnv(seed=seed, mode=mode), "raw"
    if benchmark_name == "fourway_4action":
        from main_experiment_6action.environments.traffic_env import TrafficEnv

        return TrafficEnv(seed=seed, mode=mode), "raw"
    raise ValueError(benchmark_name)


def reset_env(env, kind: str):
    if kind == "raw":
        state, _ = env.reset()
        return state
    raise ValueError(kind)


def step_env(env, action: int, kind: str):
    if kind == "raw":
        next_state, _, reward, done, _ = env.step(action)
        return next_state, reward, done
    raise ValueError(kind)


def train_one(benchmark_name: str, config: dict, device: torch.device):
    benchmark = BENCHMARKS[benchmark_name]
    env, kind = make_env(benchmark_name, config["seed"], benchmark["mode"])
    agent = PriDDQNAgent(benchmark["state_dim"], benchmark["action_dim"], config, device)
    metrics = {
        "episode_reward": [],
        "avg_queue_length": [],
        "avg_waiting_time": [],
        "total_throughput": [],
        "total_delay": [],
        "phase_changes": [],
        "loss": [],
        "epsilon": [],
    }

    for episode in range(config["episodes"]):
        state = reset_env(env, kind)
        done = False
        ep_reward = 0.0
        ep_losses = []
        epsilon = agent.epsilon_for_episode(episode, config["episodes"])

        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, done = step_env(env, action, kind)
            agent.store(state, action, reward, next_state, done)
            loss = None
            if agent.total_steps % config["update_interval"] == 0:
                loss = agent.update(episode, config["episodes"])
            if loss is not None:
                ep_losses.append(loss)
            ep_reward += reward
            state = next_state

        stats = env.get_episode_stats()
        metrics["episode_reward"].append(float(ep_reward))
        metrics["avg_queue_length"].append(float(stats["avg_queue_length"]))
        metrics["avg_waiting_time"].append(float(stats["avg_waiting_time"]))
        metrics["total_throughput"].append(float(stats["total_throughput"]))
        metrics["total_delay"].append(float(stats.get("total_delay", 0.0)))
        metrics["phase_changes"].append(float(stats["phase_changes"]))
        metrics["loss"].append(float(np.mean(ep_losses)) if ep_losses else None)
        metrics["epsilon"].append(float(epsilon))

        if (episode + 1) % 50 == 0:
            print(
                f"{benchmark_name} ep={episode+1:4d} "
                f"reward={np.mean(metrics['episode_reward'][-50:]):8.3f} "
                f"queue={np.mean(metrics['avg_queue_length'][-50:]):6.2f} "
                f"wait={np.mean(metrics['avg_waiting_time'][-50:]):7.2f} "
                f"throughput={np.mean(metrics['total_throughput'][-50:]):7.1f}"
            )

    model_path = MODEL_DIR / f"{benchmark_name}_pri_ddqn.pt"
    agent.save(model_path)
    return agent, metrics


def evaluate_one(benchmark_name: str, agent: PriDDQNAgent, config: dict):
    benchmark = BENCHMARKS[benchmark_name]
    rows = []
    for offset in range(config["eval_episodes"]):
        seed = config["seed"] + 100 + offset
        env, kind = make_env(benchmark_name, seed, benchmark["mode"])
        state = reset_env(env, kind)
        done = False
        total_reward = 0.0
        action_hist = {str(i): 0 for i in range(benchmark["action_dim"])}
        while not done:
            action = agent.greedy_action(state)
            action_hist[str(action)] += 1
            next_state, reward, done = step_env(env, action, kind)
            total_reward += reward
            state = next_state
        stats = env.get_episode_stats()
        rows.append(
            {
                "seed": seed,
                "reward": float(total_reward),
                "avg_queue_length": float(stats["avg_queue_length"]),
                "avg_waiting_time": float(stats["avg_waiting_time"]),
                "total_throughput": float(stats["total_throughput"]),
                "total_delay": float(stats.get("total_delay", 0.0)),
                "phase_changes": float(stats["phase_changes"]),
                "action_histogram": action_hist,
            }
        )

    summary = {
        "eval_reward_mean": float(np.mean([row["reward"] for row in rows])),
        "eval_reward_std": float(np.std([row["reward"] for row in rows])),
        "eval_avg_queue_length": float(np.mean([row["avg_queue_length"] for row in rows])),
        "eval_avg_waiting_time": float(np.mean([row["avg_waiting_time"] for row in rows])),
        "eval_total_throughput": float(np.mean([row["total_throughput"] for row in rows])),
        "eval_total_delay": float(np.mean([row["total_delay"] for row in rows])),
        "eval_phase_changes": float(np.mean([row["phase_changes"] for row in rows])),
    }
    return {"summary": summary, "rows": rows}


def write_outputs(train_results: dict, eval_results: dict):
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_DIR / "train_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(train_results, handle, indent=2)
    with open(OUTPUT_DIR / "eval_results.json", "w", encoding="utf-8") as handle:
        json.dump(eval_results, handle, indent=2)
    with open(OUTPUT_DIR / "summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "benchmark",
                "eval_reward_mean",
                "eval_reward_std",
                "eval_avg_queue_length",
                "eval_avg_waiting_time",
                "eval_total_throughput",
                "eval_total_delay",
                "eval_phase_changes",
            ],
        )
        writer.writeheader()
        for benchmark_name, payload in eval_results.items():
            writer.writerow({"benchmark": benchmark_name, **payload["summary"]})


def write_methodology_note():
    note = """# Pri-DDQN Benchmark Note

## Paper
- Pri-DDQN: learning adaptive traffic signal control strategy through a hybrid agent
- DOI: https://doi.org/10.1007/s40747-024-01651-5

## Extracted method components
- Double DQN backbone
- Priority-based dynamic experience replay
- Power-function exploration decay
- Asynchronous target network updates
- State and reward incorporated into loss / replay importance

## Benchmark-fair adaptation
- The original paper is single-intersection ATSC.
- Here it is implemented on two repository benchmarks:
  - original 2-action single-intersection setup
  - movement-based 4-stage 4-way setup
- Reward, horizon, seeds, and evaluation style were kept benchmark-consistent.
- Because the repository state vectors are compact rather than image-like DTSE tensors, the network uses a lightweight 1D-convolution feature extractor instead of a large image CNN.

## Environment note
- Requested environment `traffic-rl` was not usable for PyTorch training on this machine due to a Windows DLL import failure.
- Pri-DDQN training and evaluation were run with `traffic_dqn`, while keeping benchmark code and seeds unchanged.
"""
    (OUTPUT_DIR / "methodology_note.md").write_text(note, encoding="utf-8")


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    set_seed(COMMON_CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_results = {}
    eval_results = {}
    for benchmark_name in ("single_2action", "fourway_4action"):
        print(f"\n=== Training Pri-DDQN on {benchmark_name} ===")
        agent, metrics = train_one(benchmark_name, COMMON_CONFIG, device)
        train_results[benchmark_name] = metrics
        eval_results[benchmark_name] = evaluate_one(benchmark_name, agent, COMMON_CONFIG)

    write_outputs(train_results, eval_results)
    write_methodology_note()

    print("\nFinal Pri-DDQN summary")
    print("=" * 90)
    print(
        f"{'Benchmark':<22} {'Reward':>10} {'AvgQueue':>10} {'AvgWait':>10} "
        f"{'Throughput':>12} {'Delay':>12} {'PhChanges':>12}"
    )
    print("-" * 90)
    for benchmark_name, payload in eval_results.items():
        summary = payload["summary"]
        print(
            f"{benchmark_name:<22} "
            f"{summary['eval_reward_mean']:>10.3f} "
            f"{summary['eval_avg_queue_length']:>10.3f} "
            f"{summary['eval_avg_waiting_time']:>10.3f} "
            f"{summary['eval_total_throughput']:>12.3f} "
            f"{summary['eval_total_delay']:>12.3f} "
            f"{summary['eval_phase_changes']:>12.3f}"
        )


if __name__ == "__main__":
    main()
