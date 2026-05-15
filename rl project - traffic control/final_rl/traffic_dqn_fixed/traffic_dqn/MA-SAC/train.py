import csv
import json
import random
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent import MASACAgent
from config import BENCHMARK_ACTIONS, OUTPUT_DIR, RESULT_PATHS, TRAIN_CONFIG
from replay_buffer import ReplayBuffer
from main_experiment_6action.environments.traffic_env import TrafficEnv


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_state_sequence(history: deque, temporal_window: int):
    items = list(history)
    while len(items) < temporal_window:
        items.insert(0, items[0])
    return np.stack(items[-temporal_window:], axis=0)


def train():
    OUTPUT_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(TRAIN_CONFIG["seed"])

    env = TrafficEnv(seed=TRAIN_CONFIG["seed"], mode=TRAIN_CONFIG["mode"])
    agent = MASACAgent(device=device)
    replay = ReplayBuffer(TRAIN_CONFIG["buffer_capacity"])
    total_steps = 0
    train_metrics = defaultdict(list)

    for episode in range(1, TRAIN_CONFIG["episodes"] + 1):
        state, _ = env.reset()
        history = deque([state.copy()], maxlen=TRAIN_CONFIG["temporal_window"])
        episode_reward = 0.0
        action_counts = [0, 0, 0, 0]
        update_losses = []

        while True:
            state_seq = make_state_sequence(history, TRAIN_CONFIG["temporal_window"])
            action, probs = agent.select_action(state_seq, greedy=False)
            next_state, _, reward, done, _ = env.step(action)
            history_next = deque(history, maxlen=TRAIN_CONFIG["temporal_window"])
            history_next.append(next_state.copy())
            next_state_seq = make_state_sequence(history_next, TRAIN_CONFIG["temporal_window"])

            replay.push(state_seq, action, reward, next_state_seq, done)
            history = history_next
            episode_reward += reward
            action_counts[action] += 1
            total_steps += 1

            if (
                len(replay) >= TRAIN_CONFIG["warmup_steps"]
                and total_steps % TRAIN_CONFIG["update_interval"] == 0
            ):
                for _ in range(TRAIN_CONFIG["updates_per_step"]):
                    batch = replay.sample(TRAIN_CONFIG["batch_size"], device)
                    update_losses.append(agent.update(batch))

            if done:
                break

        stats = env.get_episode_stats()
        train_metrics["episode_reward"].append(float(episode_reward))
        train_metrics["avg_queue_length"].append(float(stats["avg_queue_length"]))
        train_metrics["avg_waiting_time"].append(float(stats["avg_waiting_time"]))
        train_metrics["total_throughput"].append(float(stats["total_throughput"]))
        train_metrics["total_delay"].append(float(stats["total_delay"]))
        train_metrics["phase_changes"].append(float(stats["phase_changes"]))
        train_metrics["action_counts"].append(action_counts)

        if update_losses:
            for key in update_losses[0]:
                train_metrics[key].append(float(np.mean([item[key] for item in update_losses])))
        else:
            for key in ("critic1_loss", "critic2_loss", "actor_loss", "alpha", "entropy"):
                train_metrics[key].append(None)

        if episode % 25 == 0:
            print(
                f"ep={episode:4d} reward={np.mean(train_metrics['episode_reward'][-25:]):8.3f} "
                f"queue={np.mean(train_metrics['avg_queue_length'][-25:]):6.2f} "
                f"wait={np.mean(train_metrics['avg_waiting_time'][-25:]):7.2f} "
                f"throughput={np.mean(train_metrics['total_throughput'][-25:]):7.1f}"
            )

    agent.save()
    with open(RESULT_PATHS["train_metrics"], "w", encoding="utf-8") as handle:
        json.dump(train_metrics, handle, indent=2)
    return agent, train_metrics


def evaluate(agent: MASACAgent):
    eval_rows = []
    start_seed = TRAIN_CONFIG["seed"] + 100
    for offset in range(TRAIN_CONFIG["eval_episodes"]):
        seed = start_seed + offset
        env = TrafficEnv(seed=seed, mode=TRAIN_CONFIG["mode"])
        state, _ = env.reset()
        history = deque([state.copy()], maxlen=TRAIN_CONFIG["temporal_window"])
        total_reward = 0.0
        action_hist = {str(i): 0 for i in BENCHMARK_ACTIONS}
        mean_probs = []

        while True:
            state_seq = make_state_sequence(history, TRAIN_CONFIG["temporal_window"])
            action, probs = agent.select_action(state_seq, greedy=True)
            action_hist[str(action)] += 1
            mean_probs.append(probs)
            next_state, _, reward, done, _ = env.step(action)
            history.append(next_state.copy())
            total_reward += reward
            if done:
                break

        stats = env.get_episode_stats()
        eval_rows.append(
            {
                "seed": seed,
                "reward": float(total_reward),
                "avg_queue_length": float(stats["avg_queue_length"]),
                "avg_waiting_time": float(stats["avg_waiting_time"]),
                "total_throughput": float(stats["total_throughput"]),
                "total_delay": float(stats["total_delay"]),
                "phase_changes": float(stats["phase_changes"]),
                "action_histogram": action_hist,
                "mean_action_probabilities": np.mean(np.array(mean_probs), axis=0).tolist(),
            }
        )

    summary = {
        "eval_reward_mean": float(np.mean([row["reward"] for row in eval_rows])),
        "eval_reward_std": float(np.std([row["reward"] for row in eval_rows])),
        "eval_avg_queue_length": float(np.mean([row["avg_queue_length"] for row in eval_rows])),
        "eval_avg_waiting_time": float(np.mean([row["avg_waiting_time"] for row in eval_rows])),
        "eval_total_throughput": float(np.mean([row["total_throughput"] for row in eval_rows])),
        "eval_total_delay": float(np.mean([row["total_delay"] for row in eval_rows])),
        "eval_phase_changes": float(np.mean([row["phase_changes"] for row in eval_rows])),
    }
    payload = {
        "paper": {
            "title": "Towards Multi-agent Reinforcement Learning based Traffic Signal Control through Spatio-temporal Hypergraphs",
            "method": "Benchmark-fair single-intersection MA-SAC adaptation",
        },
        "setup": {
            "mode": TRAIN_CONFIG["mode"],
            "episodes": TRAIN_CONFIG["episodes"],
            "eval_episodes": TRAIN_CONFIG["eval_episodes"],
            "actions": BENCHMARK_ACTIONS,
        },
        "summary": summary,
        "eval_rows": eval_rows,
    }
    with open(RESULT_PATHS["eval_results"], "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    with open(RESULT_PATHS["summary_csv"], "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", *summary.keys()])
        writer.writeheader()
        writer.writerow({"method": "MA-SAC", **summary})
    return payload


def write_methodology_note():
    note = """# Methodology Note

## Repository audit
- Benchmark environment reused from `main_experiment_6action/environments/traffic_env.py`.
- Action space fixed to four movement stages: NS straight, NS left/U-turn, EW straight, EW left/U-turn.
- Reward, decision interval, yellow duration, horizon, and evaluation protocol were kept unchanged.
- Requested environment was `traffic-rl`, but PyTorch could not load there due to a Windows `fbgemm.dll` import failure, so MA-SAC training was run in `traffic_dqn` while keeping the benchmark code and seeds unchanged.

## Paper-method extraction
- Source paper: Towards Multi-agent Reinforcement Learning based Traffic Signal Control through Spatio-temporal Hypergraphs.
- Core method reported by the paper: MA-SAC with a spatio-temporal hypergraph-based critic for coordinating multiple traffic-signal agents over a network.

## Compatibility mapping
- Original paper setting: multiple intersections.
- Benchmark-fair adaptation here: one intersection with four movement-stage agents contributing action preferences for the same intersection controller.
- Original paper structure: spatio-temporal hypergraph critic.
- Benchmark-fair adaptation here: critic encodes the eight movement groups with fixed hyperedges and a temporal GRU before centralized Q estimation.
- Original paper execution: multi-agent traffic-network control.
- Benchmark-fair adaptation here: decentralized agent heads with a centralized critic, collapsed to one benchmark action among the same four actions used by our baseline setup.

## Risks / deviations
- This is not a paper-faithful multi-intersection reproduction because the benchmark has only one intersection.
- The hypergraph is constructed over movement groups inside the single intersection rather than over neighboring intersections.
- Results are suitable for the benchmark-fair comparison table, but they should be labeled as an adaptation rather than an exact reproduction of the original paper's experimental setting.
"""
    RESULT_PATHS["methodology_note"].write_text(note, encoding="utf-8")


if __name__ == "__main__":
    agent, train_metrics = train()
    results = evaluate(agent)
    write_methodology_note()
    print("\nMA-SAC evaluation summary")
    print("=" * 80)
    print(
        f"Reward={results['summary']['eval_reward_mean']:.3f} | "
        f"AvgQueue={results['summary']['eval_avg_queue_length']:.3f} | "
        f"AvgWait={results['summary']['eval_avg_waiting_time']:.3f} | "
        f"Throughput={results['summary']['eval_total_throughput']:.3f} | "
        f"Delay={results['summary']['eval_total_delay']:.3f} | "
        f"PhaseChanges={results['summary']['eval_phase_changes']:.3f}"
    )
