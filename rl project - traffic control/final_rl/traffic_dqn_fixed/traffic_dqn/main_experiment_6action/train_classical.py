"""
Train TD(0), SARSA, and Q-learning on the movement-based 4-action setup.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import defaultdict

import numpy as np

from main_experiment_6action.agents.classical_agents import (
    QLearningAgent,
    SARSAAgent,
    TD0Agent,
)
from main_experiment_6action.environments.traffic_env import TrafficEnv
from main_experiment_6action.utils.config import (
    ACTION_DEFINITIONS,
    MODELS_DIR,
    OUTPUTS_DIR,
    PATHS,
    TABULAR_CONFIG,
)


def run_td0(env: TrafficEnv, episodes: int):
    agent = TD0Agent()
    metrics = defaultdict(list)

    for episode in range(1, episodes + 1):
        _, state = env.reset()
        episode_reward = 0.0
        td_errors = []

        while True:
            action = agent.select_action(state)
            _, next_state, reward, done, _ = env.step(action)
            td_errors.append(agent.update(state, reward, next_state, done))
            episode_reward += reward
            state = next_state
            if done:
                break

        agent.decay_epsilon()
        agent.episode_rewards.append(episode_reward)
        stats = env.get_episode_stats()
        metrics["rewards"].append(episode_reward)
        metrics["avg_queue"].append(stats["avg_queue_length"])
        metrics["avg_wait"].append(stats["avg_waiting_time"])
        metrics["throughput"].append(stats["total_throughput"])
        metrics["total_delay"].append(stats["total_delay"])
        metrics["phase_changes"].append(stats["phase_changes"])
        metrics["td_errors"].append(float(np.mean(np.abs(td_errors))))

        if episode % 50 == 0:
            print(
                f"TD(0)   ep={episode:4d} reward={np.mean(metrics['rewards'][-50:]):8.3f} "
                f"queue={np.mean(metrics['avg_queue'][-50:]):6.2f} "
                f"wait={np.mean(metrics['avg_wait'][-50:]):8.2f} "
                f"eps={agent.epsilon:6.3f}"
            )

    agent.save(PATHS["td0_model"])
    return agent, dict(metrics)


def run_sarsa(env: TrafficEnv, episodes: int):
    agent = SARSAAgent()
    metrics = defaultdict(list)

    for episode in range(1, episodes + 1):
        _, state = env.reset()
        action = agent.select_action(state)
        episode_reward = 0.0
        td_errors = []

        while True:
            _, next_state, reward, done, _ = env.step(action)
            next_action = agent.select_action(next_state)
            td_errors.append(agent.update(state, action, reward, next_state, next_action, done))
            episode_reward += reward
            state = next_state
            action = next_action
            if done:
                break

        agent.decay_epsilon()
        agent.episode_rewards.append(episode_reward)
        stats = env.get_episode_stats()
        metrics["rewards"].append(episode_reward)
        metrics["avg_queue"].append(stats["avg_queue_length"])
        metrics["avg_wait"].append(stats["avg_waiting_time"])
        metrics["throughput"].append(stats["total_throughput"])
        metrics["total_delay"].append(stats["total_delay"])
        metrics["phase_changes"].append(stats["phase_changes"])
        metrics["td_errors"].append(float(np.mean(np.abs(td_errors))))

        if episode % 50 == 0:
            print(
                f"SARSA   ep={episode:4d} reward={np.mean(metrics['rewards'][-50:]):8.3f} "
                f"queue={np.mean(metrics['avg_queue'][-50:]):6.2f} "
                f"wait={np.mean(metrics['avg_wait'][-50:]):8.2f} "
                f"eps={agent.epsilon:6.3f}"
            )

    agent.save(PATHS["sarsa_model"])
    return agent, dict(metrics)


def run_qlearning(env: TrafficEnv, episodes: int):
    agent = QLearningAgent()
    metrics = defaultdict(list)

    for episode in range(1, episodes + 1):
        _, state = env.reset()
        episode_reward = 0.0
        td_errors = []

        while True:
            action = agent.select_action(state)
            _, next_state, reward, done, _ = env.step(action)
            td_errors.append(agent.update(state, action, reward, next_state, done))
            episode_reward += reward
            state = next_state
            if done:
                break

        agent.decay_epsilon()
        agent.episode_rewards.append(episode_reward)
        stats = env.get_episode_stats()
        metrics["rewards"].append(episode_reward)
        metrics["avg_queue"].append(stats["avg_queue_length"])
        metrics["avg_wait"].append(stats["avg_waiting_time"])
        metrics["throughput"].append(stats["total_throughput"])
        metrics["total_delay"].append(stats["total_delay"])
        metrics["phase_changes"].append(stats["phase_changes"])
        metrics["td_errors"].append(float(np.mean(np.abs(td_errors))))

        if episode % 50 == 0:
            print(
                f"Q-Learn ep={episode:4d} reward={np.mean(metrics['rewards'][-50:]):8.3f} "
                f"queue={np.mean(metrics['avg_queue'][-50:]):6.2f} "
                f"wait={np.mean(metrics['avg_wait'][-50:]):8.2f} "
                f"eps={agent.epsilon:6.3f}"
            )

    agent.save(PATHS["qlearning_model"])
    return agent, dict(metrics)


def evaluate_agent(agent, mode: str, seeds: list[int]):
    eval_rows = []
    for seed in seeds:
        env = TrafficEnv(seed=seed, mode=mode)
        _, state = env.reset()
        total_reward = 0.0
        action_hist = {str(action): 0 for action in ACTION_DEFINITIONS}

        while True:
            action = agent.greedy_action(state)
            action_hist[str(action)] += 1
            _, next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
            if done:
                break

        stats = env.get_episode_stats()
        eval_rows.append(
            {
                "reward": total_reward,
                "avg_queue_length": stats["avg_queue_length"],
                "avg_waiting_time": stats["avg_waiting_time"],
                "total_throughput": stats["total_throughput"],
                "total_delay": stats["total_delay"],
                "phase_changes": stats["phase_changes"],
                "action_histogram": action_hist,
            }
        )
    return eval_rows


def summarise(train_metrics: dict, eval_rows: list[dict]):
    summary = {
        "train_reward_last50": float(np.mean(train_metrics["rewards"][-50:])),
        "train_avg_queue_last50": float(np.mean(train_metrics["avg_queue"][-50:])),
        "train_avg_wait_last50": float(np.mean(train_metrics["avg_wait"][-50:])),
        "train_throughput_last50": float(np.mean(train_metrics["throughput"][-50:])),
        "eval_reward_mean": float(np.mean([row["reward"] for row in eval_rows])),
        "eval_reward_std": float(np.std([row["reward"] for row in eval_rows])),
        "eval_avg_queue_length": float(np.mean([row["avg_queue_length"] for row in eval_rows])),
        "eval_avg_waiting_time": float(np.mean([row["avg_waiting_time"] for row in eval_rows])),
        "eval_total_throughput": float(np.mean([row["total_throughput"] for row in eval_rows])),
        "eval_total_delay": float(np.mean([row["total_delay"] for row in eval_rows])),
        "eval_phase_changes": float(np.mean([row["phase_changes"] for row in eval_rows])),
    }
    return summary


def write_results(results: dict):
    with open(PATHS["results_json"], "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    fieldnames = [
        "algorithm",
        "train_reward_last50",
        "train_avg_queue_last50",
        "train_avg_wait_last50",
        "train_throughput_last50",
        "eval_reward_mean",
        "eval_reward_std",
        "eval_avg_queue_length",
        "eval_avg_waiting_time",
        "eval_total_throughput",
        "eval_total_delay",
        "eval_phase_changes",
    ]
    with open(PATHS["results_csv"], "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for algorithm, payload in results["algorithms"].items():
            row = {"algorithm": algorithm}
            row.update(payload["summary"])
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=TABULAR_CONFIG["num_episodes"])
    parser.add_argument("--mode", choices=["normal", "peak", "asymmetric"], default="normal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=TABULAR_CONFIG["eval_episodes"])
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    results = {
        "setup": {
            "mode": args.mode,
            "train_seed": args.seed,
            "episodes": args.episodes,
            "eval_episodes": args.eval_episodes,
            "decision_actions": {
                str(action): {
                    "name": spec["name"],
                    "sumo_phase_index": spec["sumo_phase_index"],
                }
                for action, spec in ACTION_DEFINITIONS.items()
            },
        },
        "algorithms": {},
    }

    env = TrafficEnv(seed=args.seed, mode=args.mode)
    td0_agent, td0_metrics = run_td0(env, args.episodes)
    env = TrafficEnv(seed=args.seed, mode=args.mode)
    sarsa_agent, sarsa_metrics = run_sarsa(env, args.episodes)
    env = TrafficEnv(seed=args.seed, mode=args.mode)
    qlearning_agent, qlearning_metrics = run_qlearning(env, args.episodes)

    metric_bundle = {
        "TD(0)": td0_metrics,
        "SARSA": sarsa_metrics,
        "Q-Learning": qlearning_metrics,
    }
    with open(PATHS["metrics"], "wb") as handle:
        pickle.dump(metric_bundle, handle)

    eval_seeds = list(range(args.seed + 100, args.seed + 100 + args.eval_episodes))
    algorithm_artifacts = {
        "TD(0)": (td0_agent, td0_metrics),
        "SARSA": (sarsa_agent, sarsa_metrics),
        "Q-Learning": (qlearning_agent, qlearning_metrics),
    }
    for name, (agent, metrics) in algorithm_artifacts.items():
        eval_rows = evaluate_agent(agent, args.mode, eval_seeds)
        results["algorithms"][name] = {
            "summary": summarise(metrics, eval_rows),
            "eval_rows": eval_rows,
        }

    write_results(results)

    print("\nFinal evaluation summary")
    print("=" * 80)
    print(
        f"{'Algorithm':<12} {'EvalReward':>12} {'AvgQueue':>12} {'AvgWait':>12} "
        f"{'Throughput':>12} {'Delay':>12} {'PhChanges':>12}"
    )
    print("-" * 80)
    for name, payload in results["algorithms"].items():
        summary = payload["summary"]
        print(
            f"{name:<12} {summary['eval_reward_mean']:>12.3f} "
            f"{summary['eval_avg_queue_length']:>12.3f} "
            f"{summary['eval_avg_waiting_time']:>12.3f} "
            f"{summary['eval_total_throughput']:>12.3f} "
            f"{summary['eval_total_delay']:>12.3f} "
            f"{summary['eval_phase_changes']:>12.3f}"
        )


if __name__ == "__main__":
    main()
