"""
Train Classical RL Agents (TD0, SARSA, Q-Learning)
====================================================
Runs all three tabular algorithms on the lightweight TrafficEnv simulator,
compares their performance, and saves trained models as .pkl files.

Usage:
    python train_classical.py [--episodes 500] [--mode normal|peak|asymmetric]
"""

import os
import pickle
import argparse
import numpy as np
from collections import defaultdict

from environments.traffic_env import TrafficEnv
from agents.classical_agents import TD0Agent, SARSAAgent, QLearningAgent
from utils.config import TABULAR_CONFIG, PATHS


# ─────────────────────────────────────────────────────────────────
# Training loops
# ─────────────────────────────────────────────────────────────────

def train_td0(env: TrafficEnv, num_episodes: int, verbose: bool = True):
    """Train TD(0) agent and return (agent, metrics)."""
    agent = TD0Agent()
    metrics = defaultdict(list)

    print("\n" + "="*60)
    print("  Training TD(0)")
    print("="*60)

    for ep in range(1, num_episodes + 1):
        _, disc_state = env.reset()
        ep_reward = 0.0
        ep_td_errors = []

        while True:
            action = agent.select_action(disc_state)
            _, next_disc, reward, done, info = env.step(action)

            td_err = agent.update(disc_state, action, reward, next_disc, done)
            ep_td_errors.append(td_err)

            ep_reward  += reward
            disc_state  = next_disc

            if done:
                break

        agent.decay_epsilon()
        agent.episode_rewards.append(ep_reward)

        stats = env.get_episode_stats()
        metrics["rewards"].append(ep_reward)
        metrics["avg_queue"].append(stats["avg_queue_length"])
        metrics["avg_wait"].append(stats["avg_waiting_time"])
        metrics["throughput"].append(stats["total_throughput"])
        metrics["td_errors"].append(float(np.mean(np.abs(ep_td_errors))))

        if verbose and ep % 50 == 0:
            recent = np.mean(metrics["rewards"][-50:])
            print(f"  EP {ep:4d} | Reward {recent:7.3f} | "
                  f"Queue {np.mean(metrics['avg_queue'][-50:]):.2f} | "
                  f"Wait {np.mean(metrics['avg_wait'][-50:]):.1f}s | "
                  f"ε={agent.epsilon:.3f} | Q-table={agent.q_table_size()}")

    agent.save(PATHS["td0_model"])
    return agent, dict(metrics)


def train_sarsa(env: TrafficEnv, num_episodes: int, verbose: bool = True):
    """Train SARSA agent and return (agent, metrics)."""
    agent = SARSAAgent()
    metrics = defaultdict(list)

    print("\n" + "="*60)
    print("  Training SARSA")
    print("="*60)

    for ep in range(1, num_episodes + 1):
        _, disc_state = env.reset()
        agent.reset_traces()
        action = agent.select_action(disc_state)
        ep_reward = 0.0
        ep_td_errors = []

        while True:
            _, next_disc, reward, done, info = env.step(action)
            next_action = agent.select_action(next_disc)

            td_err = agent.update(disc_state, action, reward,
                                  next_disc, next_action, done)
            ep_td_errors.append(td_err)

            ep_reward  += reward
            disc_state  = next_disc
            action      = next_action

            if done:
                break

        agent.decay_epsilon()
        agent.episode_rewards.append(ep_reward)

        stats = env.get_episode_stats()
        metrics["rewards"].append(ep_reward)
        metrics["avg_queue"].append(stats["avg_queue_length"])
        metrics["avg_wait"].append(stats["avg_waiting_time"])
        metrics["throughput"].append(stats["total_throughput"])
        metrics["td_errors"].append(float(np.mean(np.abs(ep_td_errors))))

        if verbose and ep % 50 == 0:
            recent = np.mean(metrics["rewards"][-50:])
            print(f"  EP {ep:4d} | Reward {recent:7.3f} | "
                  f"Queue {np.mean(metrics['avg_queue'][-50:]):.2f} | "
                  f"Wait {np.mean(metrics['avg_wait'][-50:]):.1f}s | "
                  f"ε={agent.epsilon:.3f} | Q-table={agent.q_table_size()}")

    agent.save(PATHS["sarsa_model"])
    return agent, dict(metrics)


def train_qlearning(env: TrafficEnv, num_episodes: int, verbose: bool = True):
    """Train Q-Learning agent and return (agent, metrics)."""
    agent = QLearningAgent()
    metrics = defaultdict(list)

    print("\n" + "="*60)
    print("  Training Q-Learning")
    print("="*60)

    for ep in range(1, num_episodes + 1):
        _, disc_state = env.reset()
        ep_reward = 0.0
        ep_td_errors = []

        while True:
            action = agent.select_action(disc_state)
            _, next_disc, reward, done, info = env.step(action)

            td_err = agent.update(disc_state, action, reward, next_disc, done)
            ep_td_errors.append(td_err)

            ep_reward  += reward
            disc_state  = next_disc

            if done:
                break

        agent.decay_epsilon()
        agent.episode_rewards.append(ep_reward)

        stats = env.get_episode_stats()
        metrics["rewards"].append(ep_reward)
        metrics["avg_queue"].append(stats["avg_queue_length"])
        metrics["avg_wait"].append(stats["avg_waiting_time"])
        metrics["throughput"].append(stats["total_throughput"])
        metrics["td_errors"].append(float(np.mean(np.abs(ep_td_errors))))

        if verbose and ep % 50 == 0:
            recent = np.mean(metrics["rewards"][-50:])
            print(f"  EP {ep:4d} | Reward {recent:7.3f} | "
                  f"Queue {np.mean(metrics['avg_queue'][-50:]):.2f} | "
                  f"Wait {np.mean(metrics['avg_wait'][-50:]):.1f}s | "
                  f"ε={agent.epsilon:.3f} | Q-table={agent.q_table_size()}")

    agent.save(PATHS["qlearning_model"])
    return agent, dict(metrics)


# ─────────────────────────────────────────────────────────────────
# Comparison Report
# ─────────────────────────────────────────────────────────────────

def print_comparison(results: dict, last_n: int = 50):
    print("\n" + "="*70)
    print("  CLASSICAL RL COMPARISON REPORT")
    print("="*70)
    header = f"{'Algorithm':<15} {'Reward':>10} {'AvgQueue':>10} "
    header += f"{'AvgWait(s)':>12} {'Throughput':>12}"
    print(header)
    print("-"*70)

    for name, metrics in results.items():
        r = np.mean(metrics["rewards"][-last_n:])
        q = np.mean(metrics["avg_queue"][-last_n:])
        w = np.mean(metrics["avg_wait"][-last_n:])
        t = np.mean(metrics["throughput"][-last_n:])
        print(f"{name:<15} {r:>10.3f} {q:>10.2f} {w:>12.1f} {t:>12.0f}")

    print("="*70)
    best = max(results, key=lambda k: np.mean(results[k]["rewards"][-last_n:]))
    print(f"\n  ✓ Best algorithm: {best}")
    print("="*70)


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train classical RL agents")
    parser.add_argument("--episodes", type=int,
                        default=TABULAR_CONFIG["num_episodes"])
    parser.add_argument("--mode", choices=["normal","peak","asymmetric"],
                        default="normal")
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    os.makedirs(PATHS["models_dir"], exist_ok=True)

    env = TrafficEnv(seed=args.seed, mode=args.mode)
    print(f"\nEnvironment: TrafficEnv | Mode: {args.mode} | "
          f"Episodes: {args.episodes}")

    results = {}

    _, td0_metrics     = train_td0(env,      args.episodes)
    _, sarsa_metrics   = train_sarsa(env,    args.episodes)
    _, qlearn_metrics  = train_qlearning(env, args.episodes)

    results["TD(0)"]     = td0_metrics
    results["SARSA"]     = sarsa_metrics
    results["Q-Learning"]= qlearn_metrics

    print_comparison(results)

    # Save all metrics together for later plotting
    metrics_path = os.path.join(PATHS["models_dir"], "classical_metrics.pkl")
    with open(metrics_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nMetrics saved → {metrics_path}")


if __name__ == "__main__":
    main()
