"""
Train Deep Q-Network (DQN) Agent
==================================
Trains a Dueling Double DQN on the lightweight TrafficEnv (no SUMO needed).
Saves model checkpoints every N episodes as both .pkl and .pth files.

Usage:
    python train_dqn.py [--episodes 1000] [--mode normal|peak|asymmetric]
                        [--resume models/dqn_agent.pkl]
"""

import os
import argparse
import pickle
import numpy as np
from collections import deque

from environments.traffic_env import TrafficEnv
from agents.dqn_agent import DQNAgent
from utils.config import DQN_CONFIG, PATHS


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def moving_average(values, window=20):
    if len(values) < window:
        return np.mean(values) if values else 0.0
    return np.mean(values[-window:])


def print_status(ep, total_ep, agent, metrics, window=20):
    r  = moving_average(metrics["rewards"],    window)
    q  = moving_average(metrics["avg_queue"],  window)
    w  = moving_average(metrics["avg_wait"],   window)
    tp = moving_average(metrics["throughput"], window)
    l  = moving_average(metrics["losses"],     window) if metrics["losses"] else 0
    print(f"  EP {ep:5d}/{total_ep} | "
          f"Reward {r:7.3f} | "
          f"Queue {q:5.2f} | "
          f"Wait {w:6.1f}s | "
          f"Thput {tp:6.0f} | "
          f"Loss {l:.5f} | "
          f"ε {agent.epsilon:.4f} | "
          f"Buf {len(agent.replay_buffer):6d} | "
          f"Steps {agent.total_steps:7d}")


# ─────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────

def train_dqn(
    num_episodes:  int  = DQN_CONFIG["num_episodes"],
    mode:          str  = "normal",
    seed:          int  = 42,
    resume_path:   str  = None,
    save_freq:     int  = DQN_CONFIG["save_freq"],
    verbose_freq:  int  = 10,
):
    os.makedirs(PATHS["models_dir"], exist_ok=True)

    # Environment
    env = TrafficEnv(seed=seed, mode=mode)
    print(f"\n{'='*65}")
    print(f"  DQN Training  |  Mode: {mode}  |  Episodes: {num_episodes}")
    print(f"{'='*65}")

    # Agent – resume or fresh
    if resume_path and os.path.exists(resume_path):
        agent = DQNAgent.load(resume_path)
        print(f"  Resuming from {resume_path}")
    else:
        agent = DQNAgent()
        print(f"  {agent}")

    # Metrics tracking
    metrics = {
        "rewards":   agent.episode_rewards.copy(),
        "avg_queue": [],
        "avg_wait":  [],
        "throughput":[],
        "losses":    agent.losses.copy(),
    }
    best_reward = float("-inf")

    # ── Episode loop ────────────────────────────────────────────
    for ep in range(1, num_episodes + 1):
        raw_state, _ = env.reset()
        ep_reward    = 0.0
        ep_losses    = []
        done         = False

        while not done:
            # Agent acts
            action = agent.select_action(raw_state)

            # Environment step
            next_raw, _, reward, done, info = env.step(action)

            # Store transition
            agent.store(raw_state, action, reward, next_raw, done)

            # Train (one gradient step)
            loss = agent.train()
            if loss is not None:
                ep_losses.append(loss)

            ep_reward += reward
            raw_state  = next_raw

        # Post-episode bookkeeping
        agent.decay_epsilon()
        agent.episode_rewards.append(ep_reward)
        stats = env.get_episode_stats()

        metrics["rewards"].append(ep_reward)
        metrics["avg_queue"].append(stats["avg_queue_length"])
        metrics["avg_wait"].append(stats["avg_waiting_time"])
        metrics["throughput"].append(stats["total_throughput"])
        if ep_losses:
            metrics["losses"].extend(ep_losses)

        # Console output
        if ep % verbose_freq == 0:
            print_status(ep, num_episodes, agent, metrics)

        # Checkpoint
        if ep % save_freq == 0:
            ckpt_pkl = os.path.join(PATHS["models_dir"],
                                    f"dqn_ep{ep}.pkl")
            ckpt_pth = os.path.join(PATHS["models_dir"],
                                    f"dqn_ep{ep}.pth")
            agent.save(pkl_path=ckpt_pkl, weights_path=ckpt_pth)

        # Best model
        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save(pkl_path=PATHS["dqn_model"],
                       weights_path=PATHS["dqn_weights"])

    # ── Final save ──────────────────────────────────────────────
    agent.save(pkl_path=PATHS["dqn_model"],
               weights_path=PATHS["dqn_weights"])

    # Save metrics
    metrics_path = os.path.join(PATHS["models_dir"], "dqn_metrics.pkl")
    with open(metrics_path, "wb") as f:
        pickle.dump(metrics, f)

    print(f"\n{'='*65}")
    print(f"  Training complete!")
    print(f"  Best reward  : {best_reward:.4f}")
    print(f"  Total steps  : {agent.total_steps:,}")
    print(f"  Model saved  : {PATHS['dqn_model']}")
    print(f"  Metrics saved: {metrics_path}")
    print(f"{'='*65}")

    return agent, metrics


# ─────────────────────────────────────────────────────────────────
# Evaluation (greedy, no exploration)
# ─────────────────────────────────────────────────────────────────

def evaluate_dqn(agent: DQNAgent, num_episodes: int = 20,
                 mode: str = "normal", seed: int = 99) -> dict:
    """Evaluate a trained DQN agent (epsilon=0)."""
    env = TrafficEnv(seed=seed, mode=mode)
    original_eps = agent.epsilon
    agent.epsilon = 0.0   # pure greedy

    all_stats = []
    for _ in range(num_episodes):
        raw_state, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(raw_state)
            next_raw, _, _, done, _ = env.step(action)
            raw_state = next_raw
        all_stats.append(env.get_episode_stats())

    agent.epsilon = original_eps

    summary = {
        "avg_queue_length": np.mean([s["avg_queue_length"] for s in all_stats]),
        "avg_waiting_time": np.mean([s["avg_waiting_time"] for s in all_stats]),
        "total_throughput": np.mean([s["total_throughput"] for s in all_stats]),
        "phase_changes":    np.mean([s["phase_changes"]    for s in all_stats]),
    }
    return summary


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train DQN for traffic control")
    parser.add_argument("--episodes", type=int, default=DQN_CONFIG["num_episodes"])
    parser.add_argument("--mode",     choices=["normal","peak","asymmetric"],
                        default="normal")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--resume",   type=str, default=None,
                        help="Path to existing .pkl to resume training")
    parser.add_argument("--save-freq",type=int, default=DQN_CONFIG["save_freq"])
    args = parser.parse_args()

    agent, metrics = train_dqn(
        num_episodes = args.episodes,
        mode         = args.mode,
        seed         = args.seed,
        resume_path  = args.resume,
        save_freq    = args.save_freq,
    )

    print("\nRunning evaluation (20 episodes, greedy)…")
    eval_stats = evaluate_dqn(agent, num_episodes=20, mode=args.mode)
    print("\n  Evaluation Results:")
    for k, v in eval_stats.items():
        print(f"    {k}: {v:.3f}")


if __name__ == "__main__":
    main()
