from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from main_experiment_6action.environments.traffic_env import TrafficEnv
from main_experiment_6action.utils.config import ACTION_DEFINITIONS, MOVEMENT_KEYS


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "hybrid_enhanced"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = OUT_DIR / "hybrid_enhanced.pt"


CONFIG = {
    "seed": 42,
    "mode": "normal",
    "episodes": 360,
    "eval_episodes": 30,
    "temporal_window": 4,
    "gamma": 0.97,
    "lr": 4e-4,
    "batch_size": 64,
    "buffer_capacity": 60_000,
    "warmup_steps": 800,
    "update_interval": 4,
    "target_update_interval": 4,
    "tau": 0.04,
    "hidden_dim": 96,
    "embed_dim": 48,
    "eps_start": 1.0,
    "eps_end": 0.03,
    "priority_alpha": 0.7,
    "priority_beta_start": 0.4,
    "priority_beta_end": 1.0,
    "priority_reward_scale": 0.20,
    "priority_state_scale": 0.10,
    "prior_weight": 0.16,
    "switch_margin": 0.07,
    "stay_bonus": 0.10,
    "grad_clip": 10.0,
}


ACTION_GROUPS = {
    0: (0, 2),
    1: (1, 3),
    2: (4, 6),
    3: (5, 7),
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_state_sequence(history: deque[np.ndarray], temporal_window: int) -> np.ndarray:
    items = list(history)
    while len(items) < temporal_window:
        items.insert(0, items[0])
    return np.stack(items[-temporal_window:], axis=0)


def power_decay_epsilon(episode: int, total_episodes: int, eps_start: float, eps_end: float) -> float:
    if total_episodes <= 1:
        return eps_end
    exponent_base = (eps_end / eps_start) ** (1.0 / max(total_episodes - 1, 1))
    return max(eps_end, eps_start * (exponent_base**episode))


def latest_prior_scores_from_state(state: np.ndarray, stay_bonus: float) -> np.ndarray:
    queues = state[:8]
    waits = state[8:16]
    current_action = int(round(float(state[-1]) * 3.0))
    scores = np.zeros(4, dtype=np.float32)
    for action, (i, j) in ACTION_GROUPS.items():
        served_queue = float(queues[i] + queues[j])
        served_wait = float(waits[i] + waits[j])
        turn_bonus = 0.10 * served_queue if action in (1, 3) else 0.0
        scores[action] = served_queue + 0.35 * served_wait + turn_bonus
    max_abs = float(np.max(np.abs(scores)))
    if max_abs > 1e-6:
        scores = scores / max_abs
    scores[current_action] += stay_bonus
    return scores


def prior_scores_batch(state_seq_batch: np.ndarray, stay_bonus: float) -> np.ndarray:
    return np.stack(
        [latest_prior_scores_from_state(state_seq[-1], stay_bonus) for state_seq in state_seq_batch],
        axis=0,
    ).astype(np.float32)


class PriorityReplayBuffer:
    def __init__(self, capacity: int, alpha: float):
        self.capacity = capacity
        self.alpha = alpha
        self.data: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def __len__(self) -> int:
        return len(self.data)

    def push(self, transition, priority: float):
        if len(self.data) < self.capacity:
            self.data.append(transition)
        self.data[self.position] = transition
        self.priorities[self.position] = max(float(priority), 1e-6)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float):
        priorities = self.priorities[: len(self.data)]
        scaled = priorities**self.alpha
        probs = scaled / scaled.sum()
        indices = np.random.choice(len(self.data), batch_size, p=probs)
        transitions = [self.data[idx] for idx in indices]
        weights = (len(self.data) * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        return transitions, indices, weights.astype(np.float32)

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[int(idx)] = max(float(priority), 1e-6)


class HypergraphTemporalEncoder(nn.Module):
    def __init__(self, node_feat_dim: int = 2, embed_dim: int = 48):
        super().__init__()
        self.node_proj = nn.Linear(node_feat_dim, embed_dim)
        self.temporal = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        incidence = torch.tensor(
            [
                [1, 0, 0, 0, 1, 0, 1, 0],
                [0, 1, 0, 0, 1, 0, 0, 1],
                [0, 0, 1, 0, 0, 1, 1, 0],
                [0, 0, 0, 1, 0, 1, 0, 1],
                [1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("incidence", incidence)

    def forward(self, state_seq: torch.Tensor):
        batch, time_steps, _ = state_seq.shape
        node_feats = state_seq[:, :, :16].reshape(batch, time_steps, 8, 2)
        x = self.node_proj(node_feats)

        h = self.incidence
        h_t = h.t()
        edge_norm = h.sum(dim=1, keepdim=True).clamp_min(1.0)
        node_norm = h_t.sum(dim=1, keepdim=True).clamp_min(1.0)

        edge_msg = torch.einsum("eh,btnd->bted", h / edge_norm, x)
        node_msg = torch.einsum("ne,bted->btnd", h_t / node_norm, edge_msg)
        x = x + node_msg

        x = x.permute(0, 2, 1, 3).reshape(batch * 8, time_steps, -1)
        _, hidden = self.temporal(x)
        node_embed = hidden[-1].reshape(batch, 8, -1)
        node_embed = self.out_proj(node_embed)
        global_embed = node_embed.mean(dim=1)
        return node_embed, global_embed


class HybridQNet(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, num_actions: int = 4):
        super().__init__()
        self.encoder = HypergraphTemporalEncoder(embed_dim=embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim + 4 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value = nn.Linear(hidden_dim, 1)
        self.advantage = nn.Linear(hidden_dim, num_actions)

    def forward(self, state_seq: torch.Tensor, prior_scores: torch.Tensor):
        _, global_embed = self.encoder(state_seq)
        current_phase = state_seq[:, -1, -1:].float()
        x = torch.cat([global_embed, prior_scores, current_phase], dim=-1)
        x = self.fc(x)
        value = self.value(x)
        advantage = self.advantage(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


@dataclass
class EvalSummary:
    eval_reward_mean: float
    eval_reward_std: float
    eval_avg_queue_length: float
    eval_avg_waiting_time: float
    eval_total_throughput: float
    eval_total_delay: float
    eval_phase_changes: float


class HybridEnhancedAgent:
    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        self.online = HybridQNet(config["embed_dim"], config["hidden_dim"]).to(device)
        self.target = HybridQNet(config["embed_dim"], config["hidden_dim"]).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config["lr"])
        self.replay = PriorityReplayBuffer(config["buffer_capacity"], config["priority_alpha"])
        self.total_steps = 0
        self.train_steps = 0

    def _q_values(self, state_seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        prior = latest_prior_scores_from_state(state_seq[-1], self.config["stay_bonus"])
        with torch.no_grad():
            states_t = torch.tensor(state_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
            prior_t = torch.tensor(prior, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.online(states_t, prior_t).squeeze(0).cpu().numpy()
        return q_values, prior

    def select_action(self, state_seq: np.ndarray, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            return int(np.random.randint(4))
        q_values, prior = self._q_values(state_seq)
        blended = q_values + self.config["prior_weight"] * prior
        current_action = int(round(float(state_seq[-1, -1]) * 3.0))
        best_action = int(np.argmax(blended))
        if best_action != current_action:
            if blended[best_action] - blended[current_action] < self.config["switch_margin"]:
                return current_action
        return best_action

    def greedy_action(self, state_seq: np.ndarray) -> int:
        return self.select_action(state_seq, epsilon=0.0)

    def compute_priority(self, state_seq: np.ndarray, reward: float, td_error: float | None = None) -> float:
        state_score = float(np.mean(np.abs(state_seq)))
        reward_score = abs(float(reward))
        td_part = abs(float(td_error)) if td_error is not None else 1.0
        return (
            td_part
            + self.config["priority_reward_scale"] * reward_score
            + self.config["priority_state_scale"] * state_score
            + 1e-6
        )

    def store(self, state_seq, action, reward, next_state_seq, done):
        priority = self.compute_priority(state_seq, reward)
        self.replay.push((state_seq, action, reward, next_state_seq, done), priority)
        self.total_steps += 1

    def update(self, episode: int):
        if len(self.replay) < max(self.config["warmup_steps"], self.config["batch_size"]):
            return None

        progress = episode / max(self.config["episodes"] - 1, 1)
        beta = self.config["priority_beta_start"] + progress * (
            self.config["priority_beta_end"] - self.config["priority_beta_start"]
        )
        transitions, indices, weights = self.replay.sample(self.config["batch_size"], beta)
        states, actions, rewards, next_states, dones = zip(*transitions)

        states_np = np.array(states, dtype=np.float32)
        next_states_np = np.array(next_states, dtype=np.float32)
        state_prior_np = prior_scores_batch(states_np, self.config["stay_bonus"])
        next_prior_np = prior_scores_batch(next_states_np, self.config["stay_bonus"])

        states_t = torch.tensor(states_np, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states_np, dtype=torch.float32, device=self.device)
        state_prior_t = torch.tensor(state_prior_np, dtype=torch.float32, device=self.device)
        next_prior_t = torch.tensor(next_prior_np, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.online(states_t, state_prior_t).gather(1, actions_t)
        with torch.no_grad():
            next_online = self.online(next_states_t, next_prior_t)
            next_online = next_online + self.config["prior_weight"] * next_prior_t
            next_actions = next_online.argmax(dim=1, keepdim=True)
            next_target = self.target(next_states_t, next_prior_t).gather(1, next_actions)
            targets = rewards_t + self.config["gamma"] * next_target * (1.0 - dones_t)

        td_errors = targets - q_values
        state_penalty = states_t.abs().mean(dim=(1, 2), keepdim=True)
        reward_weight = 1.0 + self.config["priority_reward_scale"] * rewards_t.abs()
        smooth_l1 = F.smooth_l1_loss(q_values, targets, reduction="none")
        loss = weights_t * reward_weight * (smooth_l1 + self.config["priority_state_scale"] * state_penalty)
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.config["grad_clip"])
        self.optimizer.step()
        self.train_steps += 1

        new_priorities = (
            td_errors.detach().abs().squeeze(1).cpu().numpy()
            + self.config["priority_reward_scale"] * np.abs(np.array(rewards))
            + self.config["priority_state_scale"] * np.mean(np.abs(states_np), axis=(1, 2))
            + 1e-6
        )
        self.replay.update_priorities(indices, new_priorities)

        if self.train_steps % self.config["target_update_interval"] == 0:
            self._soft_update()
        return float(loss.item())

    def _soft_update(self):
        tau = self.config["tau"]
        for target_param, online_param in zip(self.target.parameters(), self.online.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + online_param.data * tau)

    def save(self, path: Path):
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "config": self.config,
            },
            path,
        )


def run_episode(agent: HybridEnhancedAgent, env: TrafficEnv, epsilon: float, train: bool, episode_idx: int = 0):
    state, _ = env.reset()
    history = deque([state.copy()], maxlen=CONFIG["temporal_window"])
    state_seq = make_state_sequence(history, CONFIG["temporal_window"])

    done = False
    total_reward = 0.0
    action_hist = {str(action): 0 for action in ACTION_DEFINITIONS}
    losses = []

    while not done:
        action = agent.select_action(state_seq, epsilon) if train else agent.greedy_action(state_seq)
        action_hist[str(action)] += 1
        next_state, _, reward, done, _ = env.step(action)
        next_history = deque(history, maxlen=CONFIG["temporal_window"])
        next_history.append(next_state.copy())
        next_state_seq = make_state_sequence(next_history, CONFIG["temporal_window"])

        if train:
            agent.store(state_seq, action, reward, next_state_seq, done)
            if agent.total_steps % CONFIG["update_interval"] == 0:
                loss = agent.update(episode_idx)
                if loss is not None:
                    losses.append(loss)

        total_reward += reward
        history = next_history
        state_seq = next_state_seq

    stats = env.get_episode_stats()
    return {
        "reward": float(total_reward),
        "avg_queue_length": float(stats["avg_queue_length"]),
        "avg_waiting_time": float(stats["avg_waiting_time"]),
        "total_throughput": float(stats["total_throughput"]),
        "total_delay": float(stats["total_delay"]),
        "phase_changes": float(stats["phase_changes"]),
        "action_histogram": action_hist,
        "loss": float(np.mean(losses)) if losses else None,
    }


def train_agent():
    device = torch.device("cpu")
    set_seed(CONFIG["seed"])
    agent = HybridEnhancedAgent(CONFIG, device)
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

    for episode in range(CONFIG["episodes"]):
        env = TrafficEnv(seed=CONFIG["seed"] + episode, mode=CONFIG["mode"])
        epsilon = power_decay_epsilon(episode, CONFIG["episodes"], CONFIG["eps_start"], CONFIG["eps_end"])
        row = run_episode(agent, env, epsilon, train=True, episode_idx=episode)
        metrics["episode_reward"].append(row["reward"])
        metrics["avg_queue_length"].append(row["avg_queue_length"])
        metrics["avg_waiting_time"].append(row["avg_waiting_time"])
        metrics["total_throughput"].append(row["total_throughput"])
        metrics["total_delay"].append(row["total_delay"])
        metrics["phase_changes"].append(row["phase_changes"])
        metrics["loss"].append(row["loss"])
        metrics["epsilon"].append(epsilon)

        if (episode + 1) % 30 == 0:
            print(
                f"ep={episode+1:4d} reward={np.mean(metrics['episode_reward'][-30:]):8.3f} "
                f"queue={np.mean(metrics['avg_queue_length'][-30:]):6.2f} "
                f"wait={np.mean(metrics['avg_waiting_time'][-30:]):7.2f} "
                f"throughput={np.mean(metrics['total_throughput'][-30:]):7.1f}"
            )
    return agent, metrics


def evaluate_agent(agent: HybridEnhancedAgent):
    rows = []
    for offset in range(CONFIG["eval_episodes"]):
        seed = CONFIG["seed"] + 100 + offset
        env = TrafficEnv(seed=seed, mode=CONFIG["mode"])
        rows.append(run_episode(agent, env, epsilon=0.0, train=False))

    summary = EvalSummary(
        eval_reward_mean=float(np.mean([row["reward"] for row in rows])),
        eval_reward_std=float(np.std([row["reward"] for row in rows])),
        eval_avg_queue_length=float(np.mean([row["avg_queue_length"] for row in rows])),
        eval_avg_waiting_time=float(np.mean([row["avg_waiting_time"] for row in rows])),
        eval_total_throughput=float(np.mean([row["total_throughput"] for row in rows])),
        eval_total_delay=float(np.mean([row["total_delay"] for row in rows])),
        eval_phase_changes=float(np.mean([row["phase_changes"] for row in rows])),
    )
    return summary, rows


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_outputs(train_metrics: dict, summary: EvalSummary, eval_rows: list[dict]):
    payload = {
        "method": {
            "name": "Hybrid-Enhanced Pri-Hyper DDQN",
            "description": (
                "Combines Pri-DDQN style prioritized double dueling Q-learning with "
                "MA-SAC inspired temporal hypergraph state encoding and a pressure-aware action prior."
            ),
        },
        "config": CONFIG,
        "summary": summary.__dict__,
        "eval_rows": eval_rows,
        "train_metrics": train_metrics,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    pri = load_json(ROOT / "Pri-DDQN" / "outputs" / "eval_results.json")["fourway_4action"]["summary"]
    ma = load_json(ROOT / "MA-SAC" / "outputs" / "eval_results.json")["summary"]
    classical = load_json(ROOT / "main_experiment_6action" / "outputs" / "final_results.json")["algorithms"]
    comparison = {
        "TD(0)": classical["TD(0)"]["summary"],
        "SARSA": classical["SARSA"]["summary"],
        "Q-Learning": classical["Q-Learning"]["summary"],
        "Pri-DDQN": pri,
        "MA-SAC": ma,
        "Hybrid-Enhanced": summary.__dict__,
    }
    (OUT_DIR / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    lines = [
        "# Hybrid Enhanced 4-Action Benchmark",
        "",
        "| Method | Reward | Std | Avg Queue | Avg Wait | Throughput | Delay | Phase Changes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, stats in comparison.items():
        lines.append(
            f"| {name} | {stats['eval_reward_mean']:.3f} | {stats['eval_reward_std']:.3f} | "
            f"{stats['eval_avg_queue_length']:.3f} | {stats['eval_avg_waiting_time']:.3f} | "
            f"{stats['eval_total_throughput']:.3f} | {stats['eval_total_delay']:.3f} | "
            f"{stats['eval_phase_changes']:.3f} |"
        )
    best_reward = max(comparison, key=lambda name: comparison[name]["eval_reward_mean"])
    lines += ["", f"- Best reward: {best_reward}"]
    (OUT_DIR / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    agent, train_metrics = train_agent()
    agent.save(CHECKPOINT_PATH)
    summary, rows = evaluate_agent(agent)
    write_outputs(train_metrics, summary, rows)
    print(json.dumps(summary.__dict__, indent=2))


if __name__ == "__main__":
    main()
