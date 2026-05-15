import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from environments.traffic_env import TrafficEnv


CONFIG = {
    "seed": 42,
    "mode": "normal",
    "episodes": 500,
    "eval_episodes": 30,
    "batch_size": 64,
    "buffer_capacity": 50_000,
    "warmup_steps": 512,
    "update_interval": 8,
    "gamma": 0.99,
    "tau": 0.01,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "alpha_lr": 1e-4,
    "hidden_dim": 64,
    "target_entropy_scale": 0.8,
}


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32, device=device),
            torch.tensor(actions, dtype=torch.int64, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(-1),
            torch.tensor(np.array(next_states), dtype=torch.float32, device=device),
            torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(-1),
        )


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class MASAC2Action:
    def __init__(self, state_dim: int, device: torch.device):
        self.device = device
        self.actor = MLP(state_dim, 2, CONFIG["hidden_dim"]).to(device)
        self.critic1 = MLP(state_dim + 2, 1, CONFIG["hidden_dim"]).to(device)
        self.critic2 = MLP(state_dim + 2, 1, CONFIG["hidden_dim"]).to(device)
        self.target1 = MLP(state_dim + 2, 1, CONFIG["hidden_dim"]).to(device)
        self.target2 = MLP(state_dim + 2, 1, CONFIG["hidden_dim"]).to(device)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=CONFIG["actor_lr"])
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=CONFIG["critic_lr"])
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=CONFIG["critic_lr"])
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=CONFIG["alpha_lr"])
        self.target_entropy = -CONFIG["target_entropy_scale"] * 2.0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state: np.ndarray, greedy: bool = False):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(state_t)
            probs = torch.softmax(logits, dim=-1)
            if greedy:
                action = torch.argmax(probs, dim=-1)
            else:
                action = torch.distributions.Categorical(probs=probs).sample()
        return int(action.item())

    def q_values(self, critic, states):
        outputs = []
        for action_idx in range(2):
            a = F.one_hot(
                torch.full((states.size(0),), action_idx, device=self.device, dtype=torch.long),
                num_classes=2,
            ).float()
            outputs.append(critic(torch.cat([states, a], dim=1)))
        return torch.cat(outputs, dim=1)

    def update(self, batch):
        states, actions, rewards, next_states, dones = batch
        actions_one_hot = F.one_hot(actions, num_classes=2).float()

        with torch.no_grad():
            next_logits = self.actor(next_states)
            next_probs = torch.softmax(next_logits, dim=-1)
            next_log_probs = torch.log(next_probs.clamp_min(1e-8))
            q1_next = self.q_values(self.target1, next_states)
            q2_next = self.q_values(self.target2, next_states)
            min_q_next = torch.min(q1_next, q2_next)
            v_next = (next_probs * (min_q_next - self.alpha.detach() * next_log_probs)).sum(dim=1, keepdim=True)
            target_q = rewards + (1.0 - dones) * CONFIG["gamma"] * v_next

        sa = torch.cat([states, actions_one_hot], dim=1)
        q1 = self.critic1(sa)
        q2 = self.critic2(sa)
        loss1 = F.mse_loss(q1, target_q)
        loss2 = F.mse_loss(q2, target_q)

        self.critic1_opt.zero_grad()
        loss1.backward()
        self.critic1_opt.step()
        self.critic2_opt.zero_grad()
        loss2.backward()
        self.critic2_opt.step()

        logits = self.actor(states)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs.clamp_min(1e-8))
        min_q = torch.min(self.q_values(self.critic1, states), self.q_values(self.critic2, states))
        actor_loss = (probs * (self.alpha.detach() * log_probs - min_q)).sum(dim=1).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        entropy = -(probs * log_probs).sum(dim=1, keepdim=True)
        alpha_loss = -(self.log_alpha * (entropy.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self._soft_update(self.critic1, self.target1)
        self._soft_update(self.critic2, self.target2)

    def _soft_update(self, source: nn.Module, target: nn.Module):
        for src, tgt in zip(source.parameters(), target.parameters()):
            tgt.data.mul_(1.0 - CONFIG["tau"]).add_(CONFIG["tau"] * src.data)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train():
    out_dir = THIS_DIR / "outputs"
    out_dir.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(CONFIG["seed"])
    env = TrafficEnv(seed=CONFIG["seed"], mode=CONFIG["mode"])
    state, _ = env.reset()
    agent = MASAC2Action(len(state), device)
    replay = ReplayBuffer(CONFIG["buffer_capacity"])
    total_steps = 0
    metrics = defaultdict(list)

    for episode in range(1, CONFIG["episodes"] + 1):
        state, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(state, greedy=False)
            next_state, _, reward, done, _ = env.step(action)
            replay.push((state, action, reward, next_state, done))
            state = next_state
            ep_reward += reward
            total_steps += 1
            if len(replay) >= CONFIG["warmup_steps"] and total_steps % CONFIG["update_interval"] == 0:
                batch = replay.sample(CONFIG["batch_size"], device)
                agent.update(batch)

        stats = env.get_episode_stats()
        metrics["episode_reward"].append(float(ep_reward))
        metrics["avg_queue_length"].append(float(stats["avg_queue_length"]))
        metrics["avg_waiting_time"].append(float(stats["avg_waiting_time"]))
        metrics["total_throughput"].append(float(stats["total_throughput"]))
        metrics["phase_changes"].append(float(stats["phase_changes"]))
        if episode % 50 == 0:
            print(
                f"single_2action ep={episode:4d} reward={np.mean(metrics['episode_reward'][-50:]):8.3f} "
                f"queue={np.mean(metrics['avg_queue_length'][-50:]):6.2f} "
                f"wait={np.mean(metrics['avg_waiting_time'][-50:]):7.2f} "
                f"throughput={np.mean(metrics['total_throughput'][-50:]):7.1f}"
            )
    (out_dir / "train_single_2action.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return agent


def evaluate(agent):
    rows = []
    for offset in range(CONFIG["eval_episodes"]):
        seed = CONFIG["seed"] + 100 + offset
        env = TrafficEnv(seed=seed, mode=CONFIG["mode"])
        state, _ = env.reset()
        total_reward = 0.0
        action_hist = {"0": 0, "1": 0}
        done = False
        while not done:
            action = agent.select_action(state, greedy=True)
            action_hist[str(action)] += 1
            next_state, _, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
        stats = env.get_episode_stats()
        rows.append(
            {
                "seed": seed,
                "reward": float(total_reward),
                "avg_queue_length": float(stats["avg_queue_length"]),
                "avg_waiting_time": float(stats["avg_waiting_time"]),
                "total_throughput": float(stats["total_throughput"]),
                "phase_changes": float(stats["phase_changes"]),
                "action_histogram": action_hist,
            }
        )
    summary = {
        "eval_reward_mean": float(np.mean([r["reward"] for r in rows])),
        "eval_reward_std": float(np.std([r["reward"] for r in rows])),
        "eval_avg_queue_length": float(np.mean([r["avg_queue_length"] for r in rows])),
        "eval_avg_waiting_time": float(np.mean([r["avg_waiting_time"] for r in rows])),
        "eval_total_throughput": float(np.mean([r["total_throughput"] for r in rows])),
        "eval_total_delay": None,
        "eval_phase_changes": float(np.mean([r["phase_changes"] for r in rows])),
    }
    out = {"summary": summary, "rows": rows}
    (THIS_DIR / "outputs" / "eval_single_2action.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    agent = train()
    result = evaluate(agent)
    print("\nMA-SAC single_2action summary")
    print(result["summary"])
