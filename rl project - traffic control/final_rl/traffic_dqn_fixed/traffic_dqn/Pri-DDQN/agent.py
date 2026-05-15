from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from model import PriDDQNNet, power_decay_epsilon
from replay_buffer import PriorityReplayBuffer


@dataclass
class TrainArtifacts:
    reward: float
    avg_queue_length: float
    avg_waiting_time: float
    total_throughput: float
    total_delay: float
    phase_changes: float
    loss: float | None
    epsilon: float


class PriDDQNAgent:
    def __init__(self, state_dim: int, action_dim: int, config: dict, device: torch.device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = device

        self.online = PriDDQNNet(state_dim, action_dim, config["hidden_dim"]).to(device)
        self.target = PriDDQNNet(state_dim, action_dim, config["hidden_dim"]).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config["lr"])
        self.replay = PriorityReplayBuffer(config["buffer_capacity"], config["priority_alpha"])
        self.total_steps = 0
        self.train_steps = 0

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            return int(np.random.randint(self.action_dim))
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.online(state_t)
        return int(q.argmax(dim=1).item())

    def greedy_action(self, state: np.ndarray) -> int:
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.online(state_t)
        return int(q.argmax(dim=1).item())

    def compute_priority(self, state: np.ndarray, reward: float, td_error: float | None = None) -> float:
        state_score = float(np.mean(np.abs(state)))
        reward_score = abs(float(reward))
        td_part = abs(float(td_error)) if td_error is not None else 1.0
        return (
            td_part
            + self.config["priority_reward_scale"] * reward_score
            + self.config["priority_state_scale"] * state_score
            + 1e-6
        )

    def store(self, state, action, reward, next_state, done):
        priority = self.compute_priority(state, reward)
        self.replay.push((state, action, reward, next_state, done), priority)
        self.total_steps += 1

    def update(self, episode: int, total_episodes: int):
        if len(self.replay) < max(self.config["warmup_steps"], self.config["batch_size"]):
            return None

        progress = episode / max(total_episodes - 1, 1)
        beta = self.config["priority_beta_start"] + progress * (
            self.config["priority_beta_end"] - self.config["priority_beta_start"]
        )
        transitions, indices, weights = self.replay.sample(self.config["batch_size"], beta)
        states, actions, rewards, next_states, dones = zip(*transitions)

        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.online(states_t).gather(1, actions_t)
        with torch.no_grad():
            next_actions = self.online(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target(next_states_t).gather(1, next_actions)
            targets = rewards_t + self.config["gamma"] * next_q * (1.0 - dones_t)

        td_errors = targets - q_values
        state_penalty = states_t.abs().mean(dim=1, keepdim=True)
        reward_weight = 1.0 + self.config["priority_reward_scale"] * rewards_t.abs()
        augmented_loss = F.smooth_l1_loss(q_values, targets, reduction="none")
        augmented_loss = augmented_loss + self.config["priority_state_scale"] * state_penalty
        loss = (weights_t * reward_weight * augmented_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.train_steps += 1

        new_priorities = (
            td_errors.detach().abs().squeeze(1).cpu().numpy()
            + self.config["priority_reward_scale"] * np.abs(np.array(rewards))
            + self.config["priority_state_scale"] * np.mean(np.abs(np.array(states)), axis=1)
            + 1e-6
        )
        self.replay.update_priorities(indices, new_priorities)

        if self.train_steps % self.config["target_update_interval"] == 0:
            self._soft_update(self.config["tau"])

        return float(loss.item())

    def _soft_update(self, tau: float):
        for target_param, online_param in zip(self.target.parameters(), self.online.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + online_param.data * tau)

    def epsilon_for_episode(self, episode: int, total_episodes: int) -> float:
        return power_decay_epsilon(
            episode=episode,
            total_episodes=total_episodes,
            eps_start=self.config["eps_start"],
            eps_end=self.config["eps_end"],
        )

    def save(self, path):
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "config": self.config,
            },
            path,
        )
