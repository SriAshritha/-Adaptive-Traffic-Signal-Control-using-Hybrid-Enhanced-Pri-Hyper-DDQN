"""
Deep Q-Network agent with pressure-aware features and prioritized replay.
"""

import os
import pickle
from typing import List, Optional

import numpy as np

from utils.config import DQN_CONFIG, PATHS
from utils.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class StandardQNetwork(nn.Module):
        def __init__(self, state_dim: int, action_dim: int, hidden_layers: List[int]):
            super().__init__()
            layers = []
            in_dim = state_dim
            for hidden_dim in hidden_layers:
                layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, action_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


    class DuelingQNetwork(nn.Module):
        def __init__(self, state_dim: int, action_dim: int, hidden_layers: List[int]):
            super().__init__()
            layers = []
            in_dim = state_dim
            for hidden_dim in hidden_layers:
                layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
                in_dim = hidden_dim
            self.feature_net = nn.Sequential(*layers)
            self.value_stream = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 1))
            self.adv_stream = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, action_dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.feature_net(x)
            value = self.value_stream(features)
            advantage = self.adv_stream(features)
            return value + advantage - advantage.mean(dim=1, keepdim=True)
else:
    class StandardQNetwork:
        pass


    class DuelingQNetwork:
        pass


class DQNAgent:
    def __init__(
        self,
        state_dim: int = None,
        action_dim: int = None,
        hidden_layers: list = None,
        alpha: float = None,
        gamma: float = None,
        epsilon_start: float = None,
        epsilon_end: float = None,
        epsilon_decay: float = None,
        batch_size: int = None,
        buffer_capacity: int = None,
        target_update: int = None,
        dueling: bool = None,
        double_dqn: bool = None,
        device: str = "auto",
        prioritized_replay: bool = None,
        soft_target_tau: float = None,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for DQNAgent.")

        cfg = DQN_CONFIG
        self.state_dim = state_dim or cfg["state_dim"]
        self.action_dim = action_dim or cfg["action_dim"]
        self.hidden_layers = hidden_layers or cfg["hidden_layers"]
        self.alpha = alpha or cfg["alpha"]
        self.gamma = gamma or cfg["gamma"]
        self.epsilon = epsilon_start if epsilon_start is not None else cfg["epsilon_start"]
        self.epsilon_end = epsilon_end if epsilon_end is not None else cfg["epsilon_end"]
        self.epsilon_decay = epsilon_decay or cfg["epsilon_decay"]
        self.batch_size = batch_size or cfg["batch_size"]
        self.target_update = target_update or cfg["target_update_freq"]
        self.dueling = dueling if dueling is not None else cfg["dueling"]
        self.double_dqn = double_dqn if double_dqn is not None else cfg["double_dqn"]
        self.prioritized_replay = (
            prioritized_replay if prioritized_replay is not None else cfg.get("prioritized_replay", False)
        )
        self.soft_target_tau = soft_target_tau if soft_target_tau is not None else cfg.get("soft_target_tau", 0.0)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        net_cls = DuelingQNetwork if self.dueling else StandardQNetwork
        self.online_net = net_cls(self.state_dim, self.action_dim, self.hidden_layers).to(self.device)
        self.target_net = net_cls(self.state_dim, self.action_dim, self.hidden_layers).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.alpha)

        capacity = buffer_capacity or cfg["buffer_capacity"]
        if self.prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=capacity,
                alpha=cfg.get("priority_alpha", 0.6),
                beta_start=cfg.get("priority_beta_start", 0.4),
                beta_frames=cfg.get("priority_beta_frames", 200000),
            )
        else:
            self.replay_buffer = ReplayBuffer(capacity)

        self.total_steps = 0
        self.train_steps = 0
        self.episode_rewards = []
        self.losses = []

    def _encode_state(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32)
        if state.shape[0] == self.state_dim:
            return state
        q = state[:4]
        w = state[4:8]
        phase = state[8:9]
        ns_queue = np.array([q[0] + q[1]], dtype=np.float32)
        ew_queue = np.array([q[2] + q[3]], dtype=np.float32)
        ns_wait = np.array([w[0] + w[1]], dtype=np.float32)
        ew_wait = np.array([w[2] + w[3]], dtype=np.float32)
        pressure = (ns_queue + 0.5 * ns_wait) - (ew_queue + 0.5 * ew_wait)
        congestion = np.array([q.mean() + 0.5 * w.mean()], dtype=np.float32)
        return np.concatenate([q, w, phase, ns_queue, ew_queue, pressure, congestion]).astype(np.float32)

    def select_action(self, state: np.ndarray) -> int:
        encoded = self._encode_state(state)
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_dim))
        return self._greedy_action(encoded)

    def _greedy_action(self, encoded_state: np.ndarray) -> int:
        self.online_net.eval()
        with torch.no_grad():
            state_t = torch.as_tensor(encoded_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.online_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.push(
            self._encode_state(state),
            action,
            reward,
            self._encode_state(next_state),
            done,
        )
        self.total_steps += 1

    def train(self) -> Optional[float]:
        min_size = DQN_CONFIG["min_replay_size"]
        if not self.replay_buffer.is_ready(min_size):
            return None

        self.online_net.train()
        prioritized = isinstance(self.replay_buffer, PrioritizedReplayBuffer)
        if prioritized:
            states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.batch_size)
            weights_t = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            indices = None
            weights_t = torch.ones(self.batch_size, dtype=torch.float32, device=self.device)

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.online_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.online_net(next_states_t).argmax(dim=1)
                next_q = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q = self.target_net(next_states_t).max(dim=1)[0]
            targets = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        td_errors = targets - q_values
        losses = F.smooth_l1_loss(q_values, targets, reduction="none")
        loss = (weights_t * losses).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.train_steps += 1
        loss_value = float(loss.item())
        self.losses.append(loss_value)

        if prioritized:
            self.replay_buffer.update_priorities(indices, np.abs(td_errors.detach().cpu().numpy()) + 1e-5)

        if self.soft_target_tau and self.soft_target_tau > 0:
            self._soft_update()
        elif self.train_steps % self.target_update == 0:
            self._hard_update()

        return loss_value

    def _hard_update(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def _soft_update(self):
        tau = self.soft_target_tau
        with torch.no_grad():
            for target_param, online_param in zip(self.target_net.parameters(), self.online_net.parameters()):
                target_param.data.mul_(1.0 - tau).add_(tau * online_param.data)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, pkl_path: str = None, weights_path: str = None):
        pkl_path = pkl_path or PATHS["dqn_model"]
        weights_path = weights_path or PATHS["dqn_weights"]
        os.makedirs(os.path.dirname(pkl_path) or ".", exist_ok=True)

        torch.save(
            {
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            weights_path,
        )
        payload = {
            "config": {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "hidden_layers": self.hidden_layers,
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay": self.epsilon_decay,
                "batch_size": self.batch_size,
                "target_update": self.target_update,
                "dueling": self.dueling,
                "double_dqn": self.double_dqn,
                "prioritized_replay": self.prioritized_replay,
                "soft_target_tau": self.soft_target_tau,
            },
            "training": {
                "total_steps": self.total_steps,
                "train_steps": self.train_steps,
                "episode_rewards": self.episode_rewards,
                "losses": self.losses[-1000:],
            },
            "weights_path": weights_path,
        }
        with open(pkl_path, "wb") as handle:
            pickle.dump(payload, handle)
        print(f"[DQNAgent] Saved -> {pkl_path} | {weights_path}")

    @classmethod
    def load(cls, pkl_path: str = None, device: str = "auto") -> "DQNAgent":
        load_path = pkl_path or PATHS["dqn_model"]
        with open(load_path, "rb") as handle:
            payload = pickle.load(handle)

        cfg = payload["config"]
        agent = cls(
            state_dim=cfg["state_dim"],
            action_dim=cfg["action_dim"],
            hidden_layers=cfg["hidden_layers"],
            alpha=cfg["alpha"],
            gamma=cfg["gamma"],
            epsilon_start=cfg["epsilon"],
            epsilon_end=cfg["epsilon_end"],
            epsilon_decay=cfg["epsilon_decay"],
            batch_size=cfg["batch_size"],
            target_update=cfg["target_update"],
            dueling=cfg["dueling"],
            double_dqn=cfg["double_dqn"],
            prioritized_replay=cfg.get("prioritized_replay"),
            soft_target_tau=cfg.get("soft_target_tau"),
            device=device,
        )
        training = payload["training"]
        agent.total_steps = training["total_steps"]
        agent.train_steps = training["train_steps"]
        agent.episode_rewards = training["episode_rewards"]
        agent.losses = training["losses"]

        weights_path = payload["weights_path"]
        if os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=agent.device)
            agent.online_net.load_state_dict(checkpoint["online_net"])
            agent.target_net.load_state_dict(checkpoint["target_net"])
            agent.optimizer.load_state_dict(checkpoint["optimizer"])
            print(f"[DQNAgent] Loaded -> {load_path} | {weights_path}")
        else:
            print(f"[DQNAgent] WARNING: weights file not found at {weights_path}")
        return agent

    def __repr__(self):
        arch = "Dueling" if self.dueling else "Standard"
        ddqn = "Double" if self.double_dqn else "Single"
        replay = "PER" if self.prioritized_replay else "Uniform"
        return (
            f"DQNAgent({arch}, {ddqn}, {replay}, "
            f"state={self.state_dim}, actions={self.action_dim}, "
            f"epsilon={self.epsilon:.4f}, steps={self.total_steps})"
        )
