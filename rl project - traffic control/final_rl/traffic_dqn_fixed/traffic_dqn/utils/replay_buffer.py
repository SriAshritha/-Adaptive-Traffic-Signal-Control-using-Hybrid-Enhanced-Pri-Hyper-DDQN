"""
Experience Replay Buffer
========================
Implements:
  - Standard uniform replay buffer
  - Prioritized Experience Replay (PER) – optional upgrade
"""

import random
import numpy as np
from collections import deque
import pickle


class ReplayBuffer:
    """Uniform experience replay buffer used by DQN."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    # ── public API ────────────────────────────────────────────────
    def push(self, state, action, reward, next_state, done):
        """Store a transition tuple."""
        self.buffer.append((
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int):
        """Uniformly sample a mini-batch; returns six numpy arrays."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones,   dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)

    def is_ready(self, min_size: int) -> bool:
        return len(self) >= min_size

    # ── persistence ───────────────────────────────────────────────
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(list(self.buffer), f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.buffer = deque(data, maxlen=self.capacity)


class PrioritizedReplayBuffer:
    """
    Proportional Prioritized Experience Replay (PER).
    Samples transitions with probability ∝ |TD-error|^α.

    Reference: Schaul et al., 2015 – "Prioritized Experience Replay"
    """

    def __init__(self, capacity: int, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_frames: int = 100_000):
        self.capacity    = capacity
        self.alpha       = alpha       # priority exponent
        self.beta_start  = beta_start  # IS-weight exponent (annealed → 1)
        self.beta_frames = beta_frames
        self.frame       = 1

        self.buffer: list = []
        self.priorities  = np.zeros(capacity, dtype=np.float32)
        self.pos         = 0

    # ── internal helpers ──────────────────────────────────────────
    def _beta(self) -> float:
        return min(1.0, self.beta_start +
                   self.frame * (1.0 - self.beta_start) / self.beta_frames)

    # ── public API ────────────────────────────────────────────────
    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities[:len(self.buffer)].max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        )
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        n = len(self.buffer)
        priorities = self.priorities[:n]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(n, batch_size, p=probs, replace=False)
        samples = [self.buffer[i] for i in indices]

        # Importance-sampling weights
        beta = self._beta()
        self.frame += 1
        total = n
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones,   dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = abs(err) + 1e-6

    def __len__(self):
        return len(self.buffer)

    def is_ready(self, min_size: int) -> bool:
        return len(self) >= min_size
