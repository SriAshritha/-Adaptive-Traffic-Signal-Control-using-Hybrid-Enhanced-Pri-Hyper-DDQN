"""
Classical tabular RL agents for the movement-based 4-action experiment.
"""

from __future__ import annotations

import pickle
import random
from collections import defaultdict
from typing import Dict, List

import numpy as np

from main_experiment_6action.utils.config import NUM_ACTIONS, PATHS, TABULAR_CONFIG


class TabularAgent:
    def __init__(self):
        cfg = TABULAR_CONFIG
        self.alpha = cfg["alpha"]
        self.gamma = cfg["gamma"]
        self.epsilon = cfg["epsilon_start"]
        self.epsilon_end = cfg["epsilon_end"]
        self.epsilon_decay = cfg["epsilon_decay"]
        self.num_actions = NUM_ACTIONS
        self.Q: Dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(self.num_actions, dtype=np.float64)
        )
        self.episode_rewards: List[float] = []

    def select_action(self, state: tuple) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        return int(np.argmax(self.Q[state]))

    def greedy_action(self, state: tuple) -> int:
        return int(np.argmax(self.Q[state]))

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path):
        payload = {
            "Q": dict(self.Q),
            "epsilon": self.epsilon,
            "episode_rewards": self.episode_rewards,
            "num_actions": self.num_actions,
        }
        with open(path, "wb") as handle:
            pickle.dump(payload, handle)

    def load(self, path):
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        self.Q = defaultdict(lambda: np.zeros(self.num_actions, dtype=np.float64), payload["Q"])
        self.epsilon = payload["epsilon"]
        self.episode_rewards = payload.get("episode_rewards", [])

    def q_table_size(self) -> int:
        return len(self.Q)


class TD0Agent(TabularAgent):
    """
    TD(0) baseline retained for methodology continuity.

    It learns a state-value estimate under the behaviour policy and mirrors the
    state value into each action slot so it can share the same reporting path.
    """

    def __init__(self):
        super().__init__()
        self.V: Dict[tuple, float] = defaultdict(float)

    def update(self, state: tuple, reward: float, next_state: tuple, done: bool) -> float:
        target = reward if done else reward + self.gamma * self.V[next_state]
        td_error = target - self.V[state]
        self.V[state] += self.alpha * td_error
        self.Q[state][:] = self.V[state]
        return float(td_error)

    def save(self, path=PATHS["td0_model"]):
        payload = {
            "Q": dict(self.Q),
            "V": dict(self.V),
            "epsilon": self.epsilon,
            "episode_rewards": self.episode_rewards,
            "num_actions": self.num_actions,
        }
        with open(path, "wb") as handle:
            pickle.dump(payload, handle)


class SARSAAgent(TabularAgent):
    def update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        next_action: int,
        done: bool,
    ) -> float:
        q_next = 0.0 if done else self.Q[next_state][next_action]
        target = reward + self.gamma * q_next
        td_error = target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error
        return float(td_error)

    def save(self, path=PATHS["sarsa_model"]):
        super().save(path)


class QLearningAgent(TabularAgent):
    def update(self, state: tuple, action: int, reward: float, next_state: tuple, done: bool) -> float:
        q_next = 0.0 if done else np.max(self.Q[next_state])
        target = reward + self.gamma * q_next
        td_error = target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error
        return float(td_error)

    def save(self, path=PATHS["qlearning_model"]):
        super().save(path)
