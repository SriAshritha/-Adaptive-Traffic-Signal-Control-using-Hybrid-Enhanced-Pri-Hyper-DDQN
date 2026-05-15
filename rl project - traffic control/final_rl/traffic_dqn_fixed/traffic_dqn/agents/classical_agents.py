"""
Classical tabular RL agents for two-phase traffic control.
"""

import pickle
import random
from collections import defaultdict
from typing import Dict, List

import numpy as np

from utils.config import NUM_ACTIONS, PATHS, TABULAR_CONFIG


class TabularAgent:
    """Shared functionality for tabular control agents."""

    def __init__(
        self,
        alpha: float = None,
        gamma: float = None,
        epsilon_start: float = None,
        epsilon_end: float = None,
        epsilon_decay: float = None,
    ):
        cfg = TABULAR_CONFIG
        self.alpha = alpha or cfg["alpha"]
        self.gamma = gamma or cfg["gamma"]
        self.epsilon = epsilon_start or cfg["epsilon_start"]
        self.epsilon_end = epsilon_end or cfg["epsilon_end"]
        self.epsilon_decay = epsilon_decay or cfg["epsilon_decay"]
        self.trace_lambda = cfg.get("trace_lambda", 0.8)
        self.optimistic_init = cfg.get("optimistic_init", 0.0)
        self.heuristic_prior = cfg.get("heuristic_prior", 0.0)
        self.num_actions = NUM_ACTIONS

        self.Q: Dict[tuple, np.ndarray] = defaultdict(self._zero_q)
        self.episode_rewards: List[float] = []
        self.episode_steps: List[int] = []

    def _zero_q(self) -> np.ndarray:
        return np.full(self.num_actions, self.optimistic_init, dtype=np.float64)

    def _heuristic_action_from_state(self, state: tuple) -> int:
        ns_queue, ew_queue = state[0], state[1]
        pressure_bin = state[4]
        phase = state[-1]
        midpoint = TABULAR_CONFIG.get("pressure_bins", 3) // 2

        if ns_queue > ew_queue:
            return 0
        if ew_queue > ns_queue:
            return 1
        if pressure_bin > midpoint:
            return 0
        if pressure_bin < midpoint:
            return 1
        return int(phase)

    def _ensure_state(self, state: tuple):
        if state not in self.Q:
            q_values = self._zero_q()
            q_values[self._heuristic_action_from_state(state)] += self.heuristic_prior
            self.Q[state] = q_values

    def _biased_q_values(self, state: tuple, q_values: np.ndarray = None) -> np.ndarray:
        base = np.array(self.Q[state] if q_values is None else q_values, copy=True)
        base[self._heuristic_action_from_state(state)] += self.heuristic_prior
        return base

    def _greedy_from_values(self, q_values: np.ndarray) -> int:
        best = np.flatnonzero(np.isclose(q_values, q_values.max()))
        return int(random.choice(best.tolist()))

    def select_action(self, state: tuple) -> int:
        self._ensure_state(state)
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        return self._greedy_from_values(self._biased_q_values(state))

    def greedy_action(self, state: tuple) -> int:
        self._ensure_state(state)
        return self._greedy_from_values(self._biased_q_values(state))

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        payload = {
            "Q": dict(self.Q),
            "epsilon": self.epsilon,
            "episode_rewards": self.episode_rewards,
            "episode_steps": self.episode_steps,
            "config": {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay": self.epsilon_decay,
                "num_actions": self.num_actions,
                "trace_lambda": self.trace_lambda,
                "optimistic_init": self.optimistic_init,
            },
        }
        with open(path, "wb") as handle:
            pickle.dump(payload, handle)
        print(f"[{self.__class__.__name__}] Saved to {path}")

    def load(self, path: str):
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        self.Q = defaultdict(self._zero_q, payload["Q"])
        self.epsilon = payload["epsilon"]
        self.episode_rewards = payload.get("episode_rewards", [])
        self.episode_steps = payload.get("episode_steps", [])
        print(f"[{self.__class__.__name__}] Loaded from {path} (Q-entries={len(self.Q)})")

    def q_table_size(self) -> int:
        return len(self.Q)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"alpha={self.alpha}, gamma={self.gamma}, "
            f"epsilon={self.epsilon:.4f}, Q-size={self.q_table_size()})"
        )


class TD0Agent(TabularAgent):
    """One-step TD control via Expected SARSA."""

    def update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        done: bool,
    ) -> float:
        if done:
            expected_next = 0.0
        else:
            self._ensure_state(next_state)
            q_next = self._biased_q_values(next_state)
            greedy_mask = np.isclose(q_next, q_next.max())
            policy = np.full(self.num_actions, self.epsilon / self.num_actions)
            policy[greedy_mask] += (1.0 - self.epsilon) / greedy_mask.sum()
            expected_next = float(np.dot(policy, q_next))

        td_target = reward + self.gamma * expected_next
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error
        return float(td_error)

    def save(self, path: str = None):
        super().save(path or PATHS["td0_model"])

    def load(self, path: str = None):
        super().load(path or PATHS["td0_model"])


class SARSAAgent(TabularAgent):
    """SARSA(lambda) with accumulating eligibility traces."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset_traces()

    def reset_traces(self):
        self.eligibility: Dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(self.num_actions, dtype=np.float64)
        )

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
        td_target = reward + self.gamma * q_next
        td_error = td_target - self.Q[state][action]

        self.eligibility[state][action] += 1.0
        trace_decay = self.gamma * self.trace_lambda
        for key in list(self.eligibility.keys()):
            self.Q[key] += self.alpha * td_error * self.eligibility[key]
            self.eligibility[key] *= trace_decay
            if np.max(np.abs(self.eligibility[key])) < 1e-8:
                del self.eligibility[key]
        return float(td_error)

    def save(self, path: str = None):
        super().save(path or PATHS["sarsa_model"])

    def load(self, path: str = None):
        super().load(path or PATHS["sarsa_model"])


class QLearningAgent(TabularAgent):
    """Double Q-learning for lower overestimation bias."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.QA: Dict[tuple, np.ndarray] = defaultdict(self._zero_q)
        self.QB: Dict[tuple, np.ndarray] = defaultdict(self._zero_q)

    def _combined_q(self, state: tuple) -> np.ndarray:
        return self.QA[state] + self.QB[state]

    def select_action(self, state: tuple) -> int:
        self._ensure_state(state)
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        return self._greedy_from_values(self._biased_q_values(state, self._combined_q(state)))

    def greedy_action(self, state: tuple) -> int:
        self._ensure_state(state)
        return self._greedy_from_values(self._biased_q_values(state, self._combined_q(state)))

    def update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        done: bool,
    ) -> float:
        online, target = (self.QA, self.QB) if random.random() < 0.5 else (self.QB, self.QA)
        if done:
            q_next = 0.0
        else:
            next_action = self._greedy_from_values(online[next_state])
            q_next = target[next_state][next_action]

        td_target = reward + self.gamma * q_next
        td_error = td_target - online[state][action]
        online[state][action] += self.alpha * td_error
        self.Q[state] = 0.5 * (self.QA[state] + self.QB[state])
        return float(td_error)

    def q_table_size(self) -> int:
        return len(set(self.QA.keys()) | set(self.QB.keys()))

    def save(self, path: str = None):
        save_path = path or PATHS["qlearning_model"]
        payload = {
            "QA": dict(self.QA),
            "QB": dict(self.QB),
            "epsilon": self.epsilon,
            "episode_rewards": self.episode_rewards,
            "episode_steps": self.episode_steps,
            "config": {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay": self.epsilon_decay,
                "num_actions": self.num_actions,
                "trace_lambda": self.trace_lambda,
                "optimistic_init": self.optimistic_init,
            },
        }
        with open(save_path, "wb") as handle:
            pickle.dump(payload, handle)
        print(f"[{self.__class__.__name__}] Saved to {save_path}")

    def load(self, path: str = None):
        load_path = path or PATHS["qlearning_model"]
        with open(load_path, "rb") as handle:
            payload = pickle.load(handle)

        if "QA" in payload and "QB" in payload:
            self.QA = defaultdict(self._zero_q, payload["QA"])
            self.QB = defaultdict(self._zero_q, payload["QB"])
        else:
            legacy_q = payload["Q"]
            self.QA = defaultdict(self._zero_q, legacy_q)
            self.QB = defaultdict(
                self._zero_q,
                {key: np.array(value, dtype=np.float64) for key, value in legacy_q.items()},
            )
        self.epsilon = payload["epsilon"]
        self.episode_rewards = payload.get("episode_rewards", [])
        self.episode_steps = payload.get("episode_steps", [])
        print(f"[{self.__class__.__name__}] Loaded from {load_path} (Q-entries={self.q_table_size()})")
