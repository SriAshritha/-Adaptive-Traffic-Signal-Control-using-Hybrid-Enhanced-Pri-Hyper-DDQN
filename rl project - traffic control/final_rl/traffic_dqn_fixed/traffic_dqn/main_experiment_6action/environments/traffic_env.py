"""
Movement-based traffic environment for the main 4-action experiment.

The environment models eight queue groups:
- north/south/east/west straight+right movements
- north/south/east/west left+U-turn movements

Each RL action selects one protected movement stage. Yellow loss is applied
whenever the stage changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from main_experiment_6action.utils.config import (
    ACTION_DEFINITIONS,
    ENV_CONFIG,
    MOVEMENT_KEYS,
    TABULAR_CONFIG,
)


@dataclass(frozen=True)
class ActionSpec:
    action_id: int
    name: str
    served_groups: Tuple[str, str]


class TrafficEnv:
    def __init__(self, seed: int = 42, mode: str = "normal"):
        if mode not in ENV_CONFIG["movement_rates"]:
            raise ValueError(f"Unsupported mode: {mode}")

        self.seed = seed
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.max_queue = ENV_CONFIG["max_queue"]
        self.max_wait_time = ENV_CONFIG["max_wait_time"]
        self.decision_interval = ENV_CONFIG["decision_interval"]
        self.yellow_duration = ENV_CONFIG["yellow_duration"]
        self.simulation_seconds = ENV_CONFIG["simulation_seconds"]
        self.max_decisions = self.simulation_seconds // self.decision_interval
        self.arrival_rates = ENV_CONFIG["movement_rates"][mode]
        self.service_rates = ENV_CONFIG["service_rates"]

        self.action_specs = {
            action: ActionSpec(
                action_id=action,
                name=spec["name"],
                served_groups=spec["served_groups"],
            )
            for action, spec in ACTION_DEFINITIONS.items()
        }

        self.queues = np.zeros(len(MOVEMENT_KEYS), dtype=np.float32)
        self.wait_times = np.zeros(len(MOVEMENT_KEYS), dtype=np.float32)
        self.current_action = 0
        self.step_count = 0

        self.total_reward = 0.0
        self.total_waiting = 0.0
        self.cumulative_queue = 0.0
        self.total_delay = 0.0
        self.total_throughput = 0
        self.phase_changes = 0

    def reset(self) -> Tuple[np.ndarray, tuple]:
        self.queues.fill(0.0)
        self.wait_times.fill(0.0)
        self.current_action = 0
        self.step_count = 0
        self.total_reward = 0.0
        self.total_waiting = 0.0
        self.cumulative_queue = 0.0
        self.total_delay = 0.0
        self.total_throughput = 0
        self.phase_changes = 0
        return self._get_raw_state(), self._discretise_state()

    def _effective_green(self, phase_changed: bool) -> int:
        if not phase_changed:
            return self.decision_interval
        return max(self.decision_interval - self.yellow_duration, 1)

    def _arrivals(self) -> np.ndarray:
        arrivals = []
        for key in MOVEMENT_KEYS:
            lam = self.arrival_rates[key] * self.decision_interval
            arrivals.append(self.rng.poisson(lam))
        return np.array(arrivals, dtype=np.float32)

    def _served_indices(self, action: int) -> Tuple[int, int]:
        served = self.action_specs[action].served_groups
        return tuple(MOVEMENT_KEYS.index(group) for group in served)

    def _departures(self, action: int, green_time: int) -> int:
        served_indices = self._served_indices(action)
        total_departed = 0
        for idx in served_indices:
            movement_key = MOVEMENT_KEYS[idx]
            service_family = "turn" if movement_key.endswith("turn") else "through"
            rate = self.service_rates[service_family] * green_time
            departed = int(min(self.queues[idx], self.rng.poisson(rate)))
            self.queues[idx] = max(0.0, self.queues[idx] - departed)
            if departed > 0:
                retained_ratio = self.queues[idx] / (self.queues[idx] + departed)
                self.wait_times[idx] *= retained_ratio
            total_departed += departed
        return total_departed

    def _update_wait_times(self, action: int) -> None:
        served_indices = set(self._served_indices(action))
        for idx in range(len(MOVEMENT_KEYS)):
            if idx not in served_indices:
                self.wait_times[idx] += self.queues[idx] * self.decision_interval

    def _compute_reward(self, phase_changed: bool, throughput: int) -> float:
        queue_pressure = self.queues.sum() / (len(MOVEMENT_KEYS) * self.max_queue)
        wait_penalty = self.wait_times.sum() / (
            len(MOVEMENT_KEYS) * self.max_wait_time
        )
        fairness_penalty = float(np.std(self.queues)) / (self.max_queue + 1e-8)
        throughput_bonus = throughput / (2.0 * self.max_queue)
        switch_penalty = 1.0 if phase_changed else 0.0

        reward = (
            -0.45 * queue_pressure
            -0.30 * wait_penalty
            +0.20 * throughput_bonus
            -0.05 * fairness_penalty
            -0.05 * switch_penalty
        )
        return float(np.clip(reward, -1.0, 1.0))

    def _get_raw_state(self) -> np.ndarray:
        q_norm = self.queues / self.max_queue
        w_norm = np.clip(self.wait_times / self.max_wait_time, 0.0, 1.0)
        current_phase = np.array([self.current_action / 3.0], dtype=np.float32)
        return np.concatenate([q_norm, w_norm, current_phase]).astype(np.float32)

    def _discretise_state(self) -> tuple:
        bins = TABULAR_CONFIG["queue_bins"]
        q_disc = np.minimum((self.queues / self.max_queue * bins).astype(int), bins - 1)
        return tuple(int(x) for x in q_disc) + (self.current_action,)

    def step(self, action: int):
        phase_changed = action != self.current_action
        if phase_changed:
            self.phase_changes += 1
        self.current_action = action

        self.queues = np.minimum(self.queues + self._arrivals(), self.max_queue)
        green_time = self._effective_green(phase_changed)
        throughput = self._departures(action, green_time)
        self._update_wait_times(action)

        reward = self._compute_reward(phase_changed, throughput)
        non_zero_mask = self.queues > 0
        if np.any(non_zero_mask):
            current_wait = float(
                np.mean(self.wait_times[non_zero_mask] / np.maximum(self.queues[non_zero_mask], 1.0))
            )
        else:
            current_wait = 0.0
        self.step_count += 1
        self.total_reward += reward
        self.total_waiting += current_wait
        self.cumulative_queue += float(self.queues.mean())
        self.total_delay += float(self.queues.sum() * self.decision_interval)
        self.total_throughput += throughput

        done = self.step_count >= self.max_decisions
        info = {
            "step": self.step_count,
            "action_name": self.action_specs[action].name,
            "queues": self.queues.copy(),
            "wait_times": self.wait_times.copy(),
            "throughput": throughput,
            "phase_changes": self.phase_changes,
        }
        return self._get_raw_state(), self._discretise_state(), reward, done, info

    def get_episode_stats(self) -> Dict[str, float]:
        decisions = max(self.step_count, 1)
        return {
            "avg_queue_length": float(self.cumulative_queue / decisions),
            "avg_waiting_time": float(self.total_waiting / decisions),
            "total_delay": float(self.total_delay),
            "total_throughput": int(self.total_throughput),
            "phase_changes": int(self.phase_changes),
            "total_steps": int(self.step_count),
        }
