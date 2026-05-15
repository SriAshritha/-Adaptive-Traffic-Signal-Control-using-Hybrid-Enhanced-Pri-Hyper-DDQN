"""
Traffic Environment (Simulated)
================================
A lightweight traffic intersection simulator that works WITHOUT SUMO.
Used for rapid prototyping of tabular RL agents (TD0, SARSA, Q-Learning).

State Space:
    - Queue length per lane  (N, S, E, W) – discretised into bins
    - Current signal phase

Action Space:
    - 0: Set phase NS_GREEN (North-South gets green)
    - 1: Set phase EW_GREEN (East-West gets green)

Reward:
    - Negative of total vehicles waiting at each step
    - Bonus for reducing cumulative waiting time
    - Penalty for unnecessary phase switching
"""

import numpy as np
import random
from typing import Tuple, Dict
from utils.config import ENV_CONFIG, TABULAR_CONFIG


class TrafficEnv:
    """
    Lightweight stochastic traffic intersection for tabular RL experiments.
    """

    # Phase constants
    NS_GREEN = 0
    EW_GREEN = 1

    def __init__(self, seed: int = 42,
                 arrival_rates: Dict[str, float] = None,
                 mode: str = "normal"):
        """
        Parameters
        ----------
        seed          : random seed for reproducibility
        arrival_rates : vehicle arrivals per step per lane
                        keys: 'north', 'south', 'east', 'west'
        mode          : 'normal', 'peak', or 'asymmetric' traffic demand
        """
        self.seed = seed
        self.rng  = np.random.default_rng(seed)
        self.mode = mode

        # arrival rates (vehicles/step) per lane
        self.arrival_rates = arrival_rates or self._default_rates(mode)

        # Environment parameters
        self.max_queue        = ENV_CONFIG["max_queue"]
        self.phase_duration   = ENV_CONFIG["phase_duration"]
        self.yellow_duration  = ENV_CONFIG["yellow_duration"]

        # State variables
        self.queues: np.ndarray = np.zeros(4, dtype=np.float32)  # N,S,E,W
        self.wait_times: np.ndarray = np.zeros(4, dtype=np.float32)
        self.current_phase: int = self.NS_GREEN
        self.phase_timer: int   = 0
        self.step_count: int    = 0

        # Tracking metrics
        self.total_waiting: float   = 0.0
        self.total_throughput: int  = 0
        self.phase_changes: int     = 0
        self.episode_rewards: list  = []

    # ── environment internals ─────────────────────────────────────

    def _default_rates(self, mode: str) -> Dict[str, float]:
        if mode == "peak":
            return {"north": 0.6, "south": 0.6, "east": 0.8, "west": 0.8}
        elif mode == "asymmetric":
            return {"north": 0.3, "south": 0.7, "east": 0.6, "west": 0.2}
        else:  # normal
            return {"north": 0.4, "south": 0.4, "east": 0.4, "west": 0.4}

    def _arrive(self):
        """Stochastic vehicle arrivals following Poisson process."""
        rates = list(self.arrival_rates.values())
        arrivals = self.rng.poisson(rates)
        self.queues = np.minimum(self.queues + arrivals, self.max_queue)

    def _depart(self, action: int):
        """
        Vehicles depart based on current green phase.
        Green lanes discharge at ~0.5 vehicles/step (saturation flow).
        """
        depart_rate = 0.5
        if action == self.NS_GREEN:
            green_lanes = [0, 1]   # North, South indices
        else:
            green_lanes = [2, 3]   # East, West indices

        departed = 0
        for lane in green_lanes:
            leaving = int(min(self.queues[lane],
                              self.rng.poisson(depart_rate) + 1))
            self.queues[lane] = max(0, self.queues[lane] - leaving)
            departed += leaving
        self.total_throughput += departed

    def _update_wait_times(self, action: int):
        """
        Red-phase vehicles accumulate waiting time;
        green-phase vehicles have wait reset as they depart.
        """
        if action == self.NS_GREEN:
            red_lanes   = [2, 3]
            green_lanes = [0, 1]
        else:
            red_lanes   = [0, 1]
            green_lanes = [2, 3]

        for l in red_lanes:
            self.wait_times[l] += self.queues[l]   # waiting vehicles × 1 step
        for l in green_lanes:
            self.wait_times[l] *= 0.5              # partial reset for departed

    # ── reward ───────────────────────────────────────────────────

    def _compute_reward(self, prev_queues: np.ndarray,
                        action: int, phase_changed: bool) -> float:
        """
        Multi-objective reward:
          R = -w1*queue_pressure - w2*wait_penalty
              + w3*throughput_bonus - w4*switch_penalty - w5*fairness_penalty
        """
        w1, w2, w3, w4, w5 = 0.5, 0.3, 0.1, 0.05, 0.05

        # 1. Queue pressure (normalized)
        queue_pressure = self.queues.sum() / (4 * self.max_queue)

        # 2. Waiting time penalty (normalize by max_wait = 300s)
        wait_penalty   = self.wait_times.sum() / (4 * ENV_CONFIG["max_wait_time"])

        # 3. Throughput bonus (vehicles cleared)
        throughput     = max(0, prev_queues.sum() - self.queues.sum())
        throughput_bonus = throughput / (4 * self.max_queue)

        # 4. Phase switch penalty (discourage rapid switching)
        switch_penalty = 0.1 if phase_changed else 0.0

        # 5. Fairness penalty (penalise large queue imbalance)
        fairness       = np.std(self.queues) / (self.max_queue + 1e-8)

        reward = (- w1 * queue_pressure
                  - w2 * wait_penalty
                  + w3 * throughput_bonus
                  - w4 * switch_penalty
                  - w5 * fairness)
        return float(np.clip(reward, -1.0, 1.0))

    # ── state representation ──────────────────────────────────────

    def _get_raw_state(self) -> np.ndarray:
        """
        Returns continuous state vector:
        [q_N, q_S, q_E, q_W,  wt_N, wt_S, wt_E, wt_W,  current_phase]
        All values normalised to [0, 1].
        """
        q_norm  = self.queues / self.max_queue
        wt_norm = np.clip(self.wait_times / ENV_CONFIG["max_wait_time"], 0, 1)
        phase   = np.array([self.current_phase], dtype=np.float32)
        return np.concatenate([q_norm, wt_norm, phase]).astype(np.float32)

    def _discretise_state(self) -> tuple:
        """
        Discretise continuous state for tabular RL.
        Returns a hashable tuple suitable as Q-table key.
        """
        queue_bins = TABULAR_CONFIG["queue_bins"]
        wait_bins = TABULAR_CONFIG["wait_bins"]
        pressure_bins = TABULAR_CONFIG["pressure_bins"]
        phase_age_bins = TABULAR_CONFIG.get("phase_age_bins", 4)

        ns_queue = float(self.queues[0] + self.queues[1])
        ew_queue = float(self.queues[2] + self.queues[3])
        ns_wait = float(self.wait_times[0] + self.wait_times[1])
        ew_wait = float(self.wait_times[2] + self.wait_times[3])

        queue_scale = max(2 * self.max_queue, 1)
        wait_scale = max(2 * ENV_CONFIG["max_wait_time"], 1)
        max_phase_age = max(ENV_CONFIG["phase_duration"] + ENV_CONFIG["yellow_duration"], 1)

        # Keep the axis queue totals effectively exact in the tabular state.
        ns_queue_bin = min(int(round(ns_queue)), queue_bins - 1)
        ew_queue_bin = min(int(round(ew_queue)), queue_bins - 1)
        ns_wait_bin = min(int(ns_wait / wait_scale * wait_bins), wait_bins - 1)
        ew_wait_bin = min(int(ew_wait / wait_scale * wait_bins), wait_bins - 1)

        pressure = (ns_queue + 0.1 * ns_wait) - (ew_queue + 0.1 * ew_wait)
        normalized_pressure = np.clip((pressure / queue_scale + 1.0) / 2.0, 0.0, 0.999999)
        pressure_bin = min(int(normalized_pressure * pressure_bins), pressure_bins - 1)
        phase_age_bin = min(int(self.phase_timer / max_phase_age * phase_age_bins), phase_age_bins - 1)

        return (
            ns_queue_bin,
            ew_queue_bin,
            ns_wait_bin,
            ew_wait_bin,
            pressure_bin,
            phase_age_bin,
            self.current_phase,
        )

    # ── Gym-like interface ────────────────────────────────────────

    def reset(self) -> Tuple[np.ndarray, tuple]:
        """Reset environment; returns (continuous_state, discrete_state)."""
        self.queues       = np.zeros(4, dtype=np.float32)
        self.wait_times   = np.zeros(4, dtype=np.float32)
        self.current_phase = self.NS_GREEN
        self.phase_timer   = 0
        self.step_count    = 0
        self.total_waiting = 0.0
        self.total_throughput = 0
        self.phase_changes = 0
        return self._get_raw_state(), self._discretise_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one time step.

        Parameters
        ----------
        action : 0 (NS_GREEN) or 1 (EW_GREEN)

        Returns
        -------
        next_raw_state   : np.ndarray  – for DQN
        next_disc_state  : tuple       – for tabular agents
        reward           : float
        done             : bool
        info             : dict
        """
        prev_queues   = self.queues.copy()
        phase_changed = (action != self.current_phase)

        if phase_changed:
            self.phase_changes += 1
            self.current_phase  = action
            self.phase_timer = 0
        else:
            self.phase_timer += 1

        self._arrive()
        self._depart(action)
        self._update_wait_times(action)

        reward = self._compute_reward(prev_queues, action, phase_changed)
        self.total_waiting += self.wait_times.sum()
        self.step_count    += 1

        max_steps = ENV_CONFIG["simulation_seconds"]
        done = (self.step_count >= max_steps)

        info = {
            "step":         self.step_count,
            "queues":       self.queues.copy(),
            "wait_times":   self.wait_times.copy(),
            "throughput":   self.total_throughput,
            "phase":        self.current_phase,
            "phase_changes":self.phase_changes,
        }

        return (self._get_raw_state(),
                self._discretise_state(),
                reward, done, info)

    # ── metrics ───────────────────────────────────────────────────

    def get_episode_stats(self) -> Dict:
        steps = max(self.step_count, 1)
        return {
            "avg_queue_length": float(self.queues.mean()),
            "avg_waiting_time": float(self.total_waiting / steps),
            "total_throughput": self.total_throughput,
            "phase_changes":    self.phase_changes,
            "total_steps":      self.step_count,
        }
