"""
SUMO Traffic Environment (TraCI Interface)
==========================================
Wraps the SUMO simulator with the TraCI Python API, providing the same
Gym-like interface as TrafficEnv so that any trained agent can be
dropped in without code changes.

Prerequisites:
    pip install traci sumolib

Usage:
    env = SumoEnv(use_gui=False)
    state = env.reset()
    next_state, reward, done, info = env.step(action)
    env.close()
"""

import os
import sys
import numpy as np
from typing import Tuple, Dict, Optional

# TraCI import – gracefully handles environments where SUMO is not installed
try:
    import traci
    import traci.constants as tc
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False
    print("[SumoEnv] WARNING: traci not found. "
          "Install SUMO and run: pip install traci sumolib")

from utils.config import ENV_CONFIG, SUMO_CONFIG, PHASES, ACTION_PHASES


class SumoEnv:
    """
    SUMO-backed traffic environment connected via TraCI.

    State vector (dim=8):
        [q_N, q_S, q_E, q_W,   wt_N, wt_S, wt_E, wt_W]
        all values normalised to [0, 1].

    Actions:
        0 → NS_GREEN  (North-South gets green)
        1 → EW_GREEN  (East-West gets green)
    """

    # SUMO phase indices (defined in .net.xml / tls logic)
    PHASE_NS_GREEN  = 0   # GrGr
    PHASE_NS_YELLOW = 1   # yryr
    PHASE_EW_GREEN  = 2   # rGrG
    PHASE_EW_YELLOW = 3   # ryry

    def __init__(self,
                 config_file: str = None,
                 use_gui: bool = False,
                 seed: int = 42,
                 max_steps: int = None):

        if not TRACI_AVAILABLE:
            raise RuntimeError("SUMO / TraCI is not installed.")

        self.config_file  = config_file  or SUMO_CONFIG["config_file"]
        self.use_gui      = use_gui
        self.seed         = seed
        self.max_steps    = max_steps or ENV_CONFIG["simulation_seconds"]
        self.tls_id       = SUMO_CONFIG["tls_id"]
        self.lane_ids     = SUMO_CONFIG["lane_ids"]
        self.step_length  = ENV_CONFIG["step_length"]
        self.phase_dur    = ENV_CONFIG["phase_duration"]
        self.yellow_dur   = ENV_CONFIG["yellow_duration"]
        self.max_queue    = ENV_CONFIG["max_queue"]
        self.max_wait     = ENV_CONFIG["max_wait_time"]

        # Episode counters
        self._step_count      = 0
        self._current_phase   = 0    # action-level phase (0 or 1)
        self._phase_changes   = 0
        self._total_waiting   = 0.0
        self._total_throughput= 0
        self._sumo_started    = False

    # ── SUMO lifecycle ────────────────────────────────────────────

    def _start_sumo(self):
        binary = "sumo-gui" if self.use_gui else SUMO_CONFIG["sumo_binary"]
        cmd = [
            binary,
            "-c", self.config_file,
            "--step-length", str(self.step_length),
            "--seed", str(self.seed),
            "--no-warnings",
            "--no-step-log",
            "--time-to-teleport", "-1",   # disable teleporting
            "--waiting-time-memory", str(self.max_wait),
            "--collision.action", "remove",
        ]
        traci.start(cmd, port=SUMO_CONFIG["port"])
        self._sumo_started = True

    def _stop_sumo(self):
        if self._sumo_started:
            traci.close()
            self._sumo_started = False

    # ── TraCI queries ─────────────────────────────────────────────

    def _get_queue_lengths(self) -> np.ndarray:
        """Query halting vehicle count for each monitored lane."""
        ql = []
        for lane in self.lane_ids:
            try:
                ql.append(traci.lane.getLastStepHaltingNumber(lane))
            except traci.TraCIException:
                ql.append(0)
        return np.array(ql, dtype=np.float32)

    def _get_wait_times(self) -> np.ndarray:
        """Sum of accumulated waiting time for vehicles on each lane."""
        wt = []
        for lane in self.lane_ids:
            try:
                veh_ids = traci.lane.getLastStepVehicleIDs(lane)
                total   = sum(traci.vehicle.getAccumulatedWaitingTime(v)
                              for v in veh_ids)
                wt.append(total)
            except traci.TraCIException:
                wt.append(0.0)
        return np.array(wt, dtype=np.float32)

    def _get_throughput_delta(self) -> int:
        """Vehicles that completed their journey in last step."""
        return traci.simulation.getArrivedNumber()

    # ── Phase control ─────────────────────────────────────────────

    def _apply_phase(self, action: int, prev_action: int):
        """
        Set traffic light phase with yellow transition if phase changes.
        action: 0 = NS_GREEN, 1 = EW_GREEN
        """
        if action == prev_action:
            # Extend current green phase
            new_phase = (self.PHASE_NS_GREEN if action == 0
                         else self.PHASE_EW_GREEN)
        else:
            # Yellow transition first, then switch
            yellow_phase = (self.PHASE_NS_YELLOW if prev_action == 0
                            else self.PHASE_EW_YELLOW)
            traci.trafficlight.setPhase(self.tls_id, yellow_phase)
            for _ in range(int(self.yellow_dur / self.step_length)):
                traci.simulationStep()
                self._step_count += 1
            new_phase = (self.PHASE_NS_GREEN if action == 0
                         else self.PHASE_EW_GREEN)
            self._phase_changes += 1

        traci.trafficlight.setPhase(self.tls_id, new_phase)
        traci.trafficlight.setPhaseDuration(self.tls_id, self.phase_dur)

    # ── Reward ────────────────────────────────────────────────────

    def _compute_reward(self,
                        queues: np.ndarray,
                        wait_times: np.ndarray,
                        throughput: int,
                        phase_changed: bool) -> float:
        """
        Multi-objective reward for SUMO environment:
            R = -w1*queue - w2*wait - w3*switch + w4*throughput - w5*fairness
        """
        w1, w2, w3, w4, w5 = 0.4, 0.3, 0.05, 0.15, 0.1

        queue_norm      = queues.sum()      / (len(self.lane_ids) * self.max_queue)
        wait_norm       = wait_times.sum()  / (len(self.lane_ids) * self.max_wait)
        switch_penalty  = 1.0 if phase_changed else 0.0
        throughput_norm = min(throughput / 10.0, 1.0)
        fairness        = float(np.std(queues)) / (self.max_queue + 1e-8)

        r = (- w1 * queue_norm
             - w2 * wait_norm
             - w3 * switch_penalty
             + w4 * throughput_norm
             - w5 * fairness)
        return float(np.clip(r, -1.0, 1.0))

    # ── State ─────────────────────────────────────────────────────

    def _build_state(self, queues: np.ndarray,
                     wait_times: np.ndarray) -> np.ndarray:
        q_norm  = np.clip(queues     / self.max_queue, 0, 1)
        wt_norm = np.clip(wait_times / self.max_wait,  0, 1)
        return np.concatenate([q_norm, wt_norm]).astype(np.float32)

    # ── Gym interface ─────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Start a new episode; returns initial state vector."""
        self._stop_sumo()
        self._step_count       = 0
        self._current_phase    = 0
        self._phase_changes    = 0
        self._total_waiting    = 0.0
        self._total_throughput = 0
        self._start_sumo()

        # Run a few warm-up steps to populate the network
        for _ in range(10):
            traci.simulationStep()
            self._step_count += 1

        queues     = self._get_queue_lengths()
        wait_times = self._get_wait_times()
        return self._build_state(queues, wait_times)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one agent step (covers `phase_duration` SUMO seconds).

        Returns
        -------
        next_state : np.ndarray
        reward     : float
        done       : bool
        info       : dict
        """
        prev_action   = self._current_phase
        phase_changed = (action != prev_action)
        self._current_phase = action

        # Apply action and run SUMO for one phase duration
        self._apply_phase(action, prev_action)

        steps_in_phase = int(self.phase_dur / self.step_length)
        accumulated_throughput = 0
        for _ in range(steps_in_phase):
            traci.simulationStep()
            self._step_count      += 1
            accumulated_throughput += self._get_throughput_delta()

        self._total_throughput += accumulated_throughput

        # Observe next state
        queues     = self._get_queue_lengths()
        wait_times = self._get_wait_times()
        self._total_waiting += wait_times.sum()

        reward = self._compute_reward(queues, wait_times,
                                      accumulated_throughput, phase_changed)

        done = (self._step_count >= self.max_steps or
                traci.simulation.getMinExpectedNumber() == 0)

        next_state = self._build_state(queues, wait_times)

        info = {
            "step":          self._step_count,
            "queues":        queues.tolist(),
            "wait_times":    wait_times.tolist(),
            "throughput":    accumulated_throughput,
            "phase":         action,
            "phase_changes": self._phase_changes,
        }
        return next_state, reward, done, info

    def close(self):
        self._stop_sumo()

    def get_episode_stats(self) -> Dict:
        steps = max(self._step_count, 1)
        return {
            "avg_queue_length": float(self._total_waiting /
                                      (steps * len(self.lane_ids) + 1e-8)),
            "avg_waiting_time": float(self._total_waiting / steps),
            "total_throughput": self._total_throughput,
            "phase_changes":    self._phase_changes,
            "total_steps":      self._step_count,
        }

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
