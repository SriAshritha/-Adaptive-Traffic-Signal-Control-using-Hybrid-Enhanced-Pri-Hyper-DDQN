"""
Deploy Trained DQN Agent in SUMO
==================================
Loads a trained DQN (or any classical RL agent) from a .pkl file and
runs a closed-loop traffic signal control experiment inside the SUMO
simulator via TraCI.

Workflow
--------
1. Load trained agent (.pkl) 
2. Start SUMO simulation via TraCI
3. At each decision step:
   a. Query SUMO for queue lengths and waiting times
   b. Build state vector and feed into DQN
   c. Agent selects traffic phase action
   d. Send phase command back to SUMO via TraCI
   e. Run SUMO for `phase_duration` seconds
   f. Collect reward and metrics
4. Save episode statistics and compare vs fixed-time baseline

Usage:
    python deploy_sumo.py --agent dqn --model models/dqn_agent.pkl
    python deploy_sumo.py --agent qlearning --model models/qlearning_qtable.pkl
    python deploy_sumo.py --agent fixed    (fixed-time baseline)
    python deploy_sumo.py --gui            (open SUMO GUI)
"""

import os
import sys
import argparse
import pickle
import numpy as np
from typing import Optional

# ── Graceful TraCI import ──────────────────────────────────────
try:
    import traci
    TRACI_OK = True
except ImportError:
    TRACI_OK = False
    print("[deploy_sumo] WARNING: traci not available. "
          "Install SUMO and `pip install traci sumolib`.")

from utils.config import ENV_CONFIG, SUMO_CONFIG, PATHS, DQN_CONFIG


# ─────────────────────────────────────────────────────────────────
# Agent loaders
# ─────────────────────────────────────────────────────────────────

def load_agent(agent_type: str, model_path: str):
    """Load the appropriate agent from a .pkl file."""
    if agent_type == "dqn":
        from agents.dqn_agent import DQNAgent
        agent = DQNAgent.load(model_path)
        agent.epsilon = 0.0   # pure greedy for deployment
        print(f"[Loader] DQN agent loaded | {agent}")
        return agent

    elif agent_type in ("qlearning", "sarsa", "td0"):
        from agents.classical_agents import (QLearningAgent,
                                             SARSAAgent, TD0Agent)
        mapping = {"qlearning": QLearningAgent,
                   "sarsa":     SARSAAgent,
                   "td0":       TD0Agent}
        agent = mapping[agent_type]()
        agent.load(model_path)
        agent.epsilon = 0.0   # greedy
        print(f"[Loader] {agent_type} agent loaded | {agent}")
        return agent

    elif agent_type == "fixed":
        print("[Loader] Fixed-time baseline (no model needed)")
        return None

    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


# ─────────────────────────────────────────────────────────────────
# Fixed-time baseline controller
# ─────────────────────────────────────────────────────────────────

class FixedTimeController:
    """
    Fixed-time traffic signal controller.
    Alternates NS/EW green every `cycle_length` seconds.
    """
    def __init__(self, cycle_length: int = 30):
        self.cycle   = cycle_length
        self.elapsed = 0

    def select_action(self, state) -> int:
        """Returns 0 (NS) for first half of cycle, 1 (EW) for second half."""
        phase = (self.elapsed // self.cycle) % 2
        self.elapsed += ENV_CONFIG["phase_duration"]
        return int(phase)

    def greedy_action(self, state) -> int:
        return self.select_action(state)


# ─────────────────────────────────────────────────────────────────
# TraCI interface helpers
# ─────────────────────────────────────────────────────────────────

def get_state(lane_ids: list, max_queue: float,
              max_wait: float, current_action: int = 0) -> np.ndarray:
    """Build normalised state vector from SUMO lane data.
    Returns dim=9: [q_N, q_S, q_E, q_W, wt_N, wt_S, wt_E, wt_W, phase]
    Matches TrafficEnv._get_raw_state() used during DQN training.
    """
    queues     = []
    wait_times = []
    for lane in lane_ids:
        try:
            q  = traci.lane.getLastStepHaltingNumber(lane)
            vs = traci.lane.getLastStepVehicleIDs(lane)
            wt = sum(traci.vehicle.getAccumulatedWaitingTime(v) for v in vs)
        except traci.TraCIException:
            q, wt = 0, 0.0
        queues.append(q)
        wait_times.append(wt)

    q_norm  = np.clip(np.array(queues,     dtype=np.float32) / max_queue, 0, 1)
    wt_norm = np.clip(np.array(wait_times, dtype=np.float32) / max_wait,  0, 1)
    phase   = np.array([float(current_action)], dtype=np.float32)
    return np.concatenate([q_norm, wt_norm, phase])


def apply_phase(tls_id: str, action: int, prev_action: int,
                yellow_dur: int, phase_dur: int, step_len: float):
    """
    Apply traffic signal phase with yellow transition if phase changes.
      action 0 → NS Green  (SUMO phase index 0)
      action 1 → EW Green  (SUMO phase index 2)
    """
    PHASE_MAP   = {0: 0, 1: 2}   # action → SUMO green phase index
    YELLOW_MAP  = {0: 1, 1: 3}   # action → SUMO yellow phase index

    if action != prev_action:
        # Apply yellow for outgoing phase
        traci.trafficlight.setPhase(tls_id, YELLOW_MAP[prev_action])
        steps = int(yellow_dur / step_len)
        for _ in range(steps):
            traci.simulationStep()

    traci.trafficlight.setPhase(tls_id, PHASE_MAP[action])
    traci.trafficlight.setPhaseDuration(tls_id, phase_dur)


def compute_reward(queues: np.ndarray, wait_times: np.ndarray,
                   throughput: int, phase_changed: bool,
                   max_queue: float, max_wait: float) -> float:
    """Same multi-objective reward as SumoEnv."""
    w1, w2, w3, w4, w5 = 0.4, 0.3, 0.05, 0.15, 0.1
    n = len(queues)
    q_norm   = queues.sum()      / (n * max_queue + 1e-8)
    wt_norm  = wait_times.sum()  / (n * max_wait  + 1e-8)
    switch   = 1.0 if phase_changed else 0.0
    tp_norm  = min(throughput / 10.0, 1.0)
    fairness = float(np.std(queues)) / (max_queue + 1e-8)
    r = -w1*q_norm - w2*wt_norm - w3*switch + w4*tp_norm - w5*fairness
    return float(np.clip(r, -1.0, 1.0))


# ─────────────────────────────────────────────────────────────────
# Main deployment loop
# ─────────────────────────────────────────────────────────────────

def run_deployment(agent,
                   agent_type:   str   = "dqn",
                   use_gui:      bool  = False,
                   config_file:  str   = None,
                   max_steps:    int   = 3600,
                   seed:         int   = 42,
                   save_results: bool  = True) -> dict:

    if not TRACI_OK:
        print("[deploy_sumo] Cannot run: TraCI not available.")
        return {}

    cfg          = SUMO_CONFIG
    lane_ids     = cfg["lane_ids"]
    tls_id       = cfg["tls_id"]
    step_len     = ENV_CONFIG["step_length"]
    phase_dur    = ENV_CONFIG["phase_duration"]
    yellow_dur   = ENV_CONFIG["yellow_duration"]
    max_queue    = ENV_CONFIG["max_queue"]
    max_wait     = ENV_CONFIG["max_wait_time"]
    config_file  = config_file or cfg["config_file"]

    # ── Start SUMO ──────────────────────────────────────────────
    binary = "sumo-gui" if use_gui else "sumo"
    cmd = [
        binary,
        "-c", config_file,
        "--step-length",  str(step_len),
        "--seed",         str(seed),
        "--no-warnings",
        "--no-step-log",
        "--time-to-teleport", "-1",
        "--waiting-time-memory", str(int(max_wait)),
    ]
    traci.start(cmd, port=cfg["port"])
    print(f"\n[SUMO] Started | Agent: {agent_type} | GUI: {use_gui}")

    # Warm-up
    for _ in range(10):
        traci.simulationStep()

    # ── Episode loop ────────────────────────────────────────────
    step_count     = 0
    total_reward   = 0.0
    total_throughput = 0
    total_wait     = 0.0
    total_queue    = 0.0
    phase_changes  = 0
    current_action = 0   # start with NS Green

    step_log = []   # detailed step-by-step log

    print(f"[SUMO] Running {max_steps} steps…")

    while step_count < max_steps:
        # 1. Observe state
        state = get_state(lane_ids, max_queue, max_wait, current_action)

        # 2. Select action
        if agent_type == "fixed":
            action = agent.select_action(state)
        elif agent_type == "dqn":
            action = agent._greedy_action(state)
        else:
            # tabular agents use discrete state – approximate
            discrete = tuple((state[:4] * 4).astype(int).clip(0, 4)) + (current_action,)
            action = agent.greedy_action(discrete)

        phase_changed = (action != current_action)
        if phase_changed:
            phase_changes += 1

        # 3. Apply phase to SUMO
        apply_phase(tls_id, action, current_action,
                    yellow_dur, phase_dur, step_len)
        current_action = action

        # 4. Run SUMO for one phase duration
        arrived_this_phase = 0
        steps_in_phase = int(phase_dur / step_len)
        for _ in range(steps_in_phase):
            traci.simulationStep()
            arrived_this_phase += traci.simulation.getArrivedNumber()
            step_count += 1
            if step_count >= max_steps:
                break

        # 5. Observe result
        next_state = get_state(lane_ids, max_queue, max_wait, current_action)
        queues     = next_state[:4] * max_queue
        waits      = next_state[4:] * max_wait

        total_throughput += arrived_this_phase
        total_wait       += waits.sum()
        total_queue      += queues.mean()

        # 6. Compute reward
        reward = compute_reward(queues, waits, arrived_this_phase,
                                phase_changed, max_queue, max_wait)
        total_reward += reward

        # 7. Log
        decision_step = step_count // steps_in_phase
        if decision_step % 10 == 0:
            print(f"  t={step_count:5d}s | phase={'NS' if action==0 else 'EW'} | "
                  f"queues={queues.round(1)} | "
                  f"wait={waits.sum():.0f}s | "
                  f"throughput={arrived_this_phase} | "
                  f"reward={reward:.3f}")

        step_log.append({
            "step":       step_count,
            "action":     action,
            "queues":     queues.tolist(),
            "waits":      waits.tolist(),
            "throughput": arrived_this_phase,
            "reward":     reward,
        })

        # Check if simulation ended
        if traci.simulation.getMinExpectedNumber() == 0:
            print("[SUMO] All vehicles completed trips.")
            break

    # ── Wrap up ─────────────────────────────────────────────────
    traci.close()

    denom = max(step_count // steps_in_phase, 1)
    results = {
        "agent_type":        agent_type,
        "total_steps":       step_count,
        "total_reward":      total_reward,
        "avg_reward":        total_reward / denom,
        "total_throughput":  total_throughput,
        "avg_queue_length":  total_queue  / denom,
        "avg_waiting_time":  total_wait   / (denom * len(lane_ids)),
        "phase_changes":     phase_changes,
        "step_log":          step_log,
    }

    # ── Print summary ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DEPLOYMENT RESULTS  –  Agent: {agent_type}")
    print(f"{'='*60}")
    for k, v in results.items():
        if k != "step_log":
            print(f"  {k:<25}: {v:.4f}" if isinstance(v, float)
                  else f"  {k:<25}: {v}")
    print(f"{'='*60}")

    if save_results:
        os.makedirs("outputs", exist_ok=True)
        out_path = f"outputs/deploy_{agent_type}_results.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(results, f)
        print(f"\n  Results saved → {out_path}")

    return results


# ─────────────────────────────────────────────────────────────────
# Comparative evaluation: DQN vs Fixed-time vs Q-Learning
# ─────────────────────────────────────────────────────────────────

def compare_controllers(use_gui: bool = False):
    """Run all controllers sequentially and print a comparison table."""
    configs = [
        ("fixed",     None,                       FixedTimeController()),
        ("qlearning", PATHS["qlearning_model"],   None),
        ("dqn",       PATHS["dqn_model"],         None),
    ]

    all_results = {}
    for agent_type, model_path, override_agent in configs:
        if override_agent:
            agent = override_agent
        else:
            if not os.path.exists(model_path):
                print(f"[Skipping] {agent_type}: model not found at {model_path}")
                continue
            agent = load_agent(agent_type, model_path)

        results = run_deployment(agent, agent_type=agent_type,
                                 use_gui=use_gui)
        all_results[agent_type] = results

    # Comparison table
    print(f"\n{'='*75}")
    print(f"  CONTROLLER COMPARISON")
    print(f"{'='*75}")
    print(f"  {'Controller':<14} {'Avg Reward':>12} {'Avg Queue':>11} "
          f"{'Avg Wait(s)':>12} {'Throughput':>12}")
    print(f"  {'-'*71}")
    for name, res in all_results.items():
        print(f"  {name:<14} "
              f"{res['avg_reward']:>12.4f} "
              f"{res['avg_queue_length']:>11.2f} "
              f"{res['avg_waiting_time']:>12.1f} "
              f"{res['total_throughput']:>12}")
    print(f"{'='*75}")

    # Save comparison
    with open("outputs/comparison_results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    print("  Comparison saved → outputs/comparison_results.pkl")


# ─────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Deploy trained RL agent in SUMO traffic simulation")
    parser.add_argument("--agent",  choices=["dqn","qlearning","sarsa",
                                              "td0","fixed","compare"],
                        default="dqn")
    parser.add_argument("--model",  type=str, default=None,
                        help="Path to .pkl model file")
    parser.add_argument("--config", type=str,
                        default=SUMO_CONFIG["config_file"],
                        help="Path to SUMO .sumocfg file")
    parser.add_argument("--steps",  type=int, default=3600)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--gui",    action="store_true",
                        help="Open SUMO GUI")
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)

    if args.agent == "compare":
        compare_controllers(use_gui=args.gui)
        return

    # Resolve model path
    if args.model is None:
        default_paths = {
            "dqn":       PATHS["dqn_model"],
            "qlearning": PATHS["qlearning_model"],
            "sarsa":     PATHS["sarsa_model"],
            "td0":       PATHS["td0_model"],
            "fixed":     None,
        }
        args.model = default_paths[args.agent]

    if args.agent == "fixed":
        agent = FixedTimeController()
    else:
        agent = load_agent(args.agent, args.model)

    run_deployment(
        agent        = agent,
        agent_type   = args.agent,
        use_gui      = args.gui,
        config_file  = args.config,
        max_steps    = args.steps,
        seed         = args.seed,
    )


if __name__ == "__main__":
    main()
