import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.classical_agents import QLearningAgent, SARSAAgent, TD0Agent
from environments.traffic_env import TrafficEnv as TrafficEnv2Action


OUTPUT_DIR = ROOT / "final_results"
OUTPUT_DIR.mkdir(exist_ok=True)


def evaluate_2action_classical(agent_cls, model_path: Path, label: str):
    agent = agent_cls()
    agent.load(str(model_path))
    agent.epsilon = 0.0
    rows = []
    for seed in range(142, 172):
        env = TrafficEnv2Action(seed=seed, mode="normal")
        _, disc_state = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            if label == "TD(0)":
                action = agent.greedy_action(disc_state)
            else:
                action = agent.greedy_action(disc_state)
            _, next_disc_state, reward, done, _ = env.step(action)
            total_reward += reward
            disc_state = next_disc_state
        stats = env.get_episode_stats()
        rows.append(
            {
                "reward": total_reward,
                "avg_queue_length": stats["avg_queue_length"],
                "avg_waiting_time": stats["avg_waiting_time"],
                "total_throughput": stats["total_throughput"],
                "total_delay": None,
                "phase_changes": stats["phase_changes"],
            }
        )
    return {
        "summary": {
            "eval_reward_mean": float(np.mean([r["reward"] for r in rows])),
            "eval_reward_std": float(np.std([r["reward"] for r in rows])),
            "eval_avg_queue_length": float(np.mean([r["avg_queue_length"] for r in rows])),
            "eval_avg_waiting_time": float(np.mean([r["avg_waiting_time"] for r in rows])),
            "eval_total_throughput": float(np.mean([r["total_throughput"] for r in rows])),
            "eval_total_delay": None,
            "eval_phase_changes": float(np.mean([r["phase_changes"] for r in rows])),
        },
        "rows": rows,
    }


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def add_row(rows, section, method, summary):
    rows.append(
        {
            "section": section,
            "method": method,
            "eval_reward_mean": summary["eval_reward_mean"],
            "eval_reward_std": summary["eval_reward_std"],
            "eval_avg_queue_length": summary["eval_avg_queue_length"],
            "eval_avg_waiting_time": summary["eval_avg_waiting_time"],
            "eval_total_throughput": summary["eval_total_throughput"],
            "eval_total_delay": summary["eval_total_delay"],
            "eval_phase_changes": summary["eval_phase_changes"],
        }
    )


def write_outputs(rows):
    csv_path = OUTPUT_DIR / "benchmark_table.csv"
    json_path = OUTPUT_DIR / "benchmark_table.json"
    md_path = OUTPUT_DIR / "benchmark_table.md"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "section",
                "method",
                "eval_reward_mean",
                "eval_reward_std",
                "eval_avg_queue_length",
                "eval_avg_waiting_time",
                "eval_total_throughput",
                "eval_total_delay",
                "eval_phase_changes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = [
        "# Final Benchmark Table",
        "",
        "## Section 1: Single Intersection (2-action)",
        "",
        "| Method | Reward | Std | Avg Queue | Avg Wait | Throughput | Delay | Phase Changes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in [r for r in rows if r["section"] == "Single Intersection (2-action)"]:
        delay = "NA" if row["eval_total_delay"] is None else f"{row['eval_total_delay']:.3f}"
        lines.append(
            f"| {row['method']} | {row['eval_reward_mean']:.3f} | {row['eval_reward_std']:.3f} | "
            f"{row['eval_avg_queue_length']:.3f} | {row['eval_avg_waiting_time']:.3f} | "
            f"{row['eval_total_throughput']:.3f} | {delay} | {row['eval_phase_changes']:.3f} |"
        )
    lines += [
        "",
        "## Section 2: 4-Way Movement Benchmark (4-action)",
        "",
        "| Method | Reward | Std | Avg Queue | Avg Wait | Throughput | Delay | Phase Changes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in [r for r in rows if r["section"] == "4-Way Movement Benchmark (4-action)"]:
        delay = "NA" if row["eval_total_delay"] is None else f"{row['eval_total_delay']:.3f}"
        lines.append(
            f"| {row['method']} | {row['eval_reward_mean']:.3f} | {row['eval_reward_std']:.3f} | "
            f"{row['eval_avg_queue_length']:.3f} | {row['eval_avg_waiting_time']:.3f} | "
            f"{row['eval_total_throughput']:.3f} | {delay} | {row['eval_phase_changes']:.3f} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, json_path, md_path


def main():
    rows = []

    td0_2 = evaluate_2action_classical(TD0Agent, ROOT / "models" / "td0_qtable.pkl", "TD(0)")
    sarsa_2 = evaluate_2action_classical(SARSAAgent, ROOT / "models" / "sarsa_qtable.pkl", "SARSA")
    ql_2 = evaluate_2action_classical(QLearningAgent, ROOT / "models" / "qlearning_qtable.pkl", "Q-Learning")

    ma_sac_2 = load_json(ROOT / "MA-SAC" / "outputs" / "eval_single_2action.json")
    pri_ddqn = load_json(ROOT / "Pri-DDQN" / "outputs" / "eval_results.json")
    fourway = load_json(ROOT / "main_experiment_6action" / "outputs" / "final_results.json")
    ma_sac_4 = load_json(ROOT / "MA-SAC" / "outputs" / "eval_results.json")

    add_row(rows, "Single Intersection (2-action)", "TD(0)", td0_2["summary"])
    add_row(rows, "Single Intersection (2-action)", "SARSA", sarsa_2["summary"])
    add_row(rows, "Single Intersection (2-action)", "Q-Learning", ql_2["summary"])
    add_row(rows, "Single Intersection (2-action)", "MA-SAC", ma_sac_2["summary"])
    add_row(rows, "Single Intersection (2-action)", "Pri-DDQN", pri_ddqn["single_2action"]["summary"])

    add_row(rows, "4-Way Movement Benchmark (4-action)", "TD(0)", fourway["algorithms"]["TD(0)"]["summary"])
    add_row(rows, "4-Way Movement Benchmark (4-action)", "SARSA", fourway["algorithms"]["SARSA"]["summary"])
    add_row(rows, "4-Way Movement Benchmark (4-action)", "Q-Learning", fourway["algorithms"]["Q-Learning"]["summary"])
    add_row(rows, "4-Way Movement Benchmark (4-action)", "MA-SAC", ma_sac_4["summary"])
    add_row(rows, "4-Way Movement Benchmark (4-action)", "Pri-DDQN", pri_ddqn["fourway_4action"]["summary"])

    csv_path, json_path, md_path = write_outputs(rows)
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
