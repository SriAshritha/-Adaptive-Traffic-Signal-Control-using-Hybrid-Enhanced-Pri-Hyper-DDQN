import argparse
import json
from pathlib import Path

import numpy as np

from agents.classical_agents import QLearningAgent, SARSAAgent, TD0Agent
from environments.traffic_env import TrafficEnv
from train_classical import train_qlearning, train_sarsa, train_td0


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "improved_benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_tabular(agent, seeds, mode="normal"):
    agent.epsilon = 0.0
    rows = []
    for seed in seeds:
        env = TrafficEnv(seed=seed, mode=mode)
        _, disc_state = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = agent.greedy_action(disc_state)
            _, next_disc_state, reward, done, _ = env.step(action)
            total_reward += reward
            disc_state = next_disc_state
        stats = env.get_episode_stats()
        rows.append(
            {
                "seed": seed,
                "reward": total_reward,
                "avg_queue_length": stats["avg_queue_length"],
                "avg_waiting_time": stats["avg_waiting_time"],
                "total_throughput": stats["total_throughput"],
                "phase_changes": stats["phase_changes"],
            }
        )
    return summarize(rows), rows


def evaluate_dqn_seeds(agent, seeds, mode="normal"):
    original_eps = agent.epsilon
    agent.epsilon = 0.0
    rows = []
    for seed in seeds:
        env = TrafficEnv(seed=seed, mode=mode)
        raw_state, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = agent.select_action(raw_state)
            next_raw, _, reward, done, _ = env.step(action)
            total_reward += reward
            raw_state = next_raw
        stats = env.get_episode_stats()
        rows.append(
            {
                "seed": seed,
                "reward": total_reward,
                "avg_queue_length": stats["avg_queue_length"],
                "avg_waiting_time": stats["avg_waiting_time"],
                "total_throughput": stats["total_throughput"],
                "phase_changes": stats["phase_changes"],
            }
        )
    agent.epsilon = original_eps
    return summarize(rows), rows


def summarize(rows):
    return {
        "eval_reward_mean": float(np.mean([row["reward"] for row in rows])),
        "eval_reward_std": float(np.std([row["reward"] for row in rows])),
        "eval_avg_queue_length": float(np.mean([row["avg_queue_length"] for row in rows])),
        "eval_avg_waiting_time": float(np.mean([row["avg_waiting_time"] for row in rows])),
        "eval_total_throughput": float(np.mean([row["total_throughput"] for row in rows])),
        "eval_phase_changes": float(np.mean([row["phase_changes"] for row in rows])),
    }


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare_against_reference(candidate, reference):
    if candidate["eval_reward_mean"] is None:
        return {
            "better_reward": False,
            "better_queue": False,
            "better_wait": False,
            "better_throughput": False,
        }
    return {
        "better_reward": candidate["eval_reward_mean"] > reference["eval_reward_mean"],
        "better_queue": candidate["eval_avg_queue_length"] < reference["eval_avg_queue_length"],
        "better_wait": candidate["eval_avg_waiting_time"] < reference["eval_avg_waiting_time"],
        "better_throughput": candidate["eval_total_throughput"] > reference["eval_total_throughput"],
    }


def write_report(results, references):
    all_summaries = {
        name: payload["summary"]
        for name, payload in results.items()
    }
    all_summaries.update(references)

    report = {
        "results": results,
        "references": references,
        "comparisons": {},
        "ranking": {
            "reward": sorted(all_summaries, key=lambda name: all_summaries[name]["eval_reward_mean"], reverse=True),
            "avg_queue_length": sorted(all_summaries, key=lambda name: all_summaries[name]["eval_avg_queue_length"]),
            "avg_waiting_time": sorted(all_summaries, key=lambda name: all_summaries[name]["eval_avg_waiting_time"]),
            "total_throughput": sorted(all_summaries, key=lambda name: all_summaries[name]["eval_total_throughput"], reverse=True),
        },
    }
    for name, payload in results.items():
        summary = payload["summary"]
        report["comparisons"][name] = {
            ref_name: compare_against_reference(summary, ref_summary)
            for ref_name, ref_summary in references.items()
        }

    json_path = OUT_DIR / "benchmark_results.json"
    md_path = OUT_DIR / "benchmark_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Improved Agent Benchmark",
        "",
        "| Agent | Reward | Std | Avg Queue | Avg Wait | Throughput | Phase Changes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, payload in results.items():
        summary = payload["summary"]
        lines.append(
            f"| {name} | {summary['eval_reward_mean']:.3f} | {summary['eval_reward_std']:.3f} | "
            f"{summary['eval_avg_queue_length']:.3f} | {summary['eval_avg_waiting_time']:.3f} | "
            f"{summary['eval_total_throughput']:.3f} | {summary['eval_phase_changes']:.3f} |"
        )
    lines += [
        "",
        "## Reference Models",
        "",
        "| Reference | Reward | Std | Avg Queue | Avg Wait | Throughput | Phase Changes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in references.items():
        lines.append(
            f"| {name} | {summary['eval_reward_mean']:.3f} | {summary['eval_reward_std']:.3f} | "
            f"{summary['eval_avg_queue_length']:.3f} | {summary['eval_avg_waiting_time']:.3f} | "
            f"{summary['eval_total_throughput']:.3f} | {summary['eval_phase_changes']:.3f} |"
        )

    lines += ["", "## Metric Leaders", ""]
    lines.append(f"- Best reward: {report['ranking']['reward'][0]}")
    lines.append(f"- Lowest avg queue: {report['ranking']['avg_queue_length'][0]}")
    lines.append(f"- Lowest avg wait: {report['ranking']['avg_waiting_time'][0]}")
    lines.append(f"- Highest throughput: {report['ranking']['total_throughput'][0]}")

    lines += ["", "## Head-to-Head", ""]
    for name, payload in report["comparisons"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| Reference | Better Reward | Better Queue | Better Wait | Better Throughput |")
        lines.append("|---|---|---|---|---|")
        for ref_name, comparison in payload.items():
            lines.append(
                f"| {ref_name} | {comparison['better_reward']} | {comparison['better_queue']} | "
                f"{comparison['better_wait']} | {comparison['better_throughput']} |"
            )
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-td0", type=int, default=40)
    parser.add_argument("--episodes-sarsa", type=int, default=80)
    parser.add_argument("--episodes-qlearning", type=int, default=0)
    parser.add_argument("--mode", choices=["normal", "peak", "asymmetric"], default="normal")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--eval-start-seed", type=int, default=142)
    parser.add_argument("--eval-episodes", type=int, default=30)
    args = parser.parse_args()

    eval_seeds = list(range(args.eval_start_seed, args.eval_start_seed + args.eval_episodes))
    results = {}

    td_env = TrafficEnv(seed=args.train_seed, mode=args.mode)
    td_agent, _ = train_td0(td_env, args.episodes_td0, verbose=False)
    td_summary, td_rows = evaluate_tabular(td_agent, eval_seeds, mode=args.mode)
    results["Improved TD(0)"] = {
        "summary": td_summary,
        "rows": td_rows,
        "train_episodes": args.episodes_td0,
    }

    sarsa_env = TrafficEnv(seed=args.train_seed, mode=args.mode)
    sarsa_agent, _ = train_sarsa(sarsa_env, args.episodes_sarsa, verbose=False)
    sarsa_summary, sarsa_rows = evaluate_tabular(sarsa_agent, eval_seeds, mode=args.mode)
    results["Improved SARSA"] = {
        "summary": sarsa_summary,
        "rows": sarsa_rows,
        "train_episodes": args.episodes_sarsa,
    }

    if args.episodes_qlearning > 0:
        q_env = TrafficEnv(seed=args.train_seed, mode=args.mode)
        q_agent, _ = train_qlearning(q_env, args.episodes_qlearning, verbose=False)
    else:
        q_agent = QLearningAgent()
        q_agent.save(str(ROOT / "models" / "qlearning_qtable.pkl"))
    q_summary, q_rows = evaluate_tabular(q_agent, eval_seeds, mode=args.mode)
    results["Improved Q-Learning"] = {
        "summary": q_summary,
        "rows": q_rows,
        "train_episodes": args.episodes_qlearning,
    }

    pri_ddqn = load_json(ROOT / "Pri-DDQN" / "outputs" / "eval_results.json")["single_2action"]["summary"]
    ma_sac = load_json(ROOT / "MA-SAC" / "outputs" / "eval_single_2action.json")["summary"]
    references = {
        "Pri-DDQN": pri_ddqn,
        "MA-SAC": ma_sac,
    }

    json_path, md_path = write_report(results, references)
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
