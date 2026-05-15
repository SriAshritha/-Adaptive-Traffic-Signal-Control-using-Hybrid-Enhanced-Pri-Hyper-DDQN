"""
Evaluation & Visualisation
===========================
Loads saved model metrics and generates:
  - Learning curves for all algorithms
  - Comparison bar charts (queue, wait, throughput)
  - Convergence analysis
  - Summary statistics table

Usage:
    python evaluate.py
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils.config import PATHS


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def smooth(values, window=20):
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')


def load_metrics():
    """Load all saved metrics pkls."""
    data = {}

    classical_path = os.path.join(PATHS["models_dir"], "classical_metrics.pkl")
    if os.path.exists(classical_path):
        with open(classical_path, "rb") as f:
            data["classical"] = pickle.load(f)
        print(f"Loaded classical metrics ({list(data['classical'].keys())})")

    dqn_path = os.path.join(PATHS["models_dir"], "dqn_metrics.pkl")
    if os.path.exists(dqn_path):
        with open(dqn_path, "rb") as f:
            data["dqn"] = pickle.load(f)
        print("Loaded DQN metrics")

    compare_path = "outputs/comparison_results.pkl"
    if os.path.exists(compare_path):
        with open(compare_path, "rb") as f:
            data["comparison"] = pickle.load(f)
        print(f"Loaded comparison results ({list(data['comparison'].keys())})")

    return data


# ─────────────────────────────────────────────────────────────────
# Plot 1: Learning Curves
# ─────────────────────────────────────────────────────────────────

def plot_learning_curves(data: dict, out_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Learning Curves – Classical RL Algorithms vs DQN",
                 fontsize=14, fontweight='bold')

    colors = {
        "TD(0)":      "#E74C3C",
        "SARSA":      "#3498DB",
        "Q-Learning": "#2ECC71",
        "DQN":        "#9B59B6",
    }
    metrics_to_plot = [
        ("rewards",   "Episode Reward",       axes[0]),
        ("avg_queue", "Avg Queue Length (veh)", axes[1]),
        ("avg_wait",  "Avg Waiting Time (s)",   axes[2]),
    ]

    # Classical
    if "classical" in data:
        for algo, metrics in data["classical"].items():
            c = colors.get(algo, "gray")
            for key, label, ax in metrics_to_plot:
                if key in metrics:
                    s = smooth(metrics[key])
                    ax.plot(s, label=algo, color=c, linewidth=1.5, alpha=0.85)

    # DQN
    if "dqn" in data:
        dqn = data["dqn"]
        for key, label, ax in metrics_to_plot:
            if key in dqn:
                s = smooth(dqn[key])
                ax.plot(s, label="DQN", color=colors["DQN"],
                        linewidth=2, linestyle="--")

    for key, label, ax in metrics_to_plot:
        ax.set_xlabel("Episode")
        ax.set_ylabel(label)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title(label)

    plt.tight_layout()
    path = os.path.join(out_dir, "learning_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────
# Plot 2: Comparison Bar Chart (SUMO deployment results)
# ─────────────────────────────────────────────────────────────────

def plot_comparison_bars(data: dict, out_dir: str):
    if "comparison" not in data:
        print("  No comparison results found – skipping bar chart.")
        return

    comp = data["comparison"]
    agents = list(comp.keys())
    colors = {"fixed": "#E74C3C", "qlearning": "#3498DB", "dqn": "#9B59B6"}

    metrics = [
        ("avg_queue_length",  "Avg Queue Length (veh)"),
        ("avg_waiting_time",  "Avg Waiting Time (s)"),
        ("total_throughput",  "Total Throughput (veh)"),
        ("avg_reward",        "Avg Reward"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("SUMO Deployment: Controller Comparison",
                 fontsize=14, fontweight='bold')

    for ax, (key, label) in zip(axes, metrics):
        vals = [comp[a].get(key, 0) for a in agents]
        bars = ax.bar(agents,
                      vals,
                      color=[colors.get(a, "gray") for a in agents],
                      edgecolor="black",
                      linewidth=0.8,
                      alpha=0.85)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel(label)
        ax.set_xticks(range(len(agents)))
        ax.set_xticklabels([a.upper() for a in agents], fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.02,
                    f"{v:.1f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, "controller_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────
# Plot 3: DQN Training Details
# ─────────────────────────────────────────────────────────────────

def plot_dqn_details(data: dict, out_dir: str):
    if "dqn" not in data:
        return

    dqn = data["dqn"]
    fig = plt.figure(figsize=(16, 8))
    gs  = gridspec.GridSpec(2, 3, figure=fig)
    fig.suptitle("DQN Training Analysis", fontsize=14, fontweight='bold')

    plots = [
        ("rewards",    "Episode Reward",        "blue",   gs[0, 0]),
        ("avg_queue",  "Avg Queue Length",       "red",    gs[0, 1]),
        ("avg_wait",   "Avg Waiting Time (s)",   "orange", gs[0, 2]),
        ("throughput", "Throughput (veh/ep)",    "green",  gs[1, 0]),
        ("losses",     "TD Loss",                "purple", gs[1, 1]),
    ]

    for key, label, color, pos in plots:
        ax = fig.add_subplot(pos)
        if key in dqn:
            raw = dqn[key]
            ax.plot(raw, alpha=0.2, color=color, linewidth=0.5)
            s = smooth(raw, window=min(20, len(raw)//4+1))
            ax.plot(s, color=color, linewidth=2, label=label)
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("Episode / Step")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

    # Convergence plot: rolling mean ± std
    ax = fig.add_subplot(gs[1, 2])
    if "rewards" in dqn:
        rw = np.array(dqn["rewards"])
        w  = 50
        if len(rw) >= w:
            means = [rw[max(0,i-w):i].mean() for i in range(w, len(rw))]
            stds  = [rw[max(0,i-w):i].std()  for i in range(w, len(rw))]
            xs    = np.arange(w, len(rw))
            means, stds = np.array(means), np.array(stds)
            ax.plot(xs, means, color='navy', linewidth=2, label="Rolling mean")
            ax.fill_between(xs, means-stds, means+stds,
                            alpha=0.2, color='navy', label="±1 std")
            ax.axhline(means[-10:].mean(), color='red', linestyle='--',
                       linewidth=1.5, label="Converged mean")
            ax.set_title("Convergence (Rolling 50-ep)", fontsize=10)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Reward")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "dqn_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────
# Summary Table
# ─────────────────────────────────────────────────────────────────

def print_summary_table(data: dict, last_n: int = 50):
    print("\n" + "="*75)
    print("  FINAL PERFORMANCE SUMMARY")
    print("="*75)
    print(f"  {'Algorithm':<15} {'Reward':>10} {'AvgQueue':>10} "
          f"{'AvgWait':>10} {'Throughput':>12}")
    print("  " + "-"*73)

    if "classical" in data:
        for name, m in data["classical"].items():
            r  = np.mean(m["rewards"][-last_n:])
            q  = np.mean(m["avg_queue"][-last_n:])
            w  = np.mean(m["avg_wait"][-last_n:])
            tp = np.mean(m["throughput"][-last_n:])
            print(f"  {name:<15} {r:>10.3f} {q:>10.2f} {w:>10.1f} {tp:>12.0f}")

    if "dqn" in data:
        m = data["dqn"]
        r  = np.mean(m["rewards"][-last_n:])
        q  = np.mean(m["avg_queue"][-last_n:])
        w  = np.mean(m["avg_wait"][-last_n:])
        tp = np.mean(m["throughput"][-last_n:])
        print(f"  {'DQN (Dueling)':<15} {r:>10.3f} {q:>10.2f} {w:>10.1f} {tp:>12.0f}")

    print("="*75)


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    os.makedirs("outputs", exist_ok=True)
    out_dir = "outputs"

    print("Loading metrics…")
    data = load_metrics()

    if not data:
        print("No metrics found. Run train_classical.py and/or train_dqn.py first.")
        return

    print("\nGenerating plots…")
    plot_learning_curves(data, out_dir)
    plot_dqn_details(data, out_dir)
    plot_comparison_bars(data, out_dir)
    print_summary_table(data)

    print(f"\nAll evaluation outputs saved to ./{out_dir}/")


if __name__ == "__main__":
    main()
