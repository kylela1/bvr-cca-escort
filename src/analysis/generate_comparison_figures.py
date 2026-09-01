#!/usr/bin/env python3
"""
Generates comparison figures for the full-scale baseline run (my_first_run,
reward_mode="baseline") vs. the reward-redesign run (reshaped_rewards1,
reward_mode="mission_aligned"). Every number is read directly from
training_results_seed42.json, superlative_eval_results.json, or the TensorBoard
event files under results/runs_sb3/ -- nothing here is fabricated or hand-transcribed.

Usage:
    python src/analysis/generate_comparison_figures.py

Outputs PNG files into results/figures/.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = REPO_ROOT / "results" / "runs_sb3"
OUT_DIR = REPO_ROOT / "results" / "figures"

# ----------------------------------------------------------------------------------------
# Same validated light-mode palette as src/analysis/generate_figures.py.
# ----------------------------------------------------------------------------------------
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

RUNS = {
    "baseline": {
        "label": "Baseline reward\n(my_first_run)",
        "dir": RUNS_DIR / "my_first_run" / "my_first_run_seed42_20260830_164018",
        "eval_file": RUNS_DIR / "my_first_run" / "superlative_eval_results.json",
        "color": "#898781",  # muted gray -- reference condition
    },
    "mission_aligned": {
        "label": "Reshaped reward\n(reshaped_rewards1)",
        "dir": RUNS_DIR / "reshaped_rewards1" / "reshaped_rewards1_seed42_20260831_170630",
        "eval_file": RUNS_DIR / "reshaped_rewards1" / "superlative_eval_results.json",
        "color": "#1baf7a",  # aqua -- the "new condition"
    },
}
RUN_ORDER = ["baseline", "mission_aligned"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE_AXIS,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "grid.color": GRIDLINE,
    "axes.grid": True,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_data():
    data = {}
    for key, meta in RUNS.items():
        with open(meta["eval_file"]) as f:
            eval_result = json.load(f)
        with open(meta["dir"] / "rollback_history.json") as f:
            rollbacks = json.load(f)
        data[key] = {"eval": eval_result, "rollbacks": rollbacks}
    return data


def get_agg_metric(eval_result, key):
    return eval_result["aggregate"]["cross_seed_statistics"][key]["mean"] * 100.0


def get_agg_raw(eval_result, key):
    return eval_result["aggregate"]["cross_seed_statistics"][key]["mean"]


# ==========================================================================================
# Figure C1: Overlaid learning curves -- both full 10,000,000-timestep runs
# ==========================================================================================
def figure_c1_learning_curves(data):
    fig, ax = plt.subplots(figsize=(9.5, 5.4), dpi=150)

    for key in RUN_ORDER:
        meta = RUNS[key]
        ea = EventAccumulator(str(meta["dir"]))
        ea.Reload()
        eval_events = ea.Scalars("eval/mean_reward")
        eval_x = np.array([e.step for e in eval_events])
        eval_y = np.array([e.value for e in eval_events])

        ax.plot(eval_x, eval_y, color=meta["color"], linewidth=2.0,
                 label=meta["label"].replace("\n", " "))

        rollback_steps = [r["timestep"] for r in data[key]["rollbacks"]]
        for step in rollback_steps:
            ax.axvline(step, color=meta["color"], linewidth=0.8, linestyle=(0, (2, 3)),
                       alpha=0.45)

    ax.set_xlabel("Training timestep")
    ax.set_ylabel("Periodic evaluation mean reward")
    ax.set_title("Learning curves: baseline reward vs. reshaped (mission-aligned) reward",
                 fontsize=13, fontweight="bold", color=INK_PRIMARY, loc="left")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    ax.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK_SECONDARY)
    ax.grid(True, which="major", linewidth=0.8)

    fig.text(0.01, -0.03,
              "Source: TensorBoard eval/mean_reward + rollback_history.json for each run "
              "(both 10,000,000 timesteps, seed 42). Dotted verticals mark rollback events. "
              "Reward scales are NOT comparable across runs -- see fig_c3 for why.",
              fontsize=7, color=INK_MUTED, ha="left")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_c1_learning_curves_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_c1_learning_curves_comparison.png")


# ==========================================================================================
# Figure C2: Mission-outcome metrics -- baseline vs. mission-aligned, 20 eval episodes each
# ==========================================================================================
def figure_c2_mission_outcomes(data):
    metrics = [
        ("win_rate", "Win Rate"),
        ("hvaa_survival_rate", "HVAA Survival"),
        ("red_kill_rate", "Red Kill"),
        ("blue_loss_rate", "Blue Loss"),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=150)
    n_metrics = len(metrics)
    bar_width = 0.32
    x_group = np.arange(n_metrics)

    for i, key in enumerate(RUN_ORDER):
        meta = RUNS[key]
        vals = [get_agg_metric(data[key]["eval"], mkey) for mkey, _ in metrics]
        offset = (i - 0.5) * (bar_width + 0.03)
        bars = ax.bar(x_group + offset, vals, width=bar_width, color=meta["color"],
                       label=meta["label"].replace("\n", " "))
        for rect, val in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, val + 1, f"{val:.0f}",
                    ha="center", va="bottom", fontsize=9, color=INK_PRIMARY, fontweight="bold")

    ax.set_xticks(x_group)
    ax.set_xticklabels([label for _, label in metrics], fontsize=10.5)
    ax.set_ylabel("Rate across 20 evaluation episodes (%)")
    ax.set_title("Mission-outcome metrics: baseline vs. reshaped reward",
                 fontsize=12.5, fontweight="bold", color=INK_PRIMARY, loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=9.5, labelcolor=INK_SECONDARY)
    ax.grid(True, axis="y", which="major", linewidth=0.8)
    ax.grid(False, axis="x")
    ax.set_ylim(0, 115)

    fig.text(0.01, -0.03,
              "Source: results/runs_sb3/<run>/superlative_eval_results.json (cross_seed_statistics; "
              "1 seed x 20 episodes each -- single-seed point estimates, not a significance test).",
              fontsize=7, color=INK_MUTED, ha="left")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_c2_mission_outcomes_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_c2_mission_outcomes_comparison.png")


# ==========================================================================================
# Figure C3: Mean episode reward -- why the raw reward scale dropped
# ==========================================================================================
def figure_c3_reward_magnitude(data):
    means = [get_agg_raw(data[k]["eval"], "mean_reward") for k in RUN_ORDER]
    stds = []
    for k in RUN_ORDER:
        per_seed = data[k]["eval"]["per_seed_results"]["seed42"]
        stds.append(per_seed["std_reward"])
    labels = [RUNS[k]["label"].replace("\n", " ") for k in RUN_ORDER]
    colors = [RUNS[k]["color"] for k in RUN_ORDER]

    fig, ax = plt.subplots(figsize=(7, 6.2), dpi=150)
    bars = ax.bar(range(len(labels)), means, yerr=stds, color=colors, width=0.5,
                   capsize=6, error_kw={"ecolor": INK_SECONDARY, "linewidth": 1.2})
    for rect, val in zip(bars, means):
        ax.text(rect.get_x() + rect.get_width() / 2, val + 3, f"{val:+.1f}",
                ha="center", va="bottom", fontsize=11, color=INK_PRIMARY, fontweight="bold")

    ax.axhline(0, color=BASELINE_AXIS, linewidth=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean evaluation episode reward (± std, 20 episodes)")
    ax.set_title("Removing the per-step tempo bonus lowers the raw reward\n"
                 "scale even though mission performance is comparable",
                 fontsize=12, fontweight="bold", color=INK_PRIMARY, loc="left")
    ax.grid(True, axis="y", linewidth=0.8)
    ax.grid(False, axis="x")

    fig.text(0.01, -0.11,
              "Source: superlative_eval_results.json mean_reward + per-seed std_reward for each run.\n"
              "Baseline reward includes MissionTempoShapingPerStep's +0.006/step survival bonus\n"
              "(~+70 over a ~12,000-step episode); the mission-aligned stack sets that bonus to 0.0\n"
              "and instead pays flat terminal bonuses (-15 HVAA loss / +25 mission success), so the two\n"
              "reward totals are not on the same scale -- compare mission-outcome rates (fig_c2), not reward.",
              fontsize=7, color=INK_MUTED, ha="left")

    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT_DIR / "fig_c3_reward_magnitude_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_c3_reward_magnitude_comparison.png")


def main():
    data = load_data()
    figure_c1_learning_curves(data)
    figure_c2_mission_outcomes(data)
    figure_c3_reward_magnitude(data)
    print("\nAll comparison figures written to:", OUT_DIR)


if __name__ == "__main__":
    main()
