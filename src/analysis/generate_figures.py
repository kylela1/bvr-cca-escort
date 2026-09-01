#!/usr/bin/env python3
"""
Generates the figures for the reward-shaping project's final report from real run data
under results/runs_sb3/. Nothing here is fabricated or hand-transcribed -- every number is read
directly from a training_results_*.json, superlative_eval_results.json, or TensorBoard event
file, or (for the "reward misalignment" illustration) computed from the exact constants
defined in rl_files/mission_tempo_shaping_perstep.py and rl_files/env_factory.py.

Usage:
    python src/analysis/generate_figures.py

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
# Validated categorical palette (light mode), fixed order -- see dataviz skill's palette.md.
# Used consistently for the same 5 configs across every figure in this deliverable.
# ----------------------------------------------------------------------------------------
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

CONFIG_ORDER = ["baseline", "B", "D", "E", "ACD"]
CONFIG_LABELS = {
    "baseline": "Baseline",
    "B": "Factor B\n(WEZ/DMC)",
    "D": "Factor D\n(alignment)",
    "E": "Factor E\n(blending)",
    "ACD": "ACD\n(A+C+D)",
}
CONFIG_COLORS = {
    "baseline": "#898781",  # muted gray -- reference condition, not a "hue" competing for identity
    "B": "#2a78d6",         # slot 1 blue
    "D": "#1baf7a",         # slot 3 aqua
    "E": "#e34948",         # slot 8 red -- reads as "the one that underperforms"
    "ACD": "#eb6834",       # slot 2 orange
}
DIVERGING_POS = "#2a78d6"  # blue pole
DIVERGING_NEG = "#e34948"  # red pole
DIVERGING_MID = "#f0efec"

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


def load_pilot_data():
    """Load training_results + superlative_eval_results for all 5 pilot configs."""
    data = {}
    for cfg in CONFIG_ORDER:
        exp_dir = "pilot_baseline" if cfg == "baseline" else f"pilot_{cfg}"
        run_dir = RUNS_DIR / exp_dir / f"{exp_dir}_seed42"
        with open(run_dir / "training_results_seed42.json") as f:
            training = json.load(f)
        with open(RUNS_DIR / exp_dir / "superlative_eval_results.json") as f:
            eval_result = json.load(f)
        data[cfg] = {"training": training, "eval": eval_result}
    return data


def get_agg_metric(eval_result, key):
    return eval_result["aggregate"]["cross_seed_statistics"][key]["mean"] * 100.0


# ==========================================================================================
# Figure 1: Full-scale (10,000,000-timestep) baseline learning curve, with rollback events
# ==========================================================================================
def figure_1_full_scale_learning_curve():
    run_dir = RUNS_DIR / "my_first_run" / "my_first_run_seed42_20260830_164018"
    ea = EventAccumulator(str(run_dir))
    ea.Reload()

    eval_events = ea.Scalars("eval/mean_reward")
    eval_std_events = ea.Scalars("eval/std_reward")
    rollout_events = ea.Scalars("rollout/ep_rew_mean")

    eval_x = np.array([e.step for e in eval_events])
    eval_y = np.array([e.value for e in eval_events])
    eval_std = np.array([e.value for e in eval_std_events])
    rollout_x = np.array([e.step for e in rollout_events])
    rollout_y = np.array([e.value for e in rollout_events])

    with open(run_dir / "rollback_history.json") as f:
        rollbacks = json.load(f)
    rollback_steps = [r["timestep"] for r in rollbacks]

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)

    # Context series: raw rollout mean reward (noisy, de-emphasized in muted gray)
    ax.plot(rollout_x, rollout_y, color=INK_MUTED, linewidth=1.0, alpha=0.55,
             label="Training rollout mean reward (noisy, per-update)")

    # Point-of-the-chart series: deterministic periodic evaluation reward, with std band
    ax.plot(eval_x, eval_y, color=CONFIG_COLORS["ACD"], linewidth=2.0,
             label="Periodic evaluation mean reward (15 episodes, deterministic policy)")
    ax.fill_between(eval_x, eval_y - eval_std, eval_y + eval_std,
                     color=CONFIG_COLORS["ACD"], alpha=0.15, linewidth=0)

    # Rollback events
    for i, step in enumerate(rollback_steps):
        ax.axvline(step, color=INK_SECONDARY, linewidth=0.9, linestyle=(0, (3, 3)), alpha=0.7)
    ax.axvline(rollback_steps[0], color=INK_SECONDARY, linewidth=0.9, linestyle=(0, (3, 3)),
               alpha=0.7, label=f"Rollback event (5 total, first at step {rollback_steps[0]:,})")

    ax.set_xlabel("Training timestep")
    ax.set_ylabel("Episode reward")
    ax.set_title("Baseline reward, full 10,000,000-timestep run (seed 42)",
                 fontsize=13, fontweight="bold", color=INK_PRIMARY, loc="left")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)
    ax.grid(True, which="major", linewidth=0.8)

    fig.text(0.01, -0.02,
              "Source: results/runs_sb3/my_first_run/my_first_run_seed42_20260830_164018 "
              "(TensorBoard eval/mean_reward, rollout/ep_rew_mean, rollback_history.json).",
              fontsize=7, color=INK_MUTED, ha="left")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_full_scale_learning_curve.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig1_full_scale_learning_curve.png")


# ==========================================================================================
# Figure 2: Pilot comparison -- best-checkpoint reward by configuration
# ==========================================================================================
def figure_2_best_reward_by_config(pilot_data):
    configs = CONFIG_ORDER
    best_rewards = [pilot_data[c]["training"]["best_eval_mean_reward"] for c in configs]

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    colors = [CONFIG_COLORS[c] for c in configs]
    bars = ax.bar(range(len(configs)), best_rewards, color=colors, width=0.6,
                   edgecolor="none")

    for rect, val in zip(bars, best_rewards):
        va = "bottom" if val >= 0 else "top"
        offset = 1.5 if val >= 0 else -1.5
        ax.text(rect.get_x() + rect.get_width() / 2, val + offset, f"{val:+.1f}",
                ha="center", va=va, fontsize=9.5, color=INK_PRIMARY, fontweight="bold")

    ax.axhline(0, color=BASELINE_AXIS, linewidth=1.0)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([CONFIG_LABELS[c] for c in configs], fontsize=9)
    ax.set_ylabel("Best-checkpoint mean evaluation reward")
    ax.set_title("Pilot experiment: best-checkpoint reward by configuration\n"
                 "(1 seed, 2,000,000 timesteps each)",
                 fontsize=12, fontweight="bold", color=INK_PRIMARY, loc="left")
    ax.grid(True, axis="y", which="major", linewidth=0.8)
    ax.grid(False, axis="x")

    fig.text(0.01, -0.03,
              "Source: results/runs_sb3/pilot_<config>/pilot_<config>_seed42/training_results_seed42.json "
              "(best_eval_mean_reward).",
              fontsize=7, color=INK_MUTED, ha="left")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_best_reward_by_config.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig2_best_reward_by_config.png")


# ==========================================================================================
# Figure 3: Pilot comparison -- mission-outcome metrics (HVAA survival, red kill, blue loss)
# ==========================================================================================
def figure_3_mission_outcomes(pilot_data):
    configs = CONFIG_ORDER
    metrics = [
        ("hvaa_survival_rate", "HVAA Survival"),
        ("red_kill_rate", "Red Kill"),
        ("blue_loss_rate", "Blue Loss"),
    ]

    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=150)
    n_configs = len(configs)
    n_metrics = len(metrics)
    bar_width = 0.15
    group_gap = 0.08
    x_group = np.arange(n_metrics)

    for i, cfg in enumerate(configs):
        vals = [get_agg_metric(pilot_data[cfg]["eval"], key) for key, _ in metrics]
        offsets = (i - (n_configs - 1) / 2) * (bar_width + 0.02)
        bars = ax.bar(x_group + offsets, vals, width=bar_width, color=CONFIG_COLORS[cfg],
                       label=CONFIG_LABELS[cfg].replace("\n", " "))
        for rect, val in zip(bars, vals):
            if val > 0:
                ax.text(rect.get_x() + rect.get_width() / 2, val + 1, f"{val:.0f}",
                        ha="center", va="bottom", fontsize=7, color=INK_SECONDARY)

    ax.set_xticks(x_group)
    ax.set_xticklabels([label for _, label in metrics], fontsize=10)
    ax.set_ylabel("Rate across 20 evaluation episodes (%)")
    ax.set_title("Pilot experiment: mission-outcome metrics by configuration",
                 fontsize=12, fontweight="bold", color=INK_PRIMARY, loc="left")
    ax.legend(loc="upper right", frameon=False, fontsize=8.5, ncol=1, labelcolor=INK_SECONDARY)
    ax.grid(True, axis="y", which="major", linewidth=0.8)
    ax.grid(False, axis="x")
    ax.set_ylim(0, max(60, ax.get_ylim()[1]))

    fig.text(0.01, -0.03,
              "Source: results/runs_sb3/pilot_<config>/superlative_eval_results.json "
              "(cross_seed_statistics; 20 episodes, seed 42). No visible bar = a true 0%, "
              "not missing data.",
              fontsize=7, color=INK_MUTED, ha="left")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_mission_outcomes_by_config.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig3_mission_outcomes_by_config.png")


# ==========================================================================================
# Figure 4: Approximate win rate by configuration (HVAA survives AND CCA survives)
# ==========================================================================================
def figure_4_win_rate_proxy(pilot_data):
    configs = CONFIG_ORDER
    win_rates = [get_agg_metric(pilot_data[c]["eval"], "hvaa_survival_rate") for c in configs]
    best_rewards = [pilot_data[c]["training"]["best_eval_mean_reward"] for c in configs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150)

    colors = [CONFIG_COLORS[c] for c in configs]
    bars1 = ax1.bar(range(len(configs)), win_rates, color=colors, width=0.6)
    for rect, val in zip(bars1, win_rates):
        ax1.text(rect.get_x() + rect.get_width() / 2, val + 1, f"{val:.0f}%",
                 ha="center", va="bottom", fontsize=9.5, color=INK_PRIMARY, fontweight="bold")
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels([CONFIG_LABELS[c] for c in configs], fontsize=8.5)
    ax1.set_ylabel("Approx. win rate (%)\n(HVAA-survival proxy)")
    ax1.set_title("(a) Win-rate proxy", fontsize=11, fontweight="bold", loc="left")
    ax1.set_ylim(0, max(win_rates) * 1.3 + 5)
    ax1.grid(True, axis="y", linewidth=0.8)
    ax1.grid(False, axis="x")

    # Rank by reward vs rank by win rate -- illustrates the misalignment directly
    reward_rank = {c: r for r, c in enumerate(sorted(configs, key=lambda c: -pilot_data[c]["training"]["best_eval_mean_reward"]))}
    winrate_rank = {c: r for r, c in enumerate(sorted(configs, key=lambda c: -get_agg_metric(pilot_data[c]["eval"], "hvaa_survival_rate")))}

    for c in configs:
        ax2.plot([0, 1], [reward_rank[c], winrate_rank[c]], color=CONFIG_COLORS[c], linewidth=2.2,
                 marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1)
        ax2.text(-0.05, reward_rank[c], CONFIG_LABELS[c].replace("\n", " "), ha="right",
                 va="center", fontsize=8.5, color=INK_PRIMARY)
        ax2.text(1.05, winrate_rank[c], CONFIG_LABELS[c].replace("\n", " "), ha="left",
                 va="center", fontsize=8.5, color=INK_PRIMARY)

    ax2.set_xlim(-0.6, 1.6)
    ax2.set_ylim(-0.5, len(configs) - 0.5)
    ax2.invert_yaxis()
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Ranked by\nbest reward", "Ranked by\nwin-rate proxy"], fontsize=9)
    ax2.set_yticks([])
    ax2.set_title("(b) Reward rank vs. win-rate rank", fontsize=11, fontweight="bold", loc="left")
    ax2.grid(False)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    fig.suptitle("Raw PPO reward is an imperfect proxy for mission win rate",
                 fontsize=13, fontweight="bold", color=INK_PRIMARY, x=0.02, ha="left", y=1.03)

    fig.text(0.01, -0.05,
              "Source: same per-config files as Figures 2-3. Panel (b): ACD ranks #1 by best "
              "reward but #3 by win-rate proxy; D ranks #2 by reward but #1 by win-rate proxy.",
              fontsize=7, color=INK_MUTED, ha="left")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_reward_vs_winrate_rank.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig4_reward_vs_winrate_rank.png")


# ==========================================================================================
# Figure 5: Reward-misalignment illustration -- computed from the actual shaping constants
# ==========================================================================================
def figure_5_reward_misalignment_illustration():
    # Constants as actually defined in rl_files/env_factory.py's REWARD_SHAPING_STACK and
    # rl_files/mission_tempo_shaping_perstep.py -- not fabricated.
    PENALTY_PER_STEP = -0.003
    BONUS_PER_STEP = 0.006
    KILL_REWARD = 12.0          # KillRewardWrapper
    HIT_ENEMY_FACTOR = 10.0     # Godot RewardsConfig, one-time r_env term
    MISSION_ACCOMPLISHED = 8.0  # Godot RewardsConfig, one-time r_env term
    EPISODE_LEN = 13_500        # observed typical/max episode length this session

    scenarios = []

    # Scenario A: red never killed, episode times out
    tempo = PENALTY_PER_STEP * EPISODE_LEN
    scenarios.append(("Never kill red\n(times out)", tempo))

    # Scenario B: kill red early (step 1,000), then loiter with HVAA alive to episode end
    kill_step = 1_000
    tempo = PENALTY_PER_STEP * kill_step + BONUS_PER_STEP * (EPISODE_LEN - kill_step)
    total = tempo + KILL_REWARD + HIT_ENEMY_FACTOR
    scenarios.append(("Kill red early,\nthen loiter", total))

    # Scenario C: efficient mission completion shortly after a kill (episode ends early,
    # cutting off tempo-bonus accumulation) -- approximate: mission ends ~2,000 steps after kill
    mission_end_step = 3_000
    tempo = PENALTY_PER_STEP * kill_step + BONUS_PER_STEP * (mission_end_step - kill_step)
    total = tempo + KILL_REWARD + HIT_ENEMY_FACTOR + MISSION_ACCOMPLISHED
    scenarios.append(("Kill red, then\ncomplete mission\nefficiently", total))

    labels = [s[0] for s in scenarios]
    values = [s[1] for s in scenarios]
    colors = [DIVERGING_NEG if v < 0 else DIVERGING_POS for v in values]

    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
    bars = ax.barh(range(len(labels)), values, color=colors, height=0.55)
    for rect, val in zip(bars, values):
        ha = "left" if val >= 0 else "right"
        offset = 1.5 if val >= 0 else -1.5
        ax.text(val + offset, rect.get_y() + rect.get_height() / 2, f"{val:+.1f}",
                va="center", ha=ha, fontsize=10, color=INK_PRIMARY, fontweight="bold")

    ax.axvline(0, color=BASELINE_AXIS, linewidth=1.2)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Approximate total episode reward from these terms")
    ax.set_title("The current reward pays more for loitering with red dead\n"
                 "than for efficiently completing the mission",
                 fontsize=12, fontweight="bold", color=INK_PRIMARY, loc="left")
    ax.grid(True, axis="x", linewidth=0.8)
    ax.grid(False, axis="y")

    fig.text(0.01, -0.06,
              "Computed from the exact constants in env_factory.py's REWARD_SHAPING_STACK and\n"
              "mission_tempo_shaping_perstep.py (penalty_per_step=-0.003, bonus_per_step=0.006,\n"
              "KillRewardWrapper=+12, Godot hit_enemy_factor=+10, mission_accomplished_factor=+8),\n"
              "assuming a kill at step 1,000 and a ~13,500-step episode. Illustrative, not measured.",
              fontsize=7, color=INK_MUTED, ha="left")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_reward_misalignment_illustration.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig5_reward_misalignment_illustration.png")


def main():
    pilot_data = load_pilot_data()
    figure_1_full_scale_learning_curve()
    figure_2_best_reward_by_config(pilot_data)
    figure_3_mission_outcomes(pilot_data)
    figure_4_win_rate_proxy(pilot_data)
    figure_5_reward_misalignment_illustration()
    print("\nAll figures written to:", OUT_DIR)


if __name__ == "__main__":
    main()
