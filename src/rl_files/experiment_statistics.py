"""
Cross-seed statistical analysis for one experiment's training curves: per-seed learning curves
overlaid with a cross-seed mean + 95% CI band (t-distribution, reusing the same
stats_utils.t_distribution_ci evaluate_experiment_seeds.py already uses for end-of-training
metrics), plus a Peak/AUC/AlgScore summary in the style of a reference benchmark chart the user
supplied. No equivalent of this existed anywhere in this repo before -- see
docs/THESIS_REPRODUCTION_GUIDE.md Sec 6.

PROVENANCE NOTE: the "AlgScore = 0.6*Peak_LCB + 0.4*AUC_LCB" formula and the Peak_LCB/AUC_LCB
names are only known from that reference chart's title text, not from any source code -- nothing
resembling this exists elsewhere in this repo or its git history on any branch. This module is a
best-effort reconstruction, not a verified reproduction of Bieger's original tool:
  - Peak_LCB / AUC_LCB are read as "the lower bound of a 95% cross-seed CI on peak return / AUC."
  - AUC is normalized by step range (time-averaged reward), not raw trapezoidal area, so it's on
    the same reward scale as peak return -- otherwise a 0.6/0.4 weighted sum of the two would mix
    incompatible units. This is this implementation's own choice.

Called automatically from evaluate_experiment_seeds.py's default (superlative) evaluation mode
after per-seed episode results are computed, unless --skip-figures is passed. Every failure mode
here (missing matplotlib, no TensorBoard event files, only one seed) degrades to a printed
warning and a skipped figure -- it must never take down a run that already paid for real Godot
episode rollouts.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from stats_utils import t_distribution_ci

EVAL_REWARD_TAG = "eval/mean_reward"

# Same validated categorical palette used by src/analysis/generate_figures.py and
# generate_comparison_figures.py -- kept visually consistent with the rest of this project's
# figures rather than matplotlib's default color cycle.
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
SEED_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def find_seed_run_dirs(experiment_dir: Path) -> Dict[str, Path]:
    """seed_name -> its run directory (the one directly holding events.out.tfevents.*),
    matching the same '*seed*' glob evaluate_experiment_seeds.py's find_seed_models() uses."""
    run_dirs: Dict[str, Path] = {}
    for seed_dir in sorted(experiment_dir.glob("*seed*")):
        if not seed_dir.is_dir():
            continue
        match = re.search(r"seed(\d+)", seed_dir.name)
        seed_name = f"seed{match.group(1)}" if match else seed_dir.name
        run_dirs[seed_name] = seed_dir
    return run_dirs


def load_seed_learning_curves(
    experiment_dir: Path, tag: str = EVAL_REWARD_TAG
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """seed_name -> (steps, values) read from each seed's own TensorBoard event file."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for seed_name, run_dir in find_seed_run_dirs(experiment_dir).items():
        if not list(run_dir.glob("events.out.tfevents.*")):
            continue
        ea = EventAccumulator(str(run_dir))
        ea.Reload()
        if tag not in ea.Tags().get("scalars", []):
            continue
        events = ea.Scalars(tag)
        if not events:
            continue
        steps = np.array([e.step for e in events], dtype=float)
        values = np.array([e.value for e in events], dtype=float)
        curves[seed_name] = (steps, values)
    return curves


def align_curves_to_common_grid(
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]], n_points: int = 100
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Interpolate every seed's curve onto one shared step grid so they can be averaged
    pointwise, even though seeds don't necessarily log eval events at identical steps (a
    rollback-affected run, or a run that stopped early, can shift the exact step values).
    The grid only extends to the shortest seed's final logged step, so no seed is extrapolated
    past data it actually has.
    """
    if not curves:
        return np.array([]), {}
    max_common_step = min(steps[-1] for steps, _ in curves.values())
    grid = np.linspace(0, max_common_step, n_points)
    aligned = {seed: np.interp(grid, steps, values) for seed, (steps, values) in curves.items()}
    return grid, aligned


def compute_cross_seed_band(aligned: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Per-grid-point cross-seed mean and 95% CI (t-distribution across seeds)."""
    matrix = np.stack(list(aligned.values()), axis=0)  # (n_seeds, n_grid_points)
    n_points = matrix.shape[1]
    mean = np.zeros(n_points)
    lower = np.zeros(n_points)
    upper = np.zeros(n_points)
    for i in range(n_points):
        ci = t_distribution_ci(list(matrix[:, i]))
        mean[i], lower[i], upper[i] = ci.mean, ci.ci_lower, ci.ci_upper
    return {"mean": mean, "ci_lower": lower, "ci_upper": upper}


def compute_peak_auc_algscore(curves: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict:
    """Per-seed peak return + time-averaged AUC, their cross-seed CIs, and AlgScore. See this
    module's docstring for the provenance caveat on the exact formula/normalization used."""
    peaks: List[float] = []
    aucs: List[float] = []
    for steps, values in curves.values():
        peaks.append(float(np.max(values)))
        step_range = steps[-1] - steps[0]
        aucs.append(float(np.trapezoid(values, steps) / step_range) if step_range > 0 else float(values[-1]))

    peak_ci = t_distribution_ci(peaks)
    auc_ci = t_distribution_ci(aucs)
    alg_score = 0.6 * peak_ci.ci_lower + 0.4 * auc_ci.ci_lower

    return {
        "per_seed_peak_return": peaks,
        "per_seed_auc_time_averaged": aucs,
        "peak_return_ci": {
            "mean": peak_ci.mean, "ci_lower": peak_ci.ci_lower,
            "ci_upper": peak_ci.ci_upper, "std": peak_ci.std,
        },
        "auc_time_averaged_ci": {
            "mean": auc_ci.mean, "ci_lower": auc_ci.ci_lower,
            "ci_upper": auc_ci.ci_upper, "std": auc_ci.std,
        },
        "alg_score": alg_score,
        "alg_score_formula": "0.6 * peak_return_ci.ci_lower + 0.4 * auc_time_averaged_ci.ci_lower",
    }


def plot_experiment_learning_curves(
    experiment_name: str,
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    grid: np.ndarray,
    band: Dict[str, np.ndarray],
    alg_stats: Dict,
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

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

    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)

    for i, (seed_name, (steps, values)) in enumerate(sorted(curves.items())):
        color = SEED_COLORS[i % len(SEED_COLORS)]
        ax.plot(steps, values, color=color, linewidth=1.1, alpha=0.65,
                 marker="o", markersize=3, label=seed_name)

    if grid.size > 0:
        ax.plot(grid, band["mean"], color=INK_PRIMARY, linewidth=2.4, label="Cross-seed mean")
        ax.fill_between(grid, band["ci_lower"], band["ci_upper"], color=INK_MUTED, alpha=0.25,
                         linewidth=0, label="95% CI (across seeds)")

    peak = alg_stats["peak_return_ci"]
    auc = alg_stats["auc_time_averaged_ci"]
    subtitle = (
        f"Peak Return: {peak['mean']:.2f} {chr(0xB1)} {(peak['ci_upper']-peak['ci_lower'])/2:.2f}  |  "
        f"Time-avg. AUC: {auc['mean']:.2f} {chr(0xB1)} {(auc['ci_upper']-auc['ci_lower'])/2:.2f}  |  "
        f"AlgScore (0.6*Peak_LCB + 0.4*AUC_LCB): {alg_stats['alg_score']:.2f}"
    )

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Evaluation Return")
    # Subtitle sits as the axes title (always directly above the plot area); the bold
    # experiment-name title sits above that as a figure-level suptitle. tight_layout reserves
    # space for both, so unlike two axes-coordinate texts, these can't collide.
    ax.set_title(subtitle, fontsize=10, color=INK_SECONDARY, loc="left", pad=10)
    fig.suptitle(f"{experiment_name} — All Seeds", fontsize=15, fontweight="bold",
                 color=INK_PRIMARY, x=0.01, ha="left", y=1.0)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))
    ax.legend(loc="best", frameon=True, fontsize=8.5, labelcolor=INK_SECONDARY,
               facecolor=SURFACE, edgecolor=GRIDLINE)

    fig.text(0.01, -0.02,
              "AlgScore/Peak_LCB/AUC_LCB are this project's own reconstruction of a reference "
              "chart's metric, not a verified reproduction of any original tool -- see "
              "experiment_statistics.py's module docstring.",
              fontsize=7, color=INK_MUTED, ha="left")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def generate_experiment_analysis(
    experiment_dir: Path, experiment_name: str, out_dir: Path
) -> Optional[Dict]:
    """
    Orchestrates the whole flow: load each seed's eval/mean_reward curve, align + band them,
    compute Peak/AUC/AlgScore, save a figure. Returns the numeric stats dict (JSON-serializable)
    for the caller to fold into its own results JSON, or None if there was nothing to analyze
    (e.g. no TensorBoard event files found -- older/legacy runs, or a run that never trained).
    """
    curves = load_seed_learning_curves(experiment_dir)
    if not curves:
        print("   [Learning-Curve Analysis] No TensorBoard event files with "
              f"'{EVAL_REWARD_TAG}' found under any seed directory -- skipping.")
        return None

    grid, aligned = align_curves_to_common_grid(curves)
    band = compute_cross_seed_band(aligned) if aligned else {"mean": np.array([]), "ci_lower": np.array([]), "ci_upper": np.array([])}
    alg_stats = compute_peak_auc_algscore(curves)

    out_path = out_dir / "learning_curve_analysis.png"
    try:
        plot_experiment_learning_curves(experiment_name, curves, grid, band, alg_stats, out_path)
        print(f"   [Learning-Curve Analysis] Figure saved to: {out_path}")
    except Exception as e:
        print(f"   [Learning-Curve Analysis] Figure generation failed ({e}); "
              f"numeric stats are still included in the results JSON.")

    print(f"   [Learning-Curve Analysis] Peak Return: {alg_stats['peak_return_ci']['mean']:.2f} "
          f"(LCB {alg_stats['peak_return_ci']['ci_lower']:.2f})  |  "
          f"Time-avg AUC: {alg_stats['auc_time_averaged_ci']['mean']:.2f} "
          f"(LCB {alg_stats['auc_time_averaged_ci']['ci_lower']:.2f})  |  "
          f"AlgScore: {alg_stats['alg_score']:.2f}")

    return {
        "n_seeds_with_curves": len(curves),
        "eval_reward_tag": EVAL_REWARD_TAG,
        **alg_stats,
        "figure_path": str(out_path) if out_path.exists() else None,
    }
