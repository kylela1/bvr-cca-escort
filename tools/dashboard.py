#!/usr/bin/env python3
"""
A local, no-cloud-account results dashboard for the B-ACE training pipeline, per
IMPROVEMENT_PLAN_1_VISUALIZATION_AND_RUNNING.md Task 2. Reads only already-existing output
files (`monitor.csv`, `training_results_seed<N>.json`) -- it does not change what
train_bace_clean.py or evaluate_experiment_seeds.py compute or write out, and does not require
a Weights & Biases account (see Task 1 / --use-wandb for that alternative).

Usage:
    streamlit run tools/dashboard.py
"""

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent
RUNS_DIR = REPO_ROOT / "runs_sb3"


def find_runs(runs_dir: Path) -> Dict[str, Path]:
    """
    Scan runs_sb3/ for every directory that directly contains a training_results_seed*.json --
    this is "a run" for this dashboard's purposes. Covers both flat single runs
    (runs_sb3/<run_name>/) and grouped multi-seed experiments
    (runs_sb3/<experiment_name>/<run_name_with_seedN>/), mirroring the directory-walking
    pattern evaluate_experiment_seeds.py::find_seed_models() already uses (glob for a
    'seed'-containing name one level down), without needing to import that module (this
    dashboard only reads training_results JSON, not model checkpoints).
    """
    runs: Dict[str, Path] = {}
    if not runs_dir.exists():
        return runs

    for candidate in sorted(runs_dir.iterdir()):
        if not candidate.is_dir():
            continue
        if list(candidate.glob("training_results_seed*.json")):
            runs[candidate.name] = candidate
            continue
        ##############################
        # Grouped multi-seed experiment: look one level down for seed subdirectories
        ##############################
        for seed_dir in sorted(candidate.glob("*seed*")):
            if seed_dir.is_dir() and list(seed_dir.glob("training_results_seed*.json")):
                runs[f"{candidate.name}/{seed_dir.name}"] = seed_dir

    return runs


def load_training_results(run_dir: Path) -> Optional[Dict[str, Any]]:
    json_files = list(run_dir.glob("training_results_seed*.json"))
    if not json_files:
        return None
    with open(json_files[0]) as f:
        return json.load(f)


def load_monitor_csv(run_dir: Path) -> Optional[pd.DataFrame]:
    monitor_path = run_dir / "monitor.csv"
    if not monitor_path.exists():
        return None
    ##############################
    # SB3's Monitor writes a '#{json header}' comment line before the real CSV header
    ##############################
    df = pd.read_csv(monitor_path, skiprows=1)
    if df.empty:
        return None
    df["cumulative_steps"] = df["l"].cumsum()
    return df


def render_training_curve(selected: Dict[str, Path], rolling_window: int):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    any_plotted = False

    for run_name, run_dir in selected.items():
        df = load_monitor_csv(run_dir)
        if df is None:
            continue
        rolling = df["r"].rolling(window=rolling_window, min_periods=1).mean()
        ax.plot(df["cumulative_steps"], rolling, label=run_name)
        any_plotted = True

    if not any_plotted:
        st.info("No monitor.csv data found for the selected run(s).")
        return

    ax.set_xlabel("Timesteps")
    ax.set_ylabel(f"Episode reward (rolling mean, window={rolling_window})")
    ax.set_title("Training reward vs. timesteps")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)


def render_evaluation_summary(selected: Dict[str, Path]):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    any_plotted = False

    for run_name, run_dir in selected.items():
        results = load_training_results(run_dir)
        if not results:
            continue
        eval_summary = results.get("evaluation_summary") or {}
        timesteps = eval_summary.get("timesteps") or []
        mean_rewards = eval_summary.get("mean_rewards") or []
        std_rewards = eval_summary.get("std_rewards") or []
        if not timesteps or not mean_rewards:
            continue

        timesteps = np.array(timesteps, dtype=float)
        mean_rewards = np.array(mean_rewards, dtype=float)
        std_rewards = np.array(std_rewards, dtype=float) if std_rewards else np.zeros_like(mean_rewards)

        line, = ax.plot(timesteps, mean_rewards, marker="o", label=run_name)
        ax.fill_between(
            timesteps,
            mean_rewards - std_rewards,
            mean_rewards + std_rewards,
            alpha=0.15,
            color=line.get_color(),
        )
        any_plotted = True

    if not any_plotted:
        st.info(
            "No periodic evaluation data found for the selected run(s) -- this is expected "
            "for short/smoke-test runs, since RollbackEvaluationCallbackTopK's periodic "
            "evaluation only fires every `eval_freq` timesteps (default 200,000)."
        )
        return

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Evaluation mean reward (± 1 std)")
    ax.set_title("Periodic evaluation reward vs. timesteps")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)


def render_config_table(selected: Dict[str, Path]):
    for run_name, run_dir in selected.items():
        results = load_training_results(run_dir)
        if not results:
            continue
        config = results.get("config") or {}
        st.markdown(f"**{run_name}**")
        st.dataframe(
            pd.DataFrame(sorted(config.items()), columns=["parameter", "value"]),
            hide_index=True,
            width="stretch",
        )


def main():
    st.set_page_config(page_title="B-ACE Training Dashboard", layout="wide")
    st.title("B-ACE Training Dashboard")
    st.caption(
        "Reads existing runs_sb3/ output files only (monitor.csv, training_results_seed*.json) "
        "-- does not affect training. For per-step PPO diagnostics (policy loss, entropy, clip "
        "fraction), use TensorBoard alongside this dashboard (see the sidebar)."
    )

    runs = find_runs(RUNS_DIR)

    if not runs:
        st.warning(f"No completed runs found under `{RUNS_DIR}`. Run some training first (see HOW_TO_RUN.md).")
        st.stop()

    with st.sidebar:
        st.header("Runs")
        run_names = st.multiselect("Select run(s) to compare", options=sorted(runs.keys()), default=sorted(runs.keys())[:1])
        rolling_window = st.slider("Training curve rolling-mean window (episodes)", min_value=1, max_value=200, value=20)

        st.header("TensorBoard")
        st.code(f"tensorboard --logdir {RUNS_DIR}", language="powershell")
        st.caption("Run this in a separate terminal for training-internals diagnostics (loss, entropy, clip fraction).")

    if not run_names:
        st.info("Select one or more runs from the sidebar to see charts.")
        st.stop()

    selected = {name: runs[name] for name in run_names}

    st.header("Training curve")
    render_training_curve(selected, rolling_window)

    st.header("Evaluation summary")
    render_evaluation_summary(selected)

    st.header("Run configuration")
    render_config_table(selected)


if __name__ == "__main__":
    main()
