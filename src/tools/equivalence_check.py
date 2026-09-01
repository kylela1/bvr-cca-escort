#!/usr/bin/env python3
"""
Equivalence-check tool for structural refactors of the environment-construction path
(see IMPROVEMENT_PLAN_2_WRAPPERS_AND_DEPENDENCIES.md, Part A, Task 1).

This does NOT run real training. It intercepts train_bace_clean.py's `main()` at the point
where it builds `DummyVecEnv([make_env(...)])` (once for the training env, once for the eval
env), and records two kinds of things about the constructed environment:

  1. STRUCTURE (the primary, pass/fail signal): the ordered chain of wrapper class names from
     outermost to innermost, each wrapper's JSON-serializable constructor-derived attributes
     (e.g. "every_n_steps": 25, "bonus": 0.01), the observation space shape/dtype, and the
     action space shape/bounds. This is fully deterministic and is exactly what a structural
     refactor (Task 2/3 of Improvement Plan 2) needs to preserve: the same wrapper classes, in
     the same order, with the same configuration, for the same CLI flags.

  2. TRAJECTORY (informational only, never gates pass/fail): observations/rewards/info from a
     fixed, seeded action sequence starting from a fresh reset. Kept for human inspection, but
     see the KNOWN LIMITATION below for why it cannot be used as a strict equality gate.

KNOWN LIMITATION, discovered empirically while building this tool: Godot's own spawn-position
randomization was observed to differ across separate runs of *identical, unmodified* code,
including the train_env path (not just eval_env). None of env.reset(seed=...), the numpy RNG
seed used for this tool's synthetic action sequence, or train_bace_clean.py's own --seed CLI
flag control this -- in fact --seed is never written into the Godot-side EnvConfig "seed"
field at all (see IMPROVEMENT_PLAN_2_WRAPPERS_AND_DEPENDENCIES.md). Because of this, per-value
trajectory diffing produces hundreds of "differences" between two runs of the exact same code,
which would make it useless as a regression gate. Structural comparison (point 1 above) does
not have this problem, since it doesn't depend on simulated physics/spawn outcomes at all.

Usage:
    # Capture a baseline snapshot of current structure + a trajectory sample:
    python src/tools/equivalence_check.py capture config/equivalence_baselines/default.json --total-timesteps 100000

    # After a refactor, compare current structure against a previously captured baseline:
    python src/tools/equivalence_check.py compare config/equivalence_baselines/default.json --total-timesteps 100000

Any arguments after the snapshot path are forwarded verbatim to train_bace_clean.py's own
argument parser (e.g. --use-enriched-obs --ablation-config all --expert-alignment), so you can
capture/compare multiple configurations. A --run-name is auto-generated and its results/runs_sb3/
output directory is deleted afterward -- this tool only cares about the captured JSON.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RL_FILES = REPO_ROOT / "src" / "rl_files"
if RL_FILES.as_posix() not in sys.path:
    sys.path.insert(0, RL_FILES.as_posix())
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SRC_ROOT.as_posix())

SEED = 12345
N_STEPS = 30

# Attribute names never worth comparing even though they happen to be JSON-serializable
# (timestamps, file paths, RNG state that legitimately varies run to run, etc.)
_ATTR_BLOCKLIST = {
    "np_random", "_np_random", "start_time", "t_start", "log_dir", "eval_checkpoint_dir",
    "best_checkpoint_dir", "port", "_seed", "seed", "device",
}


class _CaptureAndStop(Exception):
    pass


def _wrapper_attrs(w) -> dict:
    """Best-effort JSON-serializable snapshot of a single wrapper's own configuration."""
    out = {}
    for k, v in vars(w).items():
        if k.startswith("_") or k in _ATTR_BLOCKLIST:
            continue
        if k in ("env",):  # the wrapped inner env itself -- walked separately
            continue
        if isinstance(v, (int, float, bool, str)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (int, float, bool, str)) for x in v):
            out[k] = list(v)
        elif isinstance(v, np.ndarray) and v.size <= 16:
            out[k] = np.asarray(v, dtype=float).round(6).tolist()
    return out


def _describe_stack(env) -> list[dict]:
    """Walk the .env chain from outermost to innermost, describing each layer."""
    chain = []
    cur = env
    while cur is not None:
        chain.append({"class": type(cur).__name__, "attrs": _wrapper_attrs(cur)})
        cur = getattr(cur, "env", None)
    return chain


def _describe_space(space) -> dict:
    out = {"class": type(space).__name__}
    if hasattr(space, "shape"):
        out["shape"] = list(space.shape) if space.shape is not None else None
    if hasattr(space, "dtype"):
        out["dtype"] = str(space.dtype)
    if hasattr(space, "low") and getattr(space, "low", None) is not None:
        try:
            out["low"] = np.asarray(space.low, dtype=float).round(3).tolist()
            out["high"] = np.asarray(space.high, dtype=float).round(3).tolist()
        except Exception:
            pass
    return out


def _jsonify_info(info: dict) -> dict:
    out = {}
    for k, v in info.items():
        if isinstance(v, (int, float, bool, str)) or v is None:
            out[k] = v
        elif isinstance(v, np.generic):
            out[k] = v.item()
    return out


def capture(run_name: str, extra_args: list[str]) -> dict:
    argv = [
        "train_bace_clean.py",
        "--total-timesteps", "100000",  # never actually reached; overridden if passed in extra_args
        "--run-name", run_name,
    ] + extra_args
    sys.argv = argv

    import importlib
    if "train_bace_clean" in sys.modules:
        importlib.reload(sys.modules["train_bace_clean"])
    import train_bace_clean as tbc

    captured: dict = {}
    state = {"n_calls": 0}
    orig_dummy_vec_env = tbc.DummyVecEnv

    def patched_dummy_vec_env(env_fns):
        vec = orig_dummy_vec_env(env_fns)
        state["n_calls"] += 1
        label = "train_env" if state["n_calls"] == 1 else "eval_env"

        env = vec.envs[0]

        structure = {
            "wrapper_chain": _describe_stack(env),
            "observation_space": _describe_space(env.observation_space),
            "action_space": _describe_space(env.action_space),
        }

        rng = np.random.RandomState(SEED)
        obs, info = env.reset(seed=SEED)
        trajectory = {
            "reset_obs_shape": list(np.asarray(obs).shape),
            "steps": [],
        }
        for i in range(N_STEPS):
            action = rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
            obs, reward, terminated, truncated, step_info = env.step(action)
            trajectory["steps"].append(
                {
                    "reward": round(float(reward), 6),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info_keys": sorted(_jsonify_info(step_info).keys()),
                }
            )
            if terminated or truncated:
                obs, _ = env.reset(seed=SEED + i + 1)

        captured[label] = {"structure": structure, "trajectory": trajectory}

        if state["n_calls"] >= 2:
            raise _CaptureAndStop()
        return vec

    tbc.DummyVecEnv = patched_dummy_vec_env
    try:
        tbc.main()
    except _CaptureAndStop:
        pass
    finally:
        tbc.DummyVecEnv = orig_dummy_vec_env

    if "train_env" not in captured:
        raise RuntimeError(
            "Never observed a DummyVecEnv construction call -- train_bace_clean.py's "
            "construction path may have changed shape; update this tool's interception point."
        )
    return captured


def _diff(baseline, current, path: str = "") -> list[str]:
    diffs = []
    if isinstance(baseline, dict) and isinstance(current, dict):
        for k in sorted(set(baseline) | set(current)):
            if k not in baseline:
                diffs.append(f"{path}.{k}: missing in baseline, present in current")
            elif k not in current:
                diffs.append(f"{path}.{k}: present in baseline, missing in current")
            else:
                diffs.extend(_diff(baseline[k], current[k], f"{path}.{k}"))
    elif isinstance(baseline, list) and isinstance(current, list):
        if len(baseline) != len(current):
            diffs.append(f"{path}: length {len(baseline)} -> {len(current)}")
        else:
            for i, (b, c) in enumerate(zip(baseline, current)):
                diffs.extend(_diff(b, c, f"{path}[{i}]"))
    elif isinstance(baseline, float) and isinstance(current, float):
        if abs(baseline - current) > 1e-4:
            diffs.append(f"{path}: {baseline} -> {current} (delta {current - baseline:.6g})")
    else:
        if baseline != current:
            diffs.append(f"{path}: {baseline!r} -> {current!r}")
    return diffs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["capture", "compare"])
    parser.add_argument("snapshot_path", type=str)
    args, extra_args = parser.parse_known_args()

    snapshot_path = Path(args.snapshot_path)
    run_name = "equivcheck_scratch"

    print("[equivalence_check] Building environments (this launches real Godot instances)...")
    result = capture(run_name, extra_args)

    if args.mode == "capture":
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        with open(snapshot_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[equivalence_check] Baseline captured -> {snapshot_path}")
    else:
        if not snapshot_path.exists():
            print(f"[equivalence_check] ERROR: no baseline found at {snapshot_path}. Run 'capture' first.")
            sys.exit(2)
        with open(snapshot_path) as f:
            baseline = json.load(f)

        structure_diffs = []
        trajectory_diffs = []
        for label in ("train_env", "eval_env"):
            b = baseline.get(label, {})
            c = result.get(label, {})
            structure_diffs.extend(_diff(b.get("structure"), c.get("structure"), f".{label}.structure"))
            trajectory_diffs.extend(_diff(b.get("trajectory"), c.get("trajectory"), f".{label}.trajectory"))

        if trajectory_diffs:
            print(f"[equivalence_check] INFO (non-fatal, expected to vary -- see KNOWN LIMITATION): "
                  f"{len(trajectory_diffs)} trajectory difference(s).")

        if structure_diffs:
            print(f"[equivalence_check] FAIL: {len(structure_diffs)} structural difference(s) found vs. {snapshot_path}:")
            for d in structure_diffs[:80]:
                print(f"    {d}")
            if len(structure_diffs) > 80:
                print(f"    ... and {len(structure_diffs) - 80} more")
            sys.exit(1)
        else:
            print(f"[equivalence_check] PASS: wrapper-stack structure matches {snapshot_path} exactly "
                  f"(same classes, same order, same configuration, same spaces).")

    scratch_dir = REPO_ROOT / "results" / "runs_sb3" / run_name
    if scratch_dir.exists():
        import shutil
        shutil.rmtree(scratch_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
