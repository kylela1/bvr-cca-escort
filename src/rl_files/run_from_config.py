#!/usr/bin/env python3
"""
This script launches a training run from a declarative YAML config file instead of a
hand-typed command line. Its only job is "YAML keys -> CLI flags" -- it builds and then shells
out to a real `python train_bace_clean.py ...` (or `run_experiment.py ...` for multi-seed
configs) invocation via subprocess, so a YAML-launched run can never silently diverge from what
the equivalent hand-written CLI command would do (see
IMPROVEMENT_PLAN_1_VISUALIZATION_AND_RUNNING.md Task 3).

Every key in a config YAML file must equal the `dest` name of a real train_bace_clean.py
argparse flag (e.g. `--total-timesteps` -> `total_timesteps`, `--use-enriched-obs` ->
`use_enriched_obs`) -- run `python src/rl_files/train_bace_clean.py --help` to see the full flag
list, or read train_bace_clean.py's _build_arg_parser(). An unrecognized key is a hard error,
not a silent no-op, so a typo'd or renamed flag can't quietly do nothing.

Usage:
    python src/rl_files/run_from_config.py config/launch/baseline.yaml
    python src/rl_files/run_from_config.py config/launch/factor_BD.yaml --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent.parent
for p in (REPO_ROOT, THIS_FILE.parent):
    if p.as_posix() not in sys.path:
        sys.path.insert(0, p.as_posix())

import train_bace_clean as tbc

##############################
# Keys a config YAML may contain that are NOT train_bace_clean.py flags -- consumed by this
# launcher itself instead of being forwarded.
##############################
LAUNCHER_ONLY_KEYS = {"seeds"}


def _actions_by_dest() -> Dict[str, List[argparse.Action]]:
    """dest name -> list of argparse actions sharing that dest (usually 1, or 2 for
    --enable-X/--disable-X pairs)."""
    parser = tbc._build_arg_parser()
    by_dest: Dict[str, List[argparse.Action]] = {}
    for action in parser._actions:
        if not action.option_strings or action.dest == "help":
            continue  # positional args / argparse's own -h/--help, not relevant here
        by_dest.setdefault(action.dest, []).append(action)
    return by_dest


def config_to_cli_args(config: Dict[str, Any]) -> List[str]:
    """
    Convert a YAML config dict into the equivalent train_bace_clean.py CLI argument list.
    Raises ValueError with a clear message for any key that isn't a real flag's dest.
    """
    by_dest = _actions_by_dest()
    args: List[str] = []

    for key, value in config.items():
        if key in LAUNCHER_ONLY_KEYS:
            continue

        if key not in by_dest:
            valid = ", ".join(sorted(by_dest.keys()))
            raise ValueError(
                f"Unknown config key '{key}' -- no train_bace_clean.py flag has this dest name. "
                f"Config keys must match a flag's argparse dest exactly "
                f"(e.g. --total-timesteps -> 'total_timesteps'). Valid keys: {valid}"
            )

        matching_actions = by_dest[key]

        ##############################
        # Paired --enable-X / --disable-X flags (both store_true and store_false on one dest)
        ##############################
        if len(matching_actions) == 2:
            true_action = next(a for a in matching_actions if isinstance(a, argparse._StoreTrueAction))
            false_action = next(a for a in matching_actions if isinstance(a, argparse._StoreFalseAction))
            args.append(true_action.option_strings[0] if value else false_action.option_strings[0])
            continue

        action = matching_actions[0]

        ##############################
        # Single boolean flag (store_true or store_false) -- only emit it when the requested
        # value differs from the flag's default, since these flags take no value.
        ##############################
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            is_store_true = isinstance(action, argparse._StoreTrueAction)
            want_true = bool(value)
            if want_true == bool(action.default):
                continue
            if want_true != is_store_true:
                raise ValueError(
                    f"Config key '{key}'={value!r} isn't reachable via the single available "
                    f"flag {action.option_strings[0]} (default={action.default})."
                )
            args.append(action.option_strings[0])
            continue

        ##############################
        # Value-taking flag
        ##############################
        args.append(action.option_strings[0])
        args.append(str(value))

    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config_path", type=str, help="Path to a YAML run config (see config/launch/*.yaml for examples).")
    parser.add_argument("--dry-run", action="store_true", help="Print the command that would run, without running it.")
    args = parser.parse_args()

    config_path = Path(args.config_path).expanduser().resolve()
    if not config_path.exists():
        print(f"ERROR: config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    try:
        cli_args = config_to_cli_args(config)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    seeds = config.get("seeds")

    ##############################
    # Multi-seed configs delegate to run_experiment.py (which already loops seeds and forwards
    # unrecognized args to train_bace_clean.py) rather than reimplementing the seed loop here.
    ##############################
    if seeds:
        if "experiment_name" not in config:
            print("ERROR: multi-seed configs (with a 'seeds' key) must also set 'experiment_name'.")
            sys.exit(1)
        run_experiment_args = list(cli_args)
        ##############################
        # experiment_name and seed are handled by run_experiment.py itself (--experiment-name
        # directly, --seed assigned per-seed internally) -- strip them from the forwarded args
        # so they aren't passed through twice / conflictingly.
        ##############################
        for flag in ("--experiment-name", "--seed"):
            if flag in run_experiment_args:
                idx = run_experiment_args.index(flag)
                del run_experiment_args[idx:idx + 2]

        cmd = [
            sys.executable,
            str(REPO_ROOT / "src" / "rl_files" / "run_experiment.py"),
            "--experiment-name", str(config["experiment_name"]),
            "--seeds", *[str(s) for s in seeds],
        ] + run_experiment_args
    else:
        cmd = [sys.executable, str(REPO_ROOT / "src" / "rl_files" / "train_bace_clean.py")] + cli_args

    print(f"[run_from_config] Config: {config_path}")
    print(f"[run_from_config] Command: {' '.join(cmd)}")

    if args.dry_run:
        print("[run_from_config] --dry-run: not executing.")
        return

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
