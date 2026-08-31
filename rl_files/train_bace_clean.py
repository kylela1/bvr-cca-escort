#!/usr/bin/env python3
"""
This is the core training script for the B-ACE reinforcement learning pipeline. It
configures and trains a PPO agent using Stable-Baselines3 to control a blue (friendly)
fighter aircraft in a Godot-based air combat simulation. The script builds a layered
environment wrapper stack that includes enriched observations, expert alignment or blending,
reward shaping (HVAA destruction penalty, sustained turn penalty, mission tempo, track-aware
range shaping), and reward logging. It also implements a custom PPO subclass with
fire-action gating and cooldown logic, decaying learning rate and entropy schedules, and
evaluation-based checkpointing with automatic rollback on performance degradation.

Within the broader pipeline, this script is invoked by run_experiment.py once per seed. It
reads a B-ACE JSON scenario config, instantiates the Godot environment via
B_ACE_GodotPettingZooWrapper, wraps it in a SingleAgentBACEEnv for SB3 compatibility, then
layers on the configurable wrapper stack. Training progress, model checkpoints, and
VecNormalize statistics are saved to a per-seed directory under runs_sb3/EXPERIMENT_NAME/.
The saved artifacts are later loaded by evaluate_experiment_seeds.py for evaluation.

Key configurable parameters (all via command-line arguments): --total-timesteps (default:
1,000,000), --initial-learning-rate (0.000258), --final-learning-rate (0.00001),
--initial-entropy-coef (0.051), --final-entropy-coef (0.005), --decay-schedule (linear or
exponential), --n_layers (2), --layer_size (512), --gamma (0.99), --gae-lambda (0.9207),
--clip-eps (0.198), --vf-coef (0.773), --n-steps (6144), --batch-size (1024), --epochs (8).
Enriched observations are enabled with --use-enriched-obs and configured via
--ablation-config (all, none, geometry_only, engagement_only, range_limited_only, etc.).
Expert integration is enabled with --expert-alignment or --expert-blending, each with their
own sub-parameters for coefficients and decay schedules.
"""

import argparse
import atexit
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Tuple, Callable, List
import math

import numpy as np
import torch

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

##############################
# Find B-ACE repository root and add to path
##############################
THIS_FILE = Path(__file__).resolve()

##############################
# Search upward for b_ace_py directory to find repo root
##############################
current_path = THIS_FILE.parent
while current_path != current_path.parent:
    if (current_path / "b_ace_py").exists():
        REPO_ROOT = current_path
        break
    current_path = current_path.parent
else:
    REPO_ROOT = THIS_FILE.parent.parent

if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

print(f"Using REPO_ROOT: {REPO_ROOT}")

from b_ace_py.utils import load_b_ace_config
from rollback_eval_callback import RollbackEvaluationCallbackTopK, SkippableTrainPPO
from env_factory import build_env, get_split_norm_wrapper_from_vec_env

try:
    from reward_component_eval_callback import RewardComponentEvalCallback
    COMPONENT_TRACKING_AVAILABLE = True
except ImportError:
    COMPONENT_TRACKING_AVAILABLE = False
    print("Note: RewardComponentEvalCallback not found. Using standard evaluation.")

def print_wrapper_stack(env, name: str = "env"):
    """
    Print wrapper chain for:
      - VecEnv (DummyVecEnv/VecNormalize)
      - underlying Gym env wrappers (Monitor, NanGuardWrapper, etc.)
    """
    print("\n" + "=" * 80)
    print(f"[WRAPPER STACK] {name}")
    print("=" * 80)

    e = env
    i = 0
    ##############################
    # Traverse VecEnv wrapper stack (VecNormalize, etc.)
    ##############################
    while True:
        print(f"  VecLayer {i}: {type(e)}")
        ##############################
        # VecNormalize wraps another VecEnv at .venv attribute
        ##############################
        if hasattr(e, "venv"):
            e = e.venv
            i += 1
            continue
        break

    ##############################
    # Print underlying gym env chain if using DummyVecEnv
    ##############################
    base = None
    if hasattr(e, "envs") and len(e.envs) > 0:
        base = e.envs[0]
        print("\n  Underlying gym wrapper chain (envs[0]):")
        j = 0
        cur = base
        ##############################
        # Traverse gym wrapper stack
        ##############################
        while True:
            print(f"    GymLayer {j}: {type(cur)}")
            if hasattr(cur, "env"):
                cur = cur.env
                j += 1
                continue
            break

    print("=" * 80 + "\n")


def linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    """
    Linear learning rate or entropy schedule.

    Args:
        initial_value: Starting value
        final_value: Ending value

    Returns:
        A function that takes progress (0 to 1) and returns the interpolated value
    """
    def schedule(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0 (end).
        We want to go from initial_value to final_value.
        """
        return final_value + progress_remaining * (initial_value - final_value)

    return schedule


def exponential_schedule(initial_value: float, final_value: float, decay_rate: float = 0.1) -> Callable[[float], float]:
    """
    Exponential decay schedule (alternative to linear).

    Args:
        initial_value: Starting value
        final_value: Ending value
        decay_rate: Controls how fast the decay happens

    Returns:
        A function that takes progress (0 to 1) and returns the exponentially decayed value
    """
    def schedule(progress_remaining: float) -> float:
        progress_made = 1 - progress_remaining
        decay_factor = np.exp(-decay_rate * progress_made * 10)
        return final_value + (initial_value - final_value) * decay_factor

    return schedule


class MetricsCollectionCallback(BaseCallback):
    """
    Enhanced callback to:
    1. Update entropy coefficient during training
    2. Collect episode metrics for CI / learning curves (memory-efficient)
    3. Log schedules (lr, clip, entropy coef) only every metrics_collection_interval steps
    """
    def __init__(
        self,
        entropy_schedule=None,
        metrics_collection_interval: int = 1000,
        max_stored_episodes: int = 10000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.entropy_schedule = entropy_schedule
        self.metrics_collection_interval = metrics_collection_interval
        self.max_stored_episodes = max_stored_episodes

        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps_history = []
        self.all_episode_rewards = []

        self.rolling_window = 100
        self.recent_rewards = []

    def _on_step(self) -> bool:
        ##############################
        # Update global step for scheduled wrappers
        ##############################
        try:
            self.model.get_env().env_method("set_global_step", int(self.num_timesteps))
        except Exception:
            pass

        ##############################
        # Update schedules every step
        ##############################
        progress_remaining = 1.0 - (self.num_timesteps / self.model._total_timesteps)

        ##############################
        # Update entropy coefficient from schedule
        ##############################
        if self.entropy_schedule is not None:
            new_ent_coef = self.entropy_schedule(progress_remaining)
            self.model.ent_coef = new_ent_coef

        ##############################
        # Log schedules periodically
        ##############################
        if self.num_timesteps % self.metrics_collection_interval == 0:
            ##############################
            # Log entropy coefficient
            ##############################
            if self.entropy_schedule is not None:
                self.logger.record("train/entropy_coef", float(self.model.ent_coef))

            ##############################
            # Log learning rate if scheduled
            ##############################
            if hasattr(self.model, "learning_rate"):
                lr = self.model.learning_rate
                if callable(lr):
                    current_lr = lr(progress_remaining)
                else:
                    current_lr = lr
                self.logger.record("train/learning_rate", float(current_lr))

            ##############################
            # Log clip range if scheduled
            ##############################
            if hasattr(self.model, "clip_range"):
                clip = self.model.clip_range
                if callable(clip):
                    current_clip = clip(progress_remaining)
                else:
                    current_clip = clip
                self.logger.record("train/clip_range", float(current_clip))

        ##############################
        # Collect episode metrics from this step
        ##############################
        infos = self.locals.get("infos", None)
        if infos is not None:
            for info in infos:
                ep_info = info.get("episode")
                ##############################
                # Skip if no episode info in this step
                ##############################
                if ep_info is None:
                    continue

                reward = ep_info.get("r", None)
                length = ep_info.get("l", None)
                if reward is None or length is None:
                    continue

                ##############################
                # Track all rewards for final statistics
                ##############################
                self.all_episode_rewards.append(reward)

                self.episode_rewards.append(reward)
                self.episode_lengths.append(length)
                self.timesteps_history.append(self.num_timesteps)

                ##############################
                # Downsample if we exceed storage limit
                ##############################
                if len(self.episode_rewards) > self.max_stored_episodes * 1.1:
                    current_size = len(self.episode_rewards)
                    indices = np.linspace(
                        0, current_size - 1, self.max_stored_episodes, dtype=int
                    )

                    self.episode_rewards = [self.episode_rewards[i] for i in indices]
                    self.episode_lengths = [self.episode_lengths[i] for i in indices]
                    self.timesteps_history = [self.timesteps_history[i] for i in indices]

                ##############################
                # Update rolling window for reward stats
                ##############################
                self.recent_rewards.append(reward)
                if len(self.recent_rewards) > self.rolling_window:
                    self.recent_rewards.pop(0)

                ##############################
                # Log rolling statistics
                ##############################
                if len(self.recent_rewards) >= 10:
                    self.logger.record(
                        "rollout/mean_reward_100ep", np.mean(self.recent_rewards)
                    )
                    self.logger.record(
                        "rollout/std_reward_100ep", np.std(self.recent_rewards)
                    )

        return True

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for the entire training run.
        Uses all_episode_rewards for accurate statistics, but only saves limited episode data.
        """
        if len(self.all_episode_rewards) == 0:
            return {
                "mean_reward": 0.0,
                "std_reward": 0.0,
                "mean_episode_length": 0.0,
                "total_episodes": 0,
                "episode_rewards": [],
                "episode_lengths": [],
                "timesteps": [],
            }

        all_rewards_array = np.array(self.all_episode_rewards)

        return {
            "mean_reward": float(np.mean(all_rewards_array)),
            "std_reward": float(np.std(all_rewards_array)),
            "median_reward": float(np.median(all_rewards_array)),
            "min_reward": float(np.min(all_rewards_array)),
            "max_reward": float(np.max(all_rewards_array)),
            "total_episodes": len(self.all_episode_rewards),
            "mean_episode_length": float(np.mean(self.episode_lengths))
            if self.episode_lengths
            else 0.0,
            "episode_rewards": [float(r) for r in self.episode_rewards],
            "episode_lengths": [int(l) for l in self.episode_lengths],
            "timesteps": [int(t) for t in self.timesteps_history],
            "note": (
                f"Learning curve data downsampled to {len(self.episode_rewards)} "
                f"points from {len(self.all_episode_rewards)} total episodes for file size"
            ),
        }


class EvaluationBasedCheckpointCallback(BaseCallback):
    """
    Callback for saving the best model based on ACTUAL EVALUATION performance.

    Every eval_freq timesteps:
    1. Pauses training
    2. Runs n_eval_episodes in deterministic mode
    3. Computes mean reward
    4. Saves checkpoint if it's the best so far

    This is more reliable than using training episode rewards because:
    - Evaluation uses deterministic actions (no exploration noise)
    - Prevents catastrophic forgetting from being saved
    - Standard practice in RL research
    """
    def __init__(
        self,
        eval_env: DummyVecEnv,
        eval_freq: int = 75000,
        n_eval_episodes: int = 15,
        log_dir: Path = None,
        verbose: int = 1,
        deterministic: bool = True,
        skip_initial_eval: bool = True,
        use_enriched_obs: bool = False,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_dir = Path(log_dir) if log_dir else Path(".")
        self.deterministic = deterministic
        self.skip_initial_eval = skip_initial_eval
        self.use_enriched_obs = use_enriched_obs

        self.best_mean_reward = -np.inf
        self.evaluations_timesteps = []
        self.evaluations_results = []
        self.evaluations_length = []
        self.evaluations_std = []

        self.eval_checkpoint_dir = self.log_dir / "eval_checkpoints"
        self.eval_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.last_eval_step = 0

    def _evaluate_policy(self) -> Tuple[float, float, List[float]]:
        """
        Run n_eval_episodes and return mean reward, std reward, and episode rewards.
        """
        episode_rewards = []
        episode_lengths = []

        obs = self.eval_env.reset()
        episode_reward = 0.0
        episode_length = 0
        episodes_completed = 0


        ##############################
        # Run evaluation episodes until we complete the requested number
        ##############################
        while episodes_completed < self.n_eval_episodes:
            self.eval_env.env_method("set_global_step", int(self.num_timesteps))
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            try:
                ##############################
                # Get original observation for gating if VecNormalize is used
                ##############################
                if hasattr(self.eval_env, "get_original_obs"):
                    obs_for_gate = self.eval_env.get_original_obs()
                    if obs_for_gate is None:
                        obs_for_gate = obs
                else:
                    obs_for_gate = obs

                gated = self.eval_env.env_method("gate_action", action[0], obs_for_gate[0], indices=[0])[0]
                action = np.asarray([gated], dtype=np.float32)
            except Exception as ex:
                pass
            step_out = self.eval_env.step(action)

            ##############################
            # Handle different return formats from VecEnv
            ##############################
            if len(step_out) == 4:
                obs, reward, done, infos = step_out
                terminated, truncated = done, np.array([False] * len(done))
            else:
                obs, reward, terminated, truncated, infos = step_out
                done = np.logical_or(terminated, truncated)

            episode_reward += reward[0]
            episode_length += 1

            ##############################
            # Check if episode has ended
            ##############################
            if done[0]:
                info0 = infos[0] if isinstance(infos, (list, tuple)) else infos

                ep = info0.get("episode", None)

                print(f"[EVAL EP-END] ep={episodes_completed+1}/{self.n_eval_episodes} episode={ep}")

                episode_rewards.append(float(episode_reward))
                episode_lengths.append(int(episode_length))
                episodes_completed += 1

                episode_reward = 0.0
                episode_length = 0

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)

        return mean_reward, std_reward, mean_length, episode_rewards

    def _on_training_start(self) -> None:
        """
        Run an initial evaluation at timestep 0 so the learning curve starts at 0.
        """
        if self.skip_initial_eval:
            if self.verbose > 0:
                print(f"\n{'='*60}")
                print(f"SKIPPING INITIAL EVALUATION (skip_initial_eval=True)")
                print(f"{'='*60}\n")
            return

        if self.verbose > 0:
            print(f"\n{'='*60}")
            print(f"INITIAL EVALUATION at 0 steps")
            print(f"   Running {self.n_eval_episodes} episodes...")
            print(f"{'='*60}")

        eval_start_time = time.time()
        mean_reward, std_reward, mean_length, episode_rewards = self._evaluate_policy()
        eval_time = time.time() - eval_start_time

        self.evaluations_timesteps.append(0)
        self.evaluations_results.append(mean_reward)
        self.evaluations_length.append(mean_length)
        self.evaluations_std.append(std_reward)

        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/std_reward", std_reward)
        self.logger.record("eval/mean_ep_length", mean_length)

        if self.verbose > 0:
            print(f" Initial Evaluation Results:")
            print(f"   Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
            print(f"   Mean Length: {mean_length:.1f}")
            print(f"   Time: {eval_time:.1f}s")

        checkpoint_path = self.eval_checkpoint_dir / "model_0_steps"
        self.model.save(checkpoint_path.as_posix())

        vec_env = self.model.get_env()
        if isinstance(vec_env, VecNormalize):
            norm_path = self.eval_checkpoint_dir / "vecnormalize_0_steps.pkl"
            vec_env.save(norm_path.as_posix())

        ##############################
        # Save SplitNormalizationWrapper stats if using enriched observations
        ##############################
        if self.use_enriched_obs:
            split_norm = get_split_norm_wrapper_from_vec_env(vec_env)
            if split_norm is not None:
                split_norm_path = self.eval_checkpoint_dir / "split_normalization_0_steps.npz"
                split_norm.save(split_norm_path.as_posix())
                if self.verbose > 0:
                    print(f" SplitNormalizationWrapper stats saved (count={split_norm.count})")

        if self.verbose > 0:
            print(" Initial checkpoint saved")

        ##############################
        # Track best model
        ##############################
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward

            best_model_path = self.log_dir / "best_model"
            self.model.save(best_model_path.as_posix())

            if isinstance(vec_env, VecNormalize):
                best_norm_path = self.log_dir / "best_vecnormalize.pkl"
                vec_env.save(best_norm_path.as_posix())

            ##############################
            # Save SplitNormalizationWrapper stats for best model
            ##############################
            if self.use_enriched_obs:
                split_norm = get_split_norm_wrapper_from_vec_env(vec_env)
                if split_norm is not None:
                    best_split_norm_path = self.log_dir / "best_split_normalization.npz"
                    split_norm.save(best_split_norm_path.as_posix())

            best_info = {
                "timestep": 0,
                "mean_reward": float(mean_reward),
                "std_reward": float(std_reward),
                "mean_length": float(mean_length),
                "episode_rewards": [float(r) for r in episode_rewards],
            }
            best_info_path = self.log_dir / "best_model_info.json"
            with open(best_info_path, "w") as f:
                json.dump(best_info, f, indent=2)

            if self.verbose > 0:
                print(f" INITIAL BEST MODEL saved at timestep 0")
                print(f"{'='*60}\n")


    def _on_step(self) -> bool:
        ##############################
        # Check if it's time to evaluate
        ##############################
        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = self.num_timesteps

            if self.verbose > 0:
                print(f"\n{'='*60}")
                print(f"EVALUATION at {self.num_timesteps:,} steps")
                print(f"   Running {self.n_eval_episodes} episodes...")
                print(f"{'='*60}")

            eval_start_time = time.time()
            ##############################
            # Sync normalization stats from training to evaluation env
            ##############################
            try:
                train_env = self.model.get_env()
                if isinstance(train_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
                    self.eval_env.obs_rms = train_env.obs_rms
                    self.eval_env.ret_rms = train_env.ret_rms
            except Exception:
                pass
            mean_reward, std_reward, mean_length, episode_rewards = self._evaluate_policy()
            eval_time = time.time() - eval_start_time

            self.evaluations_timesteps.append(self.num_timesteps)
            self.evaluations_results.append(mean_reward)
            self.evaluations_length.append(mean_length)
            self.evaluations_std.append(std_reward)

            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/std_reward", std_reward)
            self.logger.record("eval/mean_ep_length", mean_length)

            if self.verbose > 0:
                print(f" Evaluation Results:")
                print(f"   Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
                print(f"   Mean Length: {mean_length:.1f}")
                print(f"   Time: {eval_time:.1f}s")

            checkpoint_path = self.eval_checkpoint_dir / f"model_{self.num_timesteps}_steps"
            self.model.save(checkpoint_path.as_posix())

            vec_env = self.model.get_env()
            if isinstance(vec_env, VecNormalize):
                norm_path = self.eval_checkpoint_dir / f"vecnormalize_{self.num_timesteps}_steps.pkl"
                vec_env.save(norm_path.as_posix())

            ##############################
            # Save SplitNormalizationWrapper stats for checkpoint
            ##############################
            if self.use_enriched_obs:
                split_norm = get_split_norm_wrapper_from_vec_env(vec_env)
                if split_norm is not None:
                    split_norm_path = self.eval_checkpoint_dir / f"split_normalization_{self.num_timesteps}_steps.npz"
                    split_norm.save(split_norm_path.as_posix())

            if self.verbose > 0:
                print(f"Checkpoint saved")

            ##############################
            # Check if this is the best model so far
            ##############################
            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    improvement = mean_reward - self.best_mean_reward
                    print(f"NEW BEST MODEL! (+{improvement:.2f} improvement)")

                self.best_mean_reward = mean_reward

                best_model_path = self.log_dir / "best_model"
                self.model.save(best_model_path.as_posix())

                if isinstance(vec_env, VecNormalize):
                    best_norm_path = self.log_dir / "best_vecnormalize.pkl"
                    vec_env.save(best_norm_path.as_posix())

                ##############################
                # Save SplitNormalizationWrapper stats for best model
                ##############################
                if self.use_enriched_obs:
                    split_norm = get_split_norm_wrapper_from_vec_env(vec_env)
                    if split_norm is not None:
                        best_split_norm_path = self.log_dir / "best_split_normalization.npz"
                        split_norm.save(best_split_norm_path.as_posix())
                        if self.verbose > 0:
                            print(f"   SplitNormalization stats saved (count={split_norm.count})")

                best_info = {
                    'timestep': self.num_timesteps,
                    'mean_reward': float(mean_reward),
                    'std_reward': float(std_reward),
                    'mean_length': float(mean_length),
                    'episode_rewards': [float(r) for r in episode_rewards],
                }

                best_info_path = self.log_dir / "best_model_info.json"
                with open(best_info_path, 'w') as f:
                    json.dump(best_info, f, indent=2)

                if self.verbose > 0:
                    print(f"   Saved to: {best_model_path}")
            else:
                if self.verbose > 0:
                    deficit = self.best_mean_reward - mean_reward
                    print(f"   (Best: {self.best_mean_reward:.2f}, current is {deficit:.2f} below)")

            if self.verbose > 0:
                print(f"{'='*60}\n")

        return True

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations performed during training."""
        return {
            'timesteps': self.evaluations_timesteps,
            'mean_rewards': self.evaluations_results,
            'mean_lengths': self.evaluations_length,
            "std_rewards": [float(s) for s in self.evaluations_std],
            "n_eval_episodes": int(self.n_eval_episodes),
            'best_mean_reward': float(self.best_mean_reward),
            'n_evaluations': len(self.evaluations_timesteps),
        }

def _deep_update(target: dict, updates: dict) -> None:
    for key, value in updates.items():
        ##############################
        # Recursively merge nested dicts
        ##############################
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value

def _resolve_env_path() -> str:
    platform = sys.platform
    ##############################
    # Set platform-specific binary candidates
    ##############################
    if platform == "darwin":
        candidates = [REPO_ROOT / "Godot_Air_Combat" / "B_ACE.app"]
    elif platform.startswith("linux"):
        candidates = [REPO_ROOT / "bin" / "B_ACE_v0.1.x86_64"]
    elif platform.startswith("win"):
        candidates = [REPO_ROOT / "bin" / "B_ACE_v0.1.exe"]
    else:
        raise RuntimeError(f"Unsupported platform '{platform}'.")

    ##############################
    # Search for first existing binary
    ##############################
    for binary in candidates:
        if binary.exists():
            return binary.as_posix()

    searched = "\n  - ".join(c.as_posix() for c in candidates)
    raise FileNotFoundError(
        "Could not find a suitable Godot binary. Checked:\n"
        f"  - {searched}"
    )

def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Builds (but does not parse with) the argument parser. Split out from _parse_args() so
    other tools (rl_files/run_from_config.py) can introspect the set of valid flag names
    without executing this module's __main__ body or touching sys.argv -- purely mechanical
    split, no change in parsed defaults/behavior.
    """
    parser = argparse.ArgumentParser(description="Train SB3 PPO on B-ACE with decay schedules and seed support.")
    parser.add_argument(
        "--config",
        type=str,
        default=(REPO_ROOT / "b_ace_py" / "SimpleExample_B_ACE_config.json").as_posix(),
        help="Path to a B-ACE JSON scenario config.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Total agent timesteps for SB3 PPO.learn()."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )

    parser.add_argument(
        "--initial-learning-rate",
        type=float,
        default=0.000258,
        help="Initial learning rate."
    )
    parser.add_argument(
        "--final-learning-rate",
        type=float,
        default=0.00001,
        help="Final learning rate after decay."
    )

    parser.add_argument(
        "--initial-entropy-coef",
        type=float,
        default=0.051,
        help="Initial entropy coefficient."
    )
    parser.add_argument(
        "--final-entropy-coef",
        type=float,
        default=0.005,
        help="Final entropy coefficient after decay."
    )

    parser.add_argument(
        "--decay-schedule",
        type=str,
        default="linear",
        choices=["linear", "exponential"],
        help="Type of decay schedule to use."
    )

    parser.add_argument(
        '--n_layers',
        type=int,
        default=2,
        help='Number of hidden layers in the neural network (default: 2)'
    )

    parser.add_argument(
        '--layer_size',
        type=int,
        default=512,
        help='Size of each hidden layer (default: 256)'
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor."
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.920664477208517,
        help="GAE lambda."
    )
    parser.add_argument(
        "--clip-eps",
        type=float,
        default=0.197991384633053,
        help="PPO clip range."
    )


    parser.add_argument(
        "--vf-coef",
        type=float,
        default=0.7728139,
        help="Value loss coefficient."
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=6144,
        help="Rollout length per PPO update."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Mini-batch size per PPO epoch."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="Number of epochs per PPO update."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="'cpu' or 'cuda'"
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help=(
            "PyTorch intraop thread count for CPU training (default: 1). With n_envs=1 and a "
            "small 2x512 MLP, PyTorch's default (one thread per physical core) spends more time "
            "on thread synchronization than it saves on these tiny matmuls, and those threads "
            "directly compete with the Godot subprocess for the same physical cores. Set higher "
            "only if profiling on your specific machine shows it helps."
        ),
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional custom run name for logs/checkpoints."
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Logical experiment name used to group runs under runs_sb3/EXPERIMENT_NAME"
    )

    parser.add_argument(
        "--metrics-interval",
        type=int,
        default=1000,
        help="Interval (in timesteps) for collecting detailed metrics."
    )

    parser.add_argument(
        "--use-enriched-obs",
        action="store_true",
        default=False,
        help="Enable enriched observations with pursuit-evasion heuristics"
    )
    parser.add_argument(
        "--ablation-config",
        type=str,
        default="all",
        choices=['all', 'none', 'geometry_only', 'engagement_only', 'range_limited_only',
                 'geometry_engagement', 'geometry_range', 'engagement_range'],
        help="""Ablation config for feature categories (only applies when --use-enriched-obs is set):
  all                 = A+B+C (GEOMETRY + ENGAGEMENT + RANGE_LIMITED)
  none                = No theoretical features (raw obs only)
  geometry_only       = A only (Apollonius + ATDDG)
  engagement_only     = B only (BEZ + DMC + WEZ)
  range_limited_only  = C only (Critical escape + capture probability)
  geometry_engagement = A+B (GEOMETRY + ENGAGEMENT)
  geometry_range      = A+C (GEOMETRY + RANGE_LIMITED)
  engagement_range    = B+C (ENGAGEMENT + RANGE_LIMITED)
"""
    )
    parser.add_argument(
        "--reward-mode",
        type=str,
        default="baseline",
        choices=["baseline", "mission_aligned"],
        help=(
            "'baseline' (default) trains with the existing, unmodified reward stack. "
            "'mission_aligned' zeroes MissionTempoShapingPerStep's per-step post-kill survival "
            "bonus (see env_factory.py's MISSION_ALIGNED_REWARD_STACK for why) and adds "
            "MissionOutcomeBonus (an explicit HVAA-loss penalty and a sized mission-success "
            "bonus). See REWARD_SHAPING_STEPS_FORWARD.md for the full rationale."
        ),
    )

    parser.add_argument(
        "--enable-apollonius", dest="enable_apollonius", action="store_true", default=True,
        help="Enable Apollonius circle geometry features (default: enabled)"
    )
    parser.add_argument(
        "--disable-apollonius", dest="enable_apollonius", action="store_false",
        help="Disable Apollonius circle geometry features"
    )

    parser.add_argument(
        "--enable-bez", dest="enable_bez", action="store_true", default=True,
        help="Enable Basic Engagement Zone features (default: enabled)"
    )
    parser.add_argument(
        "--disable-bez", dest="enable_bez", action="store_false",
        help="Disable Basic Engagement Zone features"
    )

    parser.add_argument(
        "--enable-dmc", dest="enable_dmc", action="store_true", default=True,
        help="Enable Dynamic Maneuvering Cue features (default: enabled)"
    )
    parser.add_argument(
        "--disable-dmc", dest="enable_dmc", action="store_false",
        help="Disable Dynamic Maneuvering Cue features"
    )

    parser.add_argument(
        "--enable-offense-wez", dest="enable_offense_wez", action="store_true", default=True,
        help="Enable offensive Weapon Engagement Zone features (default: enabled)"
    )
    parser.add_argument(
        "--disable-offense-wez", dest="enable_offense_wez", action="store_false",
        help="Disable offensive Weapon Engagement Zone features"
    )

    parser.add_argument(
        "--enable-offense-ttc", dest="enable_offense_ttc", action="store_true", default=True,
        help="Enable offensive Time-To-Capture features (default: enabled)"
    )
    parser.add_argument(
        "--disable-offense-ttc", dest="enable_offense_ttc", action="store_false",
        help="Disable offensive Time-To-Capture features"
    )

    parser.add_argument(
        "--enable-multi-threat", dest="enable_multi_threat", action="store_true", default=False,
        help="Enable multi-threat tracking features (default: disabled)"
    )
    parser.add_argument(
        "--disable-multi-threat", dest="enable_multi_threat", action="store_false",
        help="Disable multi-threat tracking features"
    )

    parser.add_argument(
        "--enable-hvaa-escort", dest="enable_hvaa_escort", action="store_true", default=False,
        help="Enable HVAA escort-specific features (default: disabled)"
    )
    parser.add_argument(
        "--disable-hvaa-escort", dest="enable_hvaa_escort", action="store_false",
        help="Disable HVAA escort-specific features"
    )

    parser.add_argument("--expert-alignment", action="store_true", default=False)
    parser.add_argument("--alignment-coef", type=float, default=0.002)
    parser.add_argument("--alignment-decay-start", type=int, default=500_000)
    parser.add_argument("--alignment-decay-end", type=int, default=2_000_000)

    parser.add_argument("--expert-blending", action="store_true", default=False,
                        help="Enable expert action blending (Config 3: blended action execution)")
    parser.add_argument("--blend-initial-alpha", type=float, default=0.8,
                        help="Initial alpha (expert influence) for blending. 1.0=all expert, 0.0=all PPO")
    parser.add_argument("--blend-final-alpha", type=float, default=0.05,
                        help="Final alpha after decay completes")
    parser.add_argument("--blend-decay-start", type=int, default=500_000,
                        help="Timestep to begin alpha decay (hold initial_alpha before this)")
    parser.add_argument("--blend-decay-end", type=int, default=5_000_000,
                        help="Timestep to reach final alpha")
    parser.add_argument("--blend-decay-type", type=str, default='linear',
                        choices=['linear', 'exponential', 'cosine'],
                        help="Type of alpha decay schedule")
    parser.add_argument("--blend-fire-mode", type=str, default='gate',
                        choices=['ppo', 'expert', 'blend', 'gate', 'or'],
                        help="How to handle fire action: gate=expert when alpha>0.3, ppo=always PPO")
    parser.add_argument("--blend-debug", action="store_true", default=False,
                        help="Enable debug output for blending wrapper")

    parser.add_argument("--use-wandb", action="store_true", default=False,
                        help="Opt-in: also mirror this run's TensorBoard logs to Weights & "
                             "Biases (requires `pip install wandb` and being logged in). "
                             "Off by default; training behaves identically without it.")
    parser.add_argument("--wandb-project", type=str, default="bvr-cca-escort",
                        help="W&B project name (only used if --use-wandb is set).")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="W&B entity/team name (only used if --use-wandb is set; "
                             "defaults to your default W&B entity).")

    parser.set_defaults(use_enriched_obs=False)

    return parser


def _parse_args():
    return _build_arg_parser().parse_args()


def main():
    args = _parse_args()

    ##############################
    # Cap PyTorch's CPU thread pool before any model/tensor work happens. See --torch-threads
    # help text: with a single environment and a small MLP, PyTorch's default multi-threaded
    # BLAS ops mostly just contend with the Godot subprocess for the same physical cores rather
    # than speeding anything up.
    ##############################
    torch.set_num_threads(max(1, args.torch_threads))

    np.random.seed(args.seed)

    config_path = Path(args.config).expanduser().resolve()
    B_ACE_config = load_b_ace_config(config_path.as_posix())
    if B_ACE_config is None:
        raise RuntimeError(f"Failed to load config file at {config_path}")

    env_cfg = B_ACE_config.setdefault("EnvConfig", {})

    if not env_cfg.get("env_path"):
        env_cfg["env_path"] = _resolve_env_path()

    env_path_value = Path(env_cfg["env_path"]).expanduser()
    ##############################
    # Resolve absolute path for environment binary
    ##############################
    if not env_path_value.is_absolute():
        env_path_value = (REPO_ROOT / env_path_value).resolve()
    else:
        env_path_value = env_path_value.resolve()

    ##############################
    # Check if environment binary exists, use fallback if needed
    ##############################
    if not env_path_value.exists():
        fallback_path = Path(_resolve_env_path()).resolve()
        if not fallback_path.exists():
            raise FileNotFoundError(
                f"[B-ACE ERROR] Tried {env_path_value}, and fallback {fallback_path}, "
                "but neither exists. You probably need to export the Godot binary."
            )
        print(
            f"Warning: Godot binary not found at {env_path_value}. "
            f"Falling back to {fallback_path}."
        )
        env_path_value = fallback_path

    env_cfg["env_path"] = env_path_value.as_posix()

    agents_cfg = B_ACE_config.setdefault("AgentsConfig", {})
    blue_cfg = agents_cfg.setdefault("blue_agents", {})
    red_cfg = agents_cfg.setdefault("red_agents", {})
    blue_cfg.setdefault("base_behavior", "external")
    red_cfg.setdefault("base_behavior", "baseline1")

    env_cfg.setdefault("renderize", 1)
    env_cfg.setdefault("speed_up", 200)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    ##############################
    # Determine run name for logging
    ##############################
    if args.run_name is not None:
        run_name = args.run_name
    elif args.experiment_name is not None:
        run_name = f"{args.experiment_name}_seed{args.seed}_{timestamp}"
    else:
        run_name = f"sb3ppo_bace_decay_seed{args.seed}_{timestamp}"

    base_runs_dir = REPO_ROOT / "runs_sb3"

    ##############################
    # Set up directory structure for logs and checkpoints
    ##############################
    if args.experiment_name is not None:
        log_dir = (base_runs_dir / args.experiment_name / run_name).resolve()
    else:
        log_dir = (base_runs_dir / run_name).resolve()

    log_dir.mkdir(parents=True, exist_ok=True)

    def make_env(log_dir_str, seed, is_eval: bool = False):
        def _thunk():
            return build_env(
                B_ACE_config,
                is_eval=is_eval,
                log_dir_str=log_dir_str,
                seed=seed,
                use_enriched_obs=args.use_enriched_obs,
                ablation_config=args.ablation_config,
                expert_alignment=args.expert_alignment,
                alignment_coef=args.alignment_coef,
                alignment_decay_start=args.alignment_decay_start,
                alignment_decay_end=args.alignment_decay_end,
                expert_blending=args.expert_blending,
                blend_initial_alpha=args.blend_initial_alpha,
                blend_final_alpha=args.blend_final_alpha,
                blend_decay_start=args.blend_decay_start,
                blend_decay_end=args.blend_decay_end,
                blend_decay_type=args.blend_decay_type,
                blend_fire_mode=args.blend_fire_mode,
                blend_debug=args.blend_debug,
                max_episode_steps=13_500,
                controlled_agent_id="agent_0",
                reward_mode=args.reward_mode,
            )
        return _thunk

    vec_env = DummyVecEnv([make_env(log_dir.as_posix(), args.seed)])

    ##############################
    # Apply VecNormalize wrapper for observation/reward normalization
    ##############################
    if args.use_enriched_obs:
        print("\n[VecNormalize] norm_obs=False (handled by SplitNormalizationWrapper)")
        vec_env = VecNormalize(
            vec_env,
            norm_obs=False,
            norm_reward=False,
            clip_obs=10.0,
            clip_reward=10.0,
        )
    else:
        print("\n[VecNormalize] norm_obs=True (baseline observations)")
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
            clip_reward=10.0,
        )
    print_wrapper_stack(vec_env, name="TRAIN vec_env (after VecNormalize)")

    print(f"[NEW***DEBUG] vec_env.observation_space: {vec_env.observation_space}")

    ##############################
    # Opt-in Weights & Biases mirror of the TensorBoard log directory (see
    # IMPROVEMENT_PLAN_1_VISUALIZATION_AND_RUNNING.md Task 1). Off by default; when off, this
    # block is a no-op and training is unaffected. `import wandb` is deliberately deferred to
    # inside this `if` so the package is only required when --use-wandb is actually passed.
    ##############################
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
            sync_tensorboard=True,
        )
        atexit.register(wandb.finish)
        print(f"\n[W&B] Logging to project='{args.wandb_project}' run='{run_name}' "
              f"(mirroring TensorBoard via sync_tensorboard=True)\n")

    logger = configure(log_dir.as_posix(), ["stdout", "tensorboard"])

    ##############################
    # Create learning rate and entropy decay schedules
    ##############################
    if args.decay_schedule == "linear":
        lr_schedule = linear_schedule(args.initial_learning_rate, args.final_learning_rate)
        ent_schedule = linear_schedule(args.initial_entropy_coef, args.final_entropy_coef)
    else:
        lr_schedule = exponential_schedule(args.initial_learning_rate, args.final_learning_rate)
        ent_schedule = exponential_schedule(args.initial_entropy_coef, args.final_entropy_coef)


    clip_schedule = args.clip_eps

    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"{'='*60}")
    print(f"Random Seed: {args.seed}")
    print(f"Decay Schedule Type: {args.decay_schedule}")
    print(f"Learning Rate: {args.initial_learning_rate:.2e} -> {args.final_learning_rate:.2e}")
    print(f"Entropy Coef: {args.initial_entropy_coef:.4f} -> {args.final_entropy_coef:.4f}")
    print(f"Clip Range: {args.clip_eps:.3f} (constant)")
    print(f"Total Timesteps: {args.total_timesteps:,}")
    print(f"{'='*60}\n")

    layer_sizes = [args.layer_size] * args.n_layers
    policy_kwargs = dict(net_arch=[dict(pi=layer_sizes, vf=layer_sizes)])

    def greedy_prob_fn(t: int) -> float:
        p_start = 1.0
        p_end   = 1.0
        hold_steps   = 500_000
        anneal_steps = 5_000_000

        if t < hold_steps:
            return p_start
        if t >= hold_steps + anneal_steps:
            return p_end

        frac = (t - hold_steps) / anneal_steps
        return p_start + frac * (p_end - p_start)

    bc_start = 1.0
    bc_end   = 0.5
    bc_steps = 6_000_000

    def bc_coef_fn(t: int) -> float:
        return 0.0

    model = SkippableTrainPPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=lr_schedule,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=clip_schedule,
        ent_coef=args.initial_entropy_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=0.5,
        target_kl=0.05,
        verbose=1,
        tensorboard_log=log_dir.as_posix(),
        device=args.device,
        policy_kwargs=policy_kwargs,
        seed=args.seed
    )

    model.set_logger(logger)

    ##############################
    # Create evaluation environment
    ##############################
    print(f"\n{'='*60}")
    print("Creating Evaluation Environment")
    print(f"{'='*60}")

    eval_env = DummyVecEnv([make_env(log_dir.as_posix(), args.seed + 10000, is_eval=True)])


    ##############################
    # Apply VecNormalize to evaluation environment
    ##############################
    if args.use_enriched_obs:
        print("\n[Eval VecNormalize] norm_obs=False (matching training - handled by SplitNormalizationWrapper)")
        eval_env = VecNormalize(
            eval_env,
            norm_obs=False,
            norm_reward=False,
            clip_obs=10.0,
            clip_reward=10.0,
            training=False,
        )
    else:
        print("\n[Eval VecNormalize] norm_obs=True (matching training - baseline observations)")
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
            clip_reward=10.0,
            training=False,
        )

    print_wrapper_stack(eval_env, name="EVAL eval_env (after VecNormalize)")

    ##############################
    # Sync normalization stats from training to evaluation environment
    ##############################
    if hasattr(vec_env, 'obs_rms') and vec_env.obs_rms is not None:
        eval_env.obs_rms = vec_env.obs_rms
        print("  [Normalization] Synced obs_rms from training to eval env")
    if hasattr(vec_env, 'ret_rms') and vec_env.ret_rms is not None:
        eval_env.ret_rms = vec_env.ret_rms
        print("  [Normalization] Synced ret_rms from training to eval env")

    print(f"Evaluation environment created")
    print(f"  Eval episodes: 10")
    print(f"  Eval frequency: every 250,000 steps")
    print(f"{'='*60}\n")

    ##############################
    # Create callbacks for training
    ##############################
    metrics_callback = MetricsCollectionCallback(
        entropy_schedule=ent_schedule,
        metrics_collection_interval=args.metrics_interval,
        max_stored_episodes=10000
    )

    eval_checkpoint_callback = RollbackEvaluationCallbackTopK(
        eval_env=eval_env,
        eval_freq=200000,
        n_eval_episodes=15,
        log_dir=log_dir,
        max_checkpoints=5,
        rollback_patience=6,
        deficit_threshold=40.0,
        min_timesteps_before_rollback=4_000_000,
        lr_decay_factor=0.7,
        entropy_decay_factor=0.5,
        min_lr_multiplier=0.4,
        max_rollbacks=5,
        skip_initial_eval=True,
        activation_threshold=12.0,
        freeze_vecnormalize_until_new_best=True,
        verbose=1,
        use_enriched_obs=args.use_enriched_obs,
    )

    ##############################
    # Combine callbacks
    ##############################
    callback_list = CallbackList([metrics_callback, eval_checkpoint_callback])

    ##############################
    # Run training
    ##############################
    start_time = time.time()
    model.learn(
        total_timesteps=args.total_timesteps,
        progress_bar=True,
        callback=callback_list
    )
    training_time = time.time() - start_time

    ##############################
    # Run final evaluation after training completes
    ##############################
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION (after training complete)")
    print(f"   Running 20 episodes with FINAL model...")
    print(f"{'='*60}")

    final_eval_start = time.time()

    final_episode_rewards = []
    final_episode_lengths = []

    obs = eval_env.reset()
    episode_reward = 0.0
    episode_length = 0
    episodes_completed = 0

    ##############################
    # Run final evaluation episodes
    ##############################
    while episodes_completed < 20:
        action, _ = model.predict(obs, deterministic=True)
        ##############################
        # Apply action gate during final eval
        ##############################
        try:
            if hasattr(eval_env, "get_original_obs"):
                obs_for_gate = eval_env.get_original_obs()
                if obs_for_gate is None:
                    obs_for_gate = obs
            else:
                obs_for_gate = obs

            gated = eval_env.env_method("gate_action", action[0], obs_for_gate[0], indices=[0])[0]
            action = np.asarray([gated], dtype=np.float32)
        except Exception:
            pass
        step_out = eval_env.step(action)
        ##############################
        # Handle different return formats from VecEnv
        ##############################
        if len(step_out) == 4:
            obs, reward, done, infos = step_out
        else:
            obs, reward, terminated, truncated, infos = step_out
            done = np.logical_or(terminated, truncated)

        episode_reward += reward[0]
        episode_length += 1

        if done[0]:
            final_episode_rewards.append(episode_reward)
            final_episode_lengths.append(episode_length)
            episodes_completed += 1

            episode_reward = 0.0
            episode_length = 0

    final_mean_reward = np.mean(final_episode_rewards)
    final_std_reward = np.std(final_episode_rewards)
    final_mean_length = np.mean(final_episode_lengths)
    final_eval_time = time.time() - final_eval_start

    print(f"Final Model Evaluation Results:")
    print(f"   Mean Reward: {final_mean_reward:.2f} ± {final_std_reward:.2f}")
    print(f"   Mean Length: {final_mean_length:.1f}")
    print(f"   Time: {final_eval_time:.1f}s")
    print(f"{'='*60}\n")

    ##############################
    # Save final model and normalization stats
    ##############################
    model_path = log_dir / "ppo_bace_blue_lead"
    norm_path = log_dir / "vecnormalize.pkl"
    model.save(model_path.as_posix())
    vec_env.save(norm_path.as_posix())

    ##############################
    # Save SplitNormalizationWrapper stats for final model
    ##############################
    if args.use_enriched_obs:
        split_norm = get_split_norm_wrapper_from_vec_env(vec_env)
        if split_norm is not None:
            final_split_norm_path = log_dir / "split_normalization.npz"
            split_norm.save(final_split_norm_path.as_posix())
            print(f"SplitNormalization stats saved to {final_split_norm_path}")
            print(f"  (count={split_norm.count}, mean_range=[{split_norm.obs_mean.min():.2f}, {split_norm.obs_mean.max():.2f}])")

    ##############################
    # Get evaluation summary from callback
    ##############################
    eval_summary = eval_checkpoint_callback.get_evaluation_summary()

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Training time: {training_time/60:.1f} minutes")
    print(f"")
    print(f"FINAL MODEL:")
    print(f"  Path: {model_path}")
    print(f"  Eval reward: {final_mean_reward:.2f} ± {final_std_reward:.2f}")
    print(f"")
    print(f"BEST MODEL:")
    print(f"  Path: {log_dir / 'best_model'}")
    print(f"  Eval reward: {eval_checkpoint_callback.best_mean_reward:.2f}")
    ##############################
    # No periodic evaluation may have run yet (e.g. total_timesteps < eval_freq),
    # in which case best_mean_reward is still its -inf sentinel and won't be in the list.
    ##############################
    if eval_summary["mean_rewards"] and eval_checkpoint_callback.best_mean_reward in eval_summary["mean_rewards"]:
        best_idx = eval_summary["mean_rewards"].index(eval_checkpoint_callback.best_mean_reward)
        print(f"  Found at: {eval_summary['timesteps'][best_idx]:,} steps")
        print(f"")
        print(f"Comparison: Best is {eval_checkpoint_callback.best_mean_reward - final_mean_reward:.2f} points better than final")
    else:
        print(f"  (no periodic evaluation completed during this run)")
        print(f"")
    print(f"Number of evaluations: {eval_summary['n_evaluations']}")
    print(f"{'='*60}\n")

    ##############################
    # Collect and save metrics from training
    ##############################
    metrics_summary = metrics_callback.get_metrics_summary()

    training_config = {
        'experiment_name': args.experiment_name or "default_experiment",
        'seed': args.seed,
        'total_timesteps': args.total_timesteps,
        'training_time_seconds': training_time,
        'decay_schedule': args.decay_schedule,
        'initial_learning_rate': args.initial_learning_rate,
        'final_learning_rate': args.final_learning_rate,
        'initial_entropy_coef': args.initial_entropy_coef,
        'final_entropy_coef': args.final_entropy_coef,
        'gamma': args.gamma,
        'gae_lambda': args.gae_lambda,
        'n_steps': args.n_steps,
        'batch_size': args.batch_size,
        'n_epochs': args.epochs,
        'n_layers': args.n_layers,
        'layer_size': args.layer_size,
        'vf_coef': args.vf_coef,
        'expert_alignment': args.expert_alignment,
        'expert_blending': args.expert_blending,
        'blend_initial_alpha': args.blend_initial_alpha if args.expert_blending else None,
        'blend_final_alpha': args.blend_final_alpha if args.expert_blending else None,
        'blend_decay_start': args.blend_decay_start if args.expert_blending else None,
        'blend_decay_end': args.blend_decay_end if args.expert_blending else None,
        'blend_decay_type': args.blend_decay_type if args.expert_blending else None,
        'blend_fire_mode': args.blend_fire_mode if args.expert_blending else None,
    }

    results = {
        'config': training_config,
        'metrics': metrics_summary,
        'evaluation_summary': eval_summary,
        'final_evaluation': {
            'mean_reward': float(final_mean_reward),
            'std_reward': float(final_std_reward),
            'mean_length': float(final_mean_length),
            'episode_rewards': [float(r) for r in final_episode_rewards],
        },
        'model_path': model_path.as_posix(),
        'best_model_path': (log_dir / "best_model.zip").as_posix(),
        'best_eval_mean_reward': float(eval_checkpoint_callback.best_mean_reward),
        'final_vs_best_difference': float(eval_checkpoint_callback.best_mean_reward - final_mean_reward),
        'eval_checkpoints_dir': eval_checkpoint_callback.eval_checkpoint_dir.as_posix(),
        'log_dir': log_dir.as_posix()
    }

    ##############################
    # Save results to JSON file
    ##############################
    results_path = log_dir / f"training_results_seed{args.seed}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"[SB3 PPO] Training complete!")
    print(f"{'='*60}")
    print(f"Seed: {args.seed}")
    print(f"Training time: {training_time/60:.2f} minutes")
    print(f"Total episodes: {metrics_summary['total_episodes']}")
    print(f"Mean reward: {metrics_summary['mean_reward']:.2f} ± {metrics_summary['std_reward']:.2f}")
    print(f"Model saved to: {model_path}")
    print(f"VecNormalize stats saved to: {norm_path}")
    print(f"Results saved to: {results_path}")
    print(f"TensorBoard logs: {log_dir}")
    print(f"{'='*60}\n")

    eval_env.close()

    if args.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
