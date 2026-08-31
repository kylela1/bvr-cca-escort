#!/usr/bin/env python3
"""
This script evaluates trained PPO models produced by run_experiment.py across all seeds in
an experiment directory. For each seed, it loads the saved model and VecNormalize statistics,
runs a configurable number of evaluation episodes in the B-ACE Godot environment, and
computes per-seed and aggregate metrics including mean reward, HVAA survival rate, red kill
rate, and blue loss rate. It supports enriched observations with split normalization and
ablation configurations matching those used during training.

Within the broader pipeline, this is the evaluation counterpart to run_experiment.py. It
reads the model checkpoints and VecNormalize pickle files saved during training, reconstructs
the same environment wrapper stack (including EnrichedObservationWrapper and
SplitNormalizationWrapper when applicable), and produces detailed JSON results with
confidence intervals for cross-seed statistical analysis.

Key configurable parameters: --episodes sets the number of evaluation episodes per seed
(default: 100). --use-enriched-obs and --ablation-config must match the training
configuration. --deterministic controls whether the policy acts greedily (default: True).
--output-json specifies where aggregate results are saved. The script auto-discovers seed
directories by globbing for patterns matching *seed* under the experiment directory.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import torch

PM = "\u00b1"

THIS_FILE = Path(__file__).resolve()
current_path = THIS_FILE.parent
##############################
# Walk up directory tree to find repository root
##############################
while current_path != current_path.parent:
    ##############################
    # Check if this is the repo root by looking for b_ace_py directory
    ##############################
    if (current_path / "b_ace_py").exists():
        REPO_ROOT = current_path
        break
    current_path = current_path.parent
else:
    REPO_ROOT = THIS_FILE.parent.parent

##############################
# Add repository root to Python path if not already present
##############################
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

print(f"Using REPO_ROOT: {REPO_ROOT}")

from b_ace_py.utils import load_b_ace_config

from env_factory import build_env, get_split_norm_wrapper_from_vec_env, FULLY_ANNEALED_STEP


@dataclass
class ConfidenceInterval:
    """Stores a metric with its confidence interval."""
    mean: float
    ci_lower: float
    ci_upper: float
    ci_width: float
    n: int
    method: str
    count: Optional[int] = None
    std: Optional[float] = None


@dataclass
class EpisodeData:
    """Per-episode data from evaluation."""
    reward: float
    length: int
    hvaa_survived: bool
    red_killed: bool
    blue_killed: bool
    blue_missiles_fired: int = 0
    red_missiles_fired: int = 0
    blue_pk: float = 0.0
    red_pk: float = 0.0
    blue_shot_first: bool = False
    red_shot_first: bool = False
    no_shots_fired: bool = False
    min_separation_nm: float = -1.0
    end_reason: str = ""
    training_seed: str = ""


@dataclass
class SeedResults:
    """Results for a single training seed."""
    seed_name: str
    model_path: str
    vecnorm_path: str
    n_episodes: int
    mean_reward: float
    std_reward: float
    hvaa_survival_rate: float
    red_kill_rate: float
    blue_loss_rate: float
    mean_length: float
    mean_blue_missiles: float
    mean_red_missiles: float
    mean_blue_pk: float
    mean_red_pk: float
    blue_shot_first_rate: float
    red_shot_first_rate: float
    no_shots_rate: float
    episodes: List[EpisodeData]


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> ConfidenceInterval:
    """Wilson score CI for binomial proportions."""
    ##############################
    # Handle edge case of zero trials
    ##############################
    if total == 0:
        return ConfidenceInterval(0.0, 0.0, 0.0, 0.0, 0, "wilson_score", count=0)

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / total

    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator

    return ConfidenceInterval(
        mean=float(p),
        ci_lower=float(max(0, center - margin)),
        ci_upper=float(min(1, center + margin)),
        ci_width=float(min(1, center + margin) - max(0, center - margin)),
        n=total,
        method="wilson_score",
        count=successes
    )


def t_distribution_ci(data: List[float], confidence: float = 0.95) -> ConfidenceInterval:
    """t-distribution CI for continuous data."""
    arr = np.array(data)
    n = len(arr)

    ##############################
    # Handle case of very few data points (less than 2 samples)
    ##############################
    if n < 2:
        mean = float(np.mean(arr)) if n > 0 else 0.0
        return ConfidenceInterval(mean, mean, mean, 0.0, n, "t_distribution", std=0.0)

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    sem = stats.sem(arr)
    h = sem * stats.t.ppf((1 + confidence) / 2, n - 1)

    return ConfidenceInterval(
        mean=mean,
        ci_lower=float(mean - h),
        ci_upper=float(mean + h),
        ci_width=float(2 * h),
        n=n,
        method="t_distribution",
        std=std
    )


class _DummyVecEnv:
    """Dummy class to replace VecEnv during unpickling."""
    def __init__(self, *args, **kwargs):
        pass
    def __setstate__(self, state):
        ##############################
        # Restore state attributes from dictionary
        ##############################
        for key, value in state.items():
            setattr(self, key, value)


class _VecNormalizeStatsOnly:
    """Dummy class that captures VecNormalize attributes without recursion."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        ##############################
        # Store state dictionary attributes directly
        ##############################
        for key, value in state.items():
            setattr(self, key, value)


class SafeVecNormalizeUnpickler(pickle.Unpickler):
    """
    Custom unpickler that safely loads VecNormalize pickle files.

    This avoids recursion errors that occur when SB3 tries to reconstruct
    the full VecEnv chain during unpickling. We only extract the stats
    we need (obs_rms, ret_rms, norm_obs) without loading the wrapped env.
    """

    def find_class(self, module, name):
        ##############################
        # Replace VecNormalize with dummy class to avoid recursion
        ##############################
        if name == 'VecNormalize':
            return _VecNormalizeStatsOnly
        ##############################
        # Use actual RunningMeanStd implementation
        ##############################
        if name == 'RunningMeanStd':
            from stable_baselines3.common.running_mean_std import RunningMeanStd
            return RunningMeanStd
        ##############################
        # Replace any VecEnv classes with dummy to prevent recursion
        ##############################
        if 'VecEnv' in name or name in ('DummyVecEnv', 'SubprocVecEnv'):
            return _DummyVecEnv
        ##############################
        # Try default behavior for everything else
        ##############################
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            ##############################
            # Return dummy for unknown classes
            ##############################
            return _DummyVecEnv


def safe_load_vecnormalize_stats(vecnorm_path: Path) -> _VecNormalizeStatsOnly:
    """Safely load VecNormalize stats without recursion errors."""
    with open(vecnorm_path, 'rb') as f:
        return SafeVecNormalizeUnpickler(f).load()


def detect_obs_dim_from_vecnormalize(vecnorm_path: Path) -> Optional[int]:
    """Load VecNormalize pickle and extract expected observation dimension."""
    try:
        data = safe_load_vecnormalize_stats(vecnorm_path)
    except Exception as e:
        print(f"   DEBUG: Failed to load vecnormalize: {e}")
        return None

    ##############################
    # Try to extract dimension from obs_rms first
    ##############################
    if hasattr(data, 'obs_rms') and data.obs_rms is not None:
        ##############################
        # Check if obs_rms has mean attribute with shape
        ##############################
        if hasattr(data.obs_rms, 'mean'):
            obs_dim = data.obs_rms.mean.shape[0]
            return obs_dim

    ##############################
    # Try observation_space as fallback
    ##############################
    if hasattr(data, 'observation_space') and data.observation_space is not None:
        ##############################
        # Extract shape from observation space
        ##############################
        if hasattr(data.observation_space, 'shape'):
            return data.observation_space.shape[0]

    ##############################
    # Debug print available attributes
    ##############################
    available_attrs = [a for a in dir(data) if not a.startswith('_')]
    print(f"   DEBUG: VecNormalize attributes: {available_attrs}")
    ##############################
    # Print obs_rms details if present
    ##############################
    if hasattr(data, 'obs_rms'):
        print(f"   DEBUG: obs_rms = {data.obs_rms}")

    return None


def detect_norm_obs_setting(vecnorm_path: Path) -> bool:
    """Detect whether the VecNormalize was saved with norm_obs=True or False."""
    try:
        data = safe_load_vecnormalize_stats(vecnorm_path)
    except Exception as e:
        print(f"   DEBUG: Failed to load vecnormalize for norm_obs detection: {e}")
        return True

    ##############################
    # Check if norm_obs attribute exists
    ##############################
    if hasattr(data, 'norm_obs'):
        return data.norm_obs

    ##############################
    # Fallback: check if obs_rms has evidence of normalization
    ##############################
    if hasattr(data, 'obs_rms') and data.obs_rms is not None:
        ##############################
        # Check observation count and variance to infer normalization use
        ##############################
        if hasattr(data.obs_rms, 'count') and data.obs_rms.count > 0:
            var_not_default = not np.allclose(data.obs_rms.var, 1.0)
            return var_not_default

    return True


def detect_obs_dim_from_model(model_path: Path) -> Optional[int]:
    """Fallback: detect observation dimension from the model's policy network."""
    try:
        ##############################
        # Handle zip-formatted SB3 models
        ##############################
        if model_path.suffix == '.zip':
            import zipfile
            import io
            with zipfile.ZipFile(model_path, 'r') as zf:
                ##############################
                # Extract policy weights from zip
                ##############################
                if 'policy.pth' in zf.namelist():
                    with zf.open('policy.pth') as f:
                        state_dict = torch.load(io.BytesIO(f.read()), map_location='cpu')
                        ##############################
                        # Find first layer input size which equals obs dimension
                        ##############################
                        for key in state_dict:
                            ##############################
                            # Check for MLP policy network first layer
                            ##############################
                            if 'mlp_extractor.policy_net.0.weight' in key:
                                return state_dict[key].shape[1]
                            ##############################
                            # Check for feature extractor layer
                            ##############################
                            if 'features_extractor' in key and 'weight' in key:
                                return state_dict[key].shape[1]
        ##############################
        # Handle raw .pt checkpoint files
        ##############################
        elif model_path.suffix == '.pt':
            state_dict = torch.load(model_path, map_location='cpu')
            ##############################
            # Search state dict for first layer weight
            ##############################
            for key in state_dict:
                ##############################
                # Look for policy network first layer
                ##############################
                if 'mlp_extractor.policy_net.0.weight' in key:
                    return state_dict[key].shape[1]
    except Exception as e:
        print(f"   DEBUG: Could not detect obs dim from model: {e}")
    return None


def determine_obs_mode(obs_dim: int) -> str:
    """Map observation dimension to mode name."""
    ##############################
    # Small observation space indicates baseline
    ##############################
    if obs_dim <= 30:
        return "baseline"
    ##############################
    # Medium observation space indicates enriched
    ##############################
    elif obs_dim <= 50:
        return "enriched"
    ##############################
    # Large observation space indicates extended enriched
    ##############################
    else:
        return "enriched_extended"


def _resolve_env_path() -> str:
    platform = sys.platform
    ##############################
    # Set search candidates based on platform
    ##############################
    if platform == "darwin":
        candidates = [REPO_ROOT / "Godot_Air_Combat" / "B_ACE.app"]
    elif platform.startswith("linux"):
        candidates = [REPO_ROOT / "bin" / "B_ACE_v0.1.x86_64"]
    elif platform.startswith("win"):
        candidates = [REPO_ROOT / "bin" / "B_ACE_v0.1.exe"]
    else:
        raise RuntimeError(f"Unsupported platform '{platform}'")

    ##############################
    # Check each candidate path and return first that exists
    ##############################
    for binary in candidates:
        ##############################
        # Return binary path if it exists
        ##############################
        if binary.exists():
            return binary.as_posix()

    raise FileNotFoundError(f"Could not find Godot binary")


def _as_bool(x) -> bool:
    """Safe boolean conversion - handles numpy types."""
    try:
        ##############################
        # Handle numpy and Python bool types
        ##############################
        if isinstance(x, (np.bool_, bool)):
            return bool(x)
        ##############################
        # Handle numeric types by checking if non-zero
        ##############################
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x) != 0.0
        return bool(x)
    except Exception:
        return False


def _as_int(v) -> int:
    """Safe integer conversion."""
    try:
        return int(v)
    except:
        return 0


def _jsonify(obj):
    ##############################
    # Convert numpy arrays to lists
    ##############################
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    ##############################
    # Convert numpy integer types
    ##############################
    if isinstance(obj, (np.integer,)):
        return int(obj)
    ##############################
    # Convert numpy floating types
    ##############################
    if isinstance(obj, (np.floating,)):
        return float(obj)
    ##############################
    # Convert numpy bool type
    ##############################
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    ##############################
    # Pass through native Python types
    ##############################
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    ##############################
    # Recursively process dictionaries
    ##############################
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    ##############################
    # Recursively process sequences
    ##############################
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    ##############################
    # Try to convert objects with __dict__ attribute
    ##############################
    if hasattr(obj, '__dict__'):
        return _jsonify(obj.__dict__)
    return str(obj)


def _infer_net_arch_from_state_dict(state_dict: dict, verbose: bool = True) -> List[int]:
    """Infer the network architecture from a policy state dict."""
    hidden_sizes = []
    layer_idx = 0
    ##############################
    # Iterate through policy network layers
    ##############################
    while True:
        key = f"mlp_extractor.policy_net.{layer_idx}.weight"
        ##############################
        # Check if layer exists in state dict
        ##############################
        if key in state_dict:
            weight_shape = state_dict[key].shape
            hidden_size = weight_shape[0]
            hidden_sizes.append(hidden_size)
            layer_idx += 2
        else:
            break

    ##############################
    # Print inferred architecture if verbose and layers found
    ##############################
    if verbose and hidden_sizes:
        print(f"   → Inferred network architecture: {hidden_sizes}")

    return hidden_sizes if hidden_sizes else [64, 64]


def load_model_with_pt_fallback(model_path: Path, env, verbose: bool = True) -> PPO:
    """Load a PPO model, falling back to .pt files if .zip is corrupt/empty.

    Also handles being given a .pt file directly (e.g., from timestep checkpoints).
    """
    ##############################
    # Handle direct .pt file input
    ##############################
    if model_path.suffix == '.pt' and model_path.exists() and model_path.stat().st_size > 0:
        policy_pt_path = model_path
        ##############################
        # Print message about loading .pt checkpoint
        ##############################
        if verbose:
            print(f"   → Loading .pt checkpoint directly: {policy_pt_path.name}")
        checkpoint = torch.load(policy_pt_path, map_location="cpu", weights_only=False)

        ##############################
        # Extract state dict from checkpoint format
        ##############################
        if isinstance(checkpoint, dict):
            ##############################
            # Try different possible keys for state dict
            ##############################
            if "policy_state_dict" in checkpoint:
                state_dict = checkpoint["policy_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        net_arch = _infer_net_arch_from_state_dict(state_dict, verbose=verbose)
        policy_kwargs = dict(net_arch=net_arch)

        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=0,
            device="cpu",
            policy_kwargs=policy_kwargs,
        )

        model.policy.load_state_dict(state_dict)

        ##############################
        # Print success message
        ##############################
        if verbose:
            print(f"   ✓ Loaded policy weights from {policy_pt_path.name}")

        return model

    ##############################
    # Try loading standard SB3 .zip file first
    ##############################
    try:
        ##############################
        # Check if file exists and is not empty
        ##############################
        if model_path.exists() and model_path.stat().st_size > 0:
            model = PPO.load(model_path.as_posix(), device="cpu")
            model.set_env(env)
            ##############################
            # Print success message
            ##############################
            if verbose:
                print(f"   ✓ Loaded model from {model_path.name}")
            return model
        else:
            raise ValueError(f"Model file is empty or missing: {model_path}")
    except Exception as zip_error:
        ##############################
        # Print warning about .zip loading failure
        ##############################
        if verbose:
            print(f"   ⚠ Failed to load {model_path.name}: {zip_error}")

        model_dir = model_path.parent
        pt_candidates = [
            model_dir / "best_policy.pt",
            model_dir / "policy.pt",
            model_dir / (model_path.stem.replace("model", "policy") + ".pt"),
        ]

        policy_pt_path = None
        ##############################
        # Search for valid .pt fallback file
        ##############################
        for pt_path in pt_candidates:
            ##############################
            # Check if candidate file exists and is not empty
            ##############################
            if pt_path.exists() and pt_path.stat().st_size > 0:
                policy_pt_path = pt_path
                break

        ##############################
        # Raise error if no fallback found
        ##############################
        if policy_pt_path is None:
            raise FileNotFoundError(
                f"Could not load model from {model_path} and no valid .pt fallback found."
            )

        ##############################
        # Print fallback message
        ##############################
        if verbose:
            print(f"   → Falling back to {policy_pt_path.name}")

        checkpoint = torch.load(policy_pt_path, map_location="cpu", weights_only=False)

        ##############################
        # Extract state dict from checkpoint format
        ##############################
        if isinstance(checkpoint, dict):
            ##############################
            # Try different possible keys for state dict
            ##############################
            if "policy_state_dict" in checkpoint:
                state_dict = checkpoint["policy_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        net_arch = _infer_net_arch_from_state_dict(state_dict, verbose=verbose)
        policy_kwargs = dict(net_arch=net_arch)

        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=0,
            device="cpu",
            policy_kwargs=policy_kwargs,
        )

        model.policy.load_state_dict(state_dict)

        ##############################
        # Print success message
        ##############################
        if verbose:
            print(f"   ✓ Loaded policy weights from {policy_pt_path.name}")

        return model


def find_seed_models(experiment_dir: Path) -> Dict[str, Tuple[Path, Path, Optional[Path]]]:
    """Find best model, vecnormalize, and split_normalization for each seed in experiment directory.

    Returns:
        Dict mapping seed_name -> (model_path, vecnorm_path, split_norm_path)
        split_norm_path may be None if not found (for baseline or legacy runs)
    """
    results: Dict[str, Tuple[Path, Path, Optional[Path]]] = {}

    ##############################
    # Iterate through all seed directories
    ##############################
    for seed_dir in sorted(experiment_dir.glob("*seed*")):
        ##############################
        # Skip non-directory entries
        ##############################
        if not seed_dir.is_dir():
            continue

        match = re.search(r"seed(\d+)", seed_dir.name)
        seed_name = f"seed{match.group(1)}" if match else seed_dir.name

        model_candidates = [
            seed_dir / "best_model.zip",
            seed_dir / "best_checkpoint" / "best_model.zip",
            seed_dir / "eval_checkpoints" / "best_model.zip",
        ]

        pt_fallback_candidates = [
            seed_dir / "best_policy.pt",
            seed_dir / "best_checkpoint" / "best_policy.pt",
            seed_dir / "eval_checkpoints" / "best_policy.pt",
        ]

        model_path = None
        using_pt_fallback = False

        ##############################
        # Search for model file
        ##############################
        for c in model_candidates:
            ##############################
            # Use first found model candidate
            ##############################
            if c.exists():
                model_path = c
                break

        ##############################
        # Fall back to .pt file if no .zip found
        ##############################
        if model_path is None:
            ##############################
            # Search for .pt fallback files
            ##############################
            for c in pt_fallback_candidates:
                ##############################
                # Use first found .pt file that is not empty
                ##############################
                if c.exists() and c.stat().st_size > 0:
                    model_path = c.parent / "best_model.zip"
                    using_pt_fallback = True
                    break

        ##############################
        # Skip seed if no model found
        ##############################
        if model_path is None:
            print(f"Warning: No best_model.zip or best_policy.pt found for {seed_dir.name}")
            continue

        vecnorm_candidates = [
            seed_dir / "best_vecnormalize.pkl",
            seed_dir / "vecnormalize.pkl",
            seed_dir / "best_checkpoint" / "best_vecnormalize.pkl",
            seed_dir / "best_checkpoint" / "vecnormalize.pkl",
            seed_dir / "eval_checkpoints" / "best_vecnormalize.pkl",
        ]
        vecnorm_path = None
        ##############################
        # Search for vecnormalize file
        ##############################
        for c in vecnorm_candidates:
            ##############################
            # Use first found vecnormalize file
            ##############################
            if c.exists():
                vecnorm_path = c
                break

        ##############################
        # Skip seed if no vecnormalize found
        ##############################
        if vecnorm_path is None:
            print(f"Warning: No vecnormalize.pkl found for {seed_dir.name}")
            continue

        ##############################
        # Look for SplitNormalizationWrapper stats
        ##############################
        split_norm_candidates = [
            seed_dir / "best_split_normalization.npz",
            seed_dir / "split_normalization.npz",
            seed_dir / "best_checkpoint" / "best_split_normalization.npz",
            seed_dir / "best_checkpoint" / "split_normalization.npz",
            seed_dir / "eval_checkpoints" / "best_split_normalization.npz",
        ]
        split_norm_path = None
        ##############################
        # Search for split normalization file
        ##############################
        for c in split_norm_candidates:
            ##############################
            # Use first found split normalization file
            ##############################
            if c.exists():
                split_norm_path = c
                break

        results[seed_name] = (model_path, vecnorm_path, split_norm_path)
        note = " (will use .pt fallback)" if using_pt_fallback else ""
        ##############################
        # Add note if model file is empty
        ##############################
        if model_path.exists() and model_path.stat().st_size == 0:
            note = " (empty .zip, will use .pt fallback)"

        split_note = f", split_norm={split_norm_path.name}" if split_norm_path else ""
        print(f"Found {seed_name}: {model_path.relative_to(seed_dir)}, {vecnorm_path.relative_to(seed_dir)}{split_note}{note}")

    return results


def discover_available_timesteps(experiment_dir: Path) -> Dict[str, List[int]]:
    """Discover all available timestep checkpoints for each seed.

    Returns:
        Dict mapping seed_name -> sorted list of available timesteps
    """
    results: Dict[str, List[int]] = {}

    ##############################
    # Iterate through all seed directories
    ##############################
    for seed_dir in sorted(experiment_dir.glob("*seed*")):
        ##############################
        # Skip non-directory entries
        ##############################
        if not seed_dir.is_dir():
            continue

        match = re.search(r"seed(\d+)", seed_dir.name)
        seed_name = f"seed{match.group(1)}" if match else seed_dir.name

        eval_ckpt_dir = seed_dir / "eval_checkpoints"
        ##############################
        # Skip seed if no eval_checkpoints directory
        ##############################
        if not eval_ckpt_dir.is_dir():
            continue

        timesteps = set()
        ##############################
        # Search for policy files with timestep in name
        ##############################
        for pt_file in eval_ckpt_dir.glob("policy_*_steps.pt"):
            ts_match = re.search(r"policy_(\d+)_steps\.pt", pt_file.name)
            ##############################
            # Extract timestep from filename
            ##############################
            if ts_match:
                timesteps.add(int(ts_match.group(1)))

        ##############################
        # Add seed to results if timesteps found
        ##############################
        if timesteps:
            results[seed_name] = sorted(timesteps)

    return results


def find_seed_checkpoints_by_timestep(
    experiment_dir: Path,
    timestep: int,
) -> Dict[str, Tuple[Path, Path, Optional[Path]]]:
    """Find checkpoint files for a specific timestep across all seeds.

    Looks inside each seed's eval_checkpoints/ directory for:
        policy_{timestep}_steps.pt
        vecnormalize_{timestep}_steps.pkl
        split_normalization_{timestep}_steps.npz  (optional)

    Returns:
        Dict mapping seed_name -> (policy_pt_path, vecnorm_path, split_norm_path)
        Only seeds that have the requested timestep are included.
    """
    results: Dict[str, Tuple[Path, Path, Optional[Path]]] = {}

    ##############################
    # Iterate through all seed directories
    ##############################
    for seed_dir in sorted(experiment_dir.glob("*seed*")):
        ##############################
        # Skip non-directory entries
        ##############################
        if not seed_dir.is_dir():
            continue

        match = re.search(r"seed(\d+)", seed_dir.name)
        seed_name = f"seed{match.group(1)}" if match else seed_dir.name

        eval_ckpt_dir = seed_dir / "eval_checkpoints"
        ##############################
        # Skip seed if no eval_checkpoints directory
        ##############################
        if not eval_ckpt_dir.is_dir():
            continue

        policy_path = eval_ckpt_dir / f"policy_{timestep}_steps.pt"
        vecnorm_path = eval_ckpt_dir / f"vecnormalize_{timestep}_steps.pkl"

        ##############################
        # Skip seed if policy checkpoint doesn't exist
        ##############################
        if not policy_path.exists():
            continue
        ##############################
        # Skip seed if vecnormalize checkpoint doesn't exist
        ##############################
        if not vecnorm_path.exists():
            print(f"Warning: {seed_name} has policy at timestep {timestep} but no vecnormalize – skipping")
            continue

        split_norm_path = eval_ckpt_dir / f"split_normalization_{timestep}_steps.npz"
        ##############################
        # Check if split normalization file exists
        ##############################
        if not split_norm_path.exists():
            split_norm_path = None

        results[seed_name] = (policy_path, vecnorm_path, split_norm_path)
        split_note = f", split_norm={split_norm_path.name}" if split_norm_path else ""
        print(f"Found {seed_name} @ timestep {timestep}: {policy_path.name}, {vecnorm_path.name}{split_note}")

    return results


def evaluate_model(
    model_path: Path,
    vecnorm_path: Path,
    bace_cfg: Dict,
    n_episodes: int,
    training_seed: str,
    eval_seed: int = 42,
    obs_mode_override: Optional[str] = None,
    use_enriched_obs: bool = False,
    feature_config: Optional[Dict[str, Any]] = None,
    ablation_config: Optional[str] = None,
    verbose: bool = True,
    strict_validation: bool = True,
    split_norm_path: Optional[Path] = None,
) -> List[EpisodeData]:
    """Run evaluation episodes for a single model."""

    ##############################
    # Auto-detect observation dimension from vecnormalize
    ##############################
    expected_obs_dim = detect_obs_dim_from_vecnormalize(vecnorm_path)

    ##############################
    # Fall back to model-based detection if vecnormalize detection fails
    ##############################
    if expected_obs_dim is None:
        expected_obs_dim = detect_obs_dim_from_model(model_path)
        ##############################
        # Print message if detection succeeds
        ##############################
        if expected_obs_dim is not None:
            print(f"   Detected obs dim from model: {expected_obs_dim}")

    ##############################
    # Use default dimensions if detection fails
    ##############################
    if expected_obs_dim is None:
        ##############################
        # Choose default based on observation mode
        ##############################
        if use_enriched_obs or (obs_mode_override and obs_mode_override != "baseline"):
            expected_obs_dim = 40
            print(f"   Using default enriched obs dim: {expected_obs_dim}")
        else:
            expected_obs_dim = 27
            print(f"   Using default baseline obs dim: {expected_obs_dim}")

    ##############################
    # Detect if normalization was applied during training
    ##############################
    saved_norm_obs = detect_norm_obs_setting(vecnorm_path)

    ##############################
    # Determine observation mode for this evaluation
    ##############################
    if obs_mode_override and obs_mode_override != "auto":
        obs_mode = obs_mode_override
    elif use_enriched_obs:
        obs_mode = "enriched"
    else:
        obs_mode = determine_obs_mode(expected_obs_dim)

    ##############################
    # Detect if split normalization was used during training
    ##############################
    use_split_normalization = (not saved_norm_obs) and (expected_obs_dim > 30)

    ##############################
    # Print configuration information
    ##############################
    if verbose:
        print(f"   Obs mode: {obs_mode} (vecnormalize dim={expected_obs_dim})")
        print(f"   VecNormalize saved with norm_obs={saved_norm_obs}")
        print(f"   → Using split normalization: {use_split_normalization}")

    use_enriched_stack = obs_mode in ["enriched", "enriched_extended"]

    def make_env():
        def _thunk():
            return build_env(
                bace_cfg,
                is_eval=True,
                log_dir_str=str(model_path.parent / "eval_monitor"),
                seed=eval_seed,
                use_enriched_obs=use_enriched_stack,
                ablation_config=ablation_config,
                max_episode_steps=14000,
                controlled_agent_id="agent_0",
                apply_split_normalization=use_split_normalization,
                freeze_schedule_at_step=FULLY_ANNEALED_STEP,
            )
        return _thunk

    venv = DummyVecEnv([make_env()])

    ##############################
    # Load saved split normalization stats onto the SplitNormalizationWrapper that build_env()
    # just constructed (only present when use_enriched_stack and use_split_normalization are
    # both true -- see apply_split_normalization in env_factory.build_env)
    ##############################
    if use_enriched_stack and use_split_normalization:
        split_norm_wrapper = get_split_norm_wrapper_from_vec_env(venv)
        if split_norm_wrapper is not None:
            ##############################
            # Load saved split normalization stats if available
            ##############################
            if split_norm_path and split_norm_path.exists():
                split_norm_wrapper.load(str(split_norm_path))
                split_norm_wrapper.training = False
                ##############################
                # Print information about loaded stats
                ##############################
                if verbose:
                    print(f"   ✓ Loaded SplitNormalization stats from {split_norm_path.name}")
                    print(f"     (count={split_norm_wrapper.count}, mean_range=[{split_norm_wrapper.obs_mean.min():.2f}, {split_norm_wrapper.obs_mean.max():.2f}])")
            else:
                split_norm_wrapper.training = False
                ##############################
                # Print warning if stats not found
                ##############################
                if verbose:
                    print(f"   ⚠️ WARNING: No SplitNormalization stats found - using identity normalization!")
                    print(f"     This may cause evaluation scores to differ significantly from training!")

    ##############################
    # Create VecNormalize with settings matching training
    ##############################
    if use_split_normalization:
        eval_env = VecNormalize(venv, norm_obs=False, norm_reward=False, clip_obs=10.0, training=False)
    else:
        eval_env = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    ##############################
    # Load saved normalization statistics
    ##############################
    eval_env = VecNormalize.load(vecnorm_path.as_posix(), eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    ##############################
    # Validate observation dimensions match
    ##############################
    actual_dim = eval_env.observation_space.shape[0]
    ##############################
    # Check if dimensions mismatch
    ##############################
    if actual_dim != expected_obs_dim:
        msg = f"MISMATCH: Actual obs dim ({actual_dim}) != vecnormalize dim ({expected_obs_dim})"
        ##############################
        # Raise error or print warning based on validation mode
        ##############################
        if strict_validation:
            raise ValueError(msg)
        else:
            print(f"   ⚠️ WARNING: {msg}")

    ##############################
    # Load trained policy model
    ##############################
    model = load_model_with_pt_fallback(model_path, eval_env, verbose=verbose)

    episodes: List[EpisodeData] = []

    eval_env.seed(eval_seed)
    obs = eval_env.reset()

    cur_reward = 0.0
    cur_length = 0
    ep_hvaa_destroyed = False
    ep_red_killed = False
    ep_blue_killed = False
    finished = 0

    ##############################
    # Run evaluation episodes
    ##############################
    while finished < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        step_out = eval_env.step(action)

        ##############################
        # Handle different return formats from environment
        ##############################
        if len(step_out) == 4:
            obs, reward, done, infos = step_out
        else:
            obs, reward, terminated, truncated, infos = step_out
            done = np.logical_or(terminated, truncated)

        info = infos[0] if isinstance(infos, (list, tuple)) else (infos if isinstance(infos, dict) else {})

        ##############################
        # Track if HVAA was destroyed this episode
        ##############################
        if _as_bool(info.get("hvaa_destroyed", False)) or not _as_bool(info.get("hvaa_alive", True)):
            ep_hvaa_destroyed = True

        ##############################
        # Track if red was killed this episode
        ##############################
        if (_as_int(info.get("red_kills_this_step", 0)) > 0 or
            _as_bool(info.get("red_killed_event", False)) or
            _as_int(info.get("red_killed_total", 0)) > 0):
            ep_red_killed = True

        ##############################
        # Track if blue was killed this episode
        ##############################
        if _as_bool(info.get("Blue_Killed", False)) or _as_bool(info.get("blue_killed", False)):
            ep_blue_killed = True

        cur_reward += float(reward[0])
        cur_length += 1

        ##############################
        # Process completed episode
        ##############################
        if bool(done[0]):
            episode = EpisodeData(
                reward=cur_reward,
                length=cur_length,
                hvaa_survived=not ep_hvaa_destroyed,
                red_killed=ep_red_killed,
                blue_killed=ep_blue_killed,
                blue_missiles_fired=_as_int(info.get("blue_missiles_fired", 0)),
                red_missiles_fired=_as_int(info.get("red_missiles_fired", 0)),
                blue_pk=float(info.get("blue_pk", 0.0)) if info.get("blue_pk") is not None else 0.0,
                red_pk=float(info.get("red_pk", 0.0)) if info.get("red_pk") is not None else 0.0,
                blue_shot_first=_as_bool(info.get("blue_shot_first", False)),
                red_shot_first=_as_bool(info.get("red_shot_first", False)),
                no_shots_fired=_as_bool(info.get("no_shots_fired", False)),
                min_separation_nm=float(info.get("min_separation_nm", -1.0)) if info.get("min_separation_nm") is not None else -1.0,
                end_reason=str(info.get("termination_reason", info.get("done_reason", ""))),
                training_seed=training_seed,
            )
            episodes.append(episode)
            finished += 1

            ##############################
            # Print progress periodically
            ##############################
            if verbose and (finished % 25 == 0 or finished == n_episodes):
                print(f"      {finished}/{n_episodes} episodes")

            cur_reward = 0.0
            cur_length = 0
            ep_hvaa_destroyed = False
            ep_red_killed = False
            ep_blue_killed = False

    eval_env.close()
    return episodes


def compute_seed_results(episodes: List[EpisodeData], seed_name: str,
                         model_path: Path, vecnorm_path: Path) -> SeedResults:
    """Compute summary statistics for a single seed."""
    n = len(episodes)

    return SeedResults(
        seed_name=seed_name,
        model_path=str(model_path),
        vecnorm_path=str(vecnorm_path),
        n_episodes=n,
        mean_reward=float(np.mean([ep.reward for ep in episodes])),
        std_reward=float(np.std([ep.reward for ep in episodes], ddof=1)) if n > 1 else 0.0,
        hvaa_survival_rate=float(np.mean([ep.hvaa_survived for ep in episodes])),
        red_kill_rate=float(np.mean([ep.red_killed for ep in episodes])),
        blue_loss_rate=float(np.mean([ep.blue_killed for ep in episodes])),
        mean_length=float(np.mean([ep.length for ep in episodes])),
        mean_blue_missiles=float(np.mean([ep.blue_missiles_fired for ep in episodes])),
        mean_red_missiles=float(np.mean([ep.red_missiles_fired for ep in episodes])),
        mean_blue_pk=float(np.mean([ep.blue_pk for ep in episodes])),
        mean_red_pk=float(np.mean([ep.red_pk for ep in episodes])),
        blue_shot_first_rate=float(np.mean([ep.blue_shot_first for ep in episodes])),
        red_shot_first_rate=float(np.mean([ep.red_shot_first for ep in episodes])),
        no_shots_rate=float(np.mean([ep.no_shots_fired for ep in episodes])),
        episodes=episodes,
    )


def compute_aggregate_results(seed_results: List[SeedResults]) -> Dict:
    """Compute aggregate statistics across training seeds."""
    n_seeds = len(seed_results)

    seed_mean_rewards = [sr.mean_reward for sr in seed_results]
    seed_hvaa_rates = [sr.hvaa_survival_rate for sr in seed_results]
    seed_red_kill_rates = [sr.red_kill_rate for sr in seed_results]
    seed_blue_loss_rates = [sr.blue_loss_rate for sr in seed_results]
    seed_mean_lengths = [sr.mean_length for sr in seed_results]
    seed_blue_missiles = [sr.mean_blue_missiles for sr in seed_results]
    seed_red_missiles = [sr.mean_red_missiles for sr in seed_results]
    seed_blue_pk = [sr.mean_blue_pk for sr in seed_results]
    seed_red_pk = [sr.mean_red_pk for sr in seed_results]
    seed_blue_first = [sr.blue_shot_first_rate for sr in seed_results]
    seed_red_first = [sr.red_shot_first_rate for sr in seed_results]
    seed_no_shots = [sr.no_shots_rate for sr in seed_results]

    return {
        "n_seeds": n_seeds,
        "episodes_per_seed": seed_results[0].n_episodes if seed_results else 0,
        "total_episodes": sum(sr.n_episodes for sr in seed_results),
        "cross_seed_statistics": {
            "mean_reward": asdict(t_distribution_ci(seed_mean_rewards)),
            "hvaa_survival_rate": asdict(t_distribution_ci(seed_hvaa_rates)),
            "red_kill_rate": asdict(t_distribution_ci(seed_red_kill_rates)),
            "blue_loss_rate": asdict(t_distribution_ci(seed_blue_loss_rates)),
            "mean_length": asdict(t_distribution_ci(seed_mean_lengths)),
            "mean_blue_missiles": asdict(t_distribution_ci(seed_blue_missiles)),
            "mean_red_missiles": asdict(t_distribution_ci(seed_red_missiles)),
            "mean_blue_pk": asdict(t_distribution_ci(seed_blue_pk)),
            "mean_red_pk": asdict(t_distribution_ci(seed_red_pk)),
            "blue_shot_first_rate": asdict(t_distribution_ci(seed_blue_first)),
            "red_shot_first_rate": asdict(t_distribution_ci(seed_red_first)),
            "no_shots_rate": asdict(t_distribution_ci(seed_no_shots)),
        },
    }


def print_results(seed_results: List[SeedResults], aggregate: Dict):
    """Print formatted results table."""
    n_seeds = len(seed_results)

    print("\n" + "="*90)
    print(f"{'SUPERLATIVE POLICY EVALUATION RESULTS':^90}")
    print("="*90)
    print(f"Training seeds evaluated: {n_seeds}")
    print(f"Episodes per seed: {aggregate['episodes_per_seed']}")
    print(f"Total episodes: {aggregate['total_episodes']}")
    print("="*90)

    print(f"\n{'PER-SEED RESULTS':^90}")
    print("-"*90)
    print(f"{'Seed':<12} {'Mean Reward':>14} {'Std':>10} {'HVAA %':>10} {'Red Kill %':>12} {'Blue Loss %':>12}")
    print("-"*90)

    ##############################
    # Print per-seed statistics
    ##############################
    for sr in sorted(seed_results, key=lambda x: x.seed_name):
        print(f"{sr.seed_name:<12} {sr.mean_reward:>14.2f} {sr.std_reward:>10.2f} "
              f"{sr.hvaa_survival_rate*100:>9.1f}% {sr.red_kill_rate*100:>11.1f}% "
              f"{sr.blue_loss_rate*100:>11.1f}%")

    print("="*90)

    print(f"\n{'CROSS-SEED AGGREGATE STATISTICS (95% CI)':^90}")
    print("-"*90)

    cs = aggregate["cross_seed_statistics"]

    def format_ci(metric_dict, is_rate=False):
        mean = metric_dict["mean"]
        ci_lower = metric_dict["ci_lower"]
        ci_upper = metric_dict["ci_upper"]
        hw = (ci_upper - ci_lower) / 2
        ##############################
        # Format as percentage or absolute value
        ##############################
        if is_rate:
            return f"{mean*100:.1f}% {PM} {hw*100:.1f}%  [{ci_lower*100:.1f}%, {ci_upper*100:.1f}%]"
        else:
            return f"{mean:.2f} {PM} {hw:.2f}  [{ci_lower:.2f}, {ci_upper:.2f}]"

    print(f"{'Metric':<30} {'Mean ± HW':>25} {'95% CI':>25}")
    print("-"*90)
    print(f"{'Mean Reward':<30} {format_ci(cs['mean_reward'])}")
    print(f"{'HVAA Survival Rate':<30} {format_ci(cs['hvaa_survival_rate'], is_rate=True)}")
    print(f"{'Red Kill Rate':<30} {format_ci(cs['red_kill_rate'], is_rate=True)}")
    print(f"{'Blue Loss Rate':<30} {format_ci(cs['blue_loss_rate'], is_rate=True)}")
    print(f"{'Mean Episode Length':<30} {format_ci(cs['mean_length'])}")
    print("="*90)


def build_feature_config(args) -> Dict[str, Any]:
    """Build feature configuration from command line arguments."""
    config = {
        'pursuer_speed': 1.0,
        'evader_speed': 1.0,
        'capture_radius': 0.01,
        'pursuer_range': 0.5,
        'normalize_distance': 1.0,
    }

    ##############################
    # Define feature flag options
    ##############################
    feature_flags = [
        'enable_apollonius', 'enable_bez', 'enable_dmc',
        'enable_multi_threat', 'enable_hvaa_escort',
        'enable_offense_wez', 'enable_offense_ttc', 'enable_reward_shaping',
    ]

    ##############################
    # Process each feature flag
    ##############################
    for flag in feature_flags:
        val = getattr(args, flag, None)
        ##############################
        # Use argument value if provided
        ##############################
        if val is not None:
            config[flag] = val
        else:
            ##############################
            # Apply default values
            ##############################
            if flag in ['enable_apollonius', 'enable_bez', 'enable_dmc',
                       'enable_offense_wez', 'enable_offense_ttc']:
                config[flag] = True
            else:
                config[flag] = False

    return config


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate policies across seeds (supports timestep checkpoints)")

    p.add_argument("--experiment-dir", type=str, required=True,
                   help="Path to experiment directory containing seed folders")
    p.add_argument("--seeds", type=str, default=None,
                   help="Comma-separated list of seeds to evaluate (default: all)")
    p.add_argument("--episodes", type=int, default=100,
                   help="Number of episodes per seed")
    p.add_argument("--eval-seed", type=int, default=42,
                   help="Random seed for evaluation")

    p.add_argument("--timesteps", type=str, default=None,
                   help="Comma-separated list of timesteps to evaluate (e.g., '400000,5800000,9400000'). "
                        "Loads policy/vecnormalize/split_norm from eval_checkpoints/ at each timestep. "
                        "If not specified, evaluates the best (superlative) checkpoint.")
    p.add_argument("--list-timesteps", action="store_true", default=False,
                   help="List available timestep checkpoints for each seed and exit")

    p.add_argument("--obs-mode", type=str, default="auto",
                   choices=["auto", "baseline", "enriched", "enriched_extended"],
                   help="Observation mode (default: auto-detect from vecnormalize)")
    p.add_argument("--use-enriched-obs", action="store_true", default=False,
                   help="Use enriched observations")
    p.add_argument("--ablation-config", type=str, default=None,
                   choices=['all', 'none', 'geometry_only', 'engagement_only',
                           'range_limited_only', 'geometry_engagement',
                           'geometry_range', 'engagement_range'],
                   help="Ablation configuration for enriched observations")

    feature_group = p.add_argument_group("Feature Flags")
    feature_group.add_argument("--enable-apollonius", action="store_true", dest="enable_apollonius", default=None)
    feature_group.add_argument("--no-enable-apollonius", action="store_false", dest="enable_apollonius")
    feature_group.add_argument("--enable-bez", action="store_true", dest="enable_bez", default=None)
    feature_group.add_argument("--no-enable-bez", action="store_false", dest="enable_bez")
    feature_group.add_argument("--enable-dmc", action="store_true", dest="enable_dmc", default=None)
    feature_group.add_argument("--no-enable-dmc", action="store_false", dest="enable_dmc")
    feature_group.add_argument("--enable-multi-threat", action="store_true", dest="enable_multi_threat", default=None)
    feature_group.add_argument("--no-enable-multi-threat", action="store_false", dest="enable_multi_threat")
    feature_group.add_argument("--enable-hvaa-escort", action="store_true", dest="enable_hvaa_escort", default=None)
    feature_group.add_argument("--no-enable-hvaa-escort", action="store_false", dest="enable_hvaa_escort")
    feature_group.add_argument("--enable-offense-wez", action="store_true", dest="enable_offense_wez", default=None)
    feature_group.add_argument("--no-enable-offense-wez", action="store_false", dest="enable_offense_wez")
    feature_group.add_argument("--enable-offense-ttc", action="store_true", dest="enable_offense_ttc", default=None)
    feature_group.add_argument("--no-enable-offense-ttc", action="store_false", dest="enable_offense_ttc")
    feature_group.add_argument("--enable-reward-shaping", action="store_true", dest="enable_reward_shaping", default=None)
    feature_group.add_argument("--no-enable-reward-shaping", action="store_false", dest="enable_reward_shaping")

    p.add_argument("--config", type=str, default=None,
                   help="B-ACE config path (default: auto-detect from repo)")

    p.add_argument("--renderize", type=int, default=0)
    p.add_argument("--speed-up", type=int, default=200)

    p.add_argument("--output-file", type=str, default=None,
                   help="Output JSON path (default: <experiment_dir>/superlative_eval_results.json)")

    p.add_argument("--no-strict-validation", action="store_true", default=False,
                   help="Continue even if observation dimension doesn't match")

    return p.parse_args()


def print_timestep_results(timestep_results: Dict[int, Tuple[List[SeedResults], Dict]], timesteps: List[int]):
    """Print formatted results table for crawl-walk-run timestep evaluation."""
    n_timesteps = len(timesteps)

    print("\n" + "="*100)
    print(f"{'CRAWL → WALK → RUN  CHECKPOINT EVALUATION':^100}")
    print("="*100)
    print(f"Timesteps evaluated: {n_timesteps}")
    ##############################
    # Print information for each timestep
    ##############################
    for i, ts in enumerate(timesteps):
        phase_label = _timestep_phase_label(i, n_timesteps)
        n_seeds = len(timestep_results[ts][0])
        n_eps = timestep_results[ts][1].get("total_episodes", 0)
        print(f"   {phase_label:<8} timestep {ts:>12,}  ({n_seeds} seeds, {n_eps} total episodes)")
    print("="*100)

    print(f"\n{'PROGRESSION SUMMARY (cross-seed means)':^100}")
    print("-"*100)
    print(f"{'Phase':<8} {'Timestep':>12} {'Mean Reward':>14} {'HVAA %':>10} "
          f"{'Red Kill %':>12} {'Blue Loss %':>12} {'Ep Length':>12}")
    print("-"*100)

    ##############################
    # Print summary row for each timestep
    ##############################
    for i, ts in enumerate(timesteps):
        seed_results, agg = timestep_results[ts]
        cs = agg["cross_seed_statistics"]
        phase = _timestep_phase_label(i, n_timesteps)
        print(f"{phase:<8} {ts:>12,} {cs['mean_reward']['mean']:>14.2f} "
              f"{cs['hvaa_survival_rate']['mean']*100:>9.1f}% "
              f"{cs['red_kill_rate']['mean']*100:>11.1f}% "
              f"{cs['blue_loss_rate']['mean']*100:>11.1f}% "
              f"{cs['mean_length']['mean']:>12.1f}")

    print("="*100)

    ##############################
    # Print detailed results for each timestep
    ##############################
    for i, ts in enumerate(timesteps):
        seed_results, agg = timestep_results[ts]
        phase = _timestep_phase_label(i, n_timesteps)
        print(f"\n{'─'*100}")
        print(f"  {phase.upper()} PHASE — Timestep {ts:,}")
        print(f"{'─'*100}")
        print_results(seed_results, agg)


def _timestep_phase_label(index: int, total: int) -> str:
    """Assign a crawl/walk/run phase label based on position in the list."""
    ##############################
    # Handle single timestep case
    ##############################
    if total <= 1:
        return "eval"
    ##############################
    # Handle two timesteps as crawl and run
    ##############################
    if total == 2:
        return ["crawl", "run"][index]
    ##############################
    # Handle three timesteps as crawl, walk, run
    ##############################
    if total == 3:
        return ["crawl", "walk", "run"][index]
    ##############################
    # Handle more than three timesteps with intermediate walk phases
    ##############################
    if index == 0:
        return "crawl"
    ##############################
    # Label final timestep as run
    ##############################
    if index == total - 1:
        return "run"
    return f"walk_{index}"


def main():
    args = parse_args()

    ##############################
    # Same reasoning as train_bace_clean.py's --torch-threads: single-environment, small-MLP
    # inference doesn't benefit from PyTorch's default multi-threaded CPU ops, and those threads
    # otherwise compete with the Godot subprocess for the same physical cores.
    ##############################
    torch.set_num_threads(1)

    experiment_dir = Path(args.experiment_dir).expanduser().resolve()
    ##############################
    # Check if experiment directory exists
    ##############################
    if not experiment_dir.exists():
        raise SystemExit(f"Directory not found: {experiment_dir}")

    ##############################
    # Handle --list-timesteps mode
    ##############################
    if args.list_timesteps:
        print(f"\nAvailable timestep checkpoints in: {experiment_dir}\n")
        available = discover_available_timesteps(experiment_dir)
        ##############################
        # Print message if no checkpoints found
        ##############################
        if not available:
            print("  No eval_checkpoints/ found in any seed directory.")
        ##############################
        # Print available timesteps for each seed
        ##############################
        for seed_name, ts_list in sorted(available.items()):
            formatted = ", ".join(f"{t:,}" for t in ts_list)
            print(f"  {seed_name}: [{formatted}]")
        ##############################
        # Show intersection of timesteps across all seeds
        ##############################
        if len(available) > 1:
            common = sorted(set.intersection(*(set(v) for v in available.values())))
            ##############################
            # Print common timesteps
            ##############################
            if common:
                formatted = ", ".join(f"{t:,}" for t in common)
                print(f"\n  Common across all seeds: [{formatted}]")
            else:
                print(f"\n  ⚠ No timesteps are common across all seeds.")
        return

    ##############################
    # Determine evaluation mode
    ##############################
    use_timestep_mode = args.timesteps is not None

    ##############################
    # Parse requested timesteps if in timestep mode
    ##############################
    if use_timestep_mode:
        requested_timesteps = sorted(int(t.strip()) for t in args.timesteps.split(","))
        mode_label = "CHECKPOINT EVALUATION (CRAWL → WALK → RUN)"
    else:
        requested_timesteps = []
        mode_label = "SUPERLATIVE POLICY EVALUATION (FIXED FOR SPLIT NORMALIZATION)"

    print("\n" + "="*90)
    print(mode_label)
    print("="*90)
    print(f"Experiment directory: {experiment_dir}")
    print(f"Episodes per seed: {args.episodes}")
    print(f"Evaluation seed: {args.eval_seed}")
    ##############################
    # Print timesteps if in timestep mode
    ##############################
    if use_timestep_mode:
        formatted_ts = ", ".join(f"{t:,}" for t in requested_timesteps)
        print(f"Timesteps: [{formatted_ts}]")
    print("="*90 + "\n")

    ##############################
    # Load B-ACE configuration
    ##############################
    if args.config:
        config_path = Path(args.config).expanduser().resolve()
    else:
        config_path = REPO_ROOT / "b_ace_py" / "SimpleExample_B_ACE_config.json"

    bace_cfg = load_b_ace_config(config_path.as_posix())
    ##############################
    # Check if configuration loaded successfully
    ##############################
    if bace_cfg is None:
        raise RuntimeError(f"Failed to load config: {config_path}")

    env_cfg = bace_cfg.setdefault("EnvConfig", {})
    ##############################
    # Set environment path if not already set
    ##############################
    if not env_cfg.get("env_path"):
        env_cfg["env_path"] = _resolve_env_path()

    env_path = Path(env_cfg["env_path"]).expanduser()
    ##############################
    # Make relative paths absolute
    ##############################
    if not env_path.is_absolute():
        env_path = (REPO_ROOT / env_path).resolve()
    ##############################
    # Verify environment path exists
    ##############################
    if not env_path.exists():
        env_cfg["env_path"] = _resolve_env_path()
    else:
        env_cfg["env_path"] = env_path.as_posix()

    env_cfg["renderize"] = args.renderize
    env_cfg["speed_up"] = args.speed_up

    agents_cfg = bace_cfg.setdefault("AgentsConfig", {})
    agents_cfg.setdefault("blue_agents", {})["base_behavior"] = "external"
    agents_cfg.setdefault("red_agents", {})["base_behavior"] = "baseline1"

    feature_config = build_feature_config(args)

    print(f"\nObservation Configuration:")
    ##############################
    # Print observation mode configuration
    ##############################
    if args.obs_mode != "auto":
        print(f"   Mode: {args.obs_mode} (explicit)")
    elif args.use_enriched_obs:
        print(f"   Mode: enriched (--use-enriched-obs flag)")
    else:
        print(f"   Mode: auto-detect from vecnormalize")

    ##############################
    # Print ablation config if using enriched observations
    ##############################
    if args.use_enriched_obs or args.obs_mode in ["enriched", "enriched_extended"]:
        ##############################
        # Check if ablation config is specified
        ##############################
        if args.ablation_config:
            print(f"   Ablation config: '{args.ablation_config}'")

    ##############################
    # TIMESTEP MODE: evaluate at each training checkpoint
    ##############################
    if use_timestep_mode:
        ##############################
        # Check which timesteps are available
        ##############################
        available = discover_available_timesteps(experiment_dir)
        ##############################
        # Error if no checkpoints found
        ##############################
        if not available:
            raise SystemExit("No eval_checkpoints/ found in any seed directory.")

        ##############################
        # Filter seeds if --seeds was passed
        ##############################
        if args.seeds:
            requested_seeds = [s.strip() for s in args.seeds.split(',')]
            available = {k: v for k, v in available.items() if k in requested_seeds}
            ##############################
            # Error if none of requested seeds found
            ##############################
            if not available:
                raise SystemExit(f"None of the requested seeds found: {requested_seeds}")

        ##############################
        # Warn about missing timesteps
        ##############################
        all_available_ts = set()
        ##############################
        # Collect all timesteps across seeds
        ##############################
        for ts_list in available.values():
            all_available_ts.update(ts_list)

        ##############################
        # Warn about each requested timestep not found
        ##############################
        for ts in requested_timesteps:
            ##############################
            # Check if timestep is available
            ##############################
            if ts not in all_available_ts:
                print(f"⚠ Warning: Timestep {ts:,} not found in any seed")

        ##############################
        # Evaluate each timestep
        ##############################
        timestep_results: Dict[int, Tuple[List[SeedResults], Dict]] = {}

        ##############################
        # Process each requested timestep
        ##############################
        for ts_idx, timestep in enumerate(requested_timesteps):
            phase = _timestep_phase_label(ts_idx, len(requested_timesteps))
            print(f"\n{'='*90}")
            print(f"  {phase.upper()} PHASE — Timestep {timestep:,}")
            print(f"{'='*90}")

            seed_checkpoints = find_seed_checkpoints_by_timestep(experiment_dir, timestep)

            ##############################
            # Apply seed filter
            ##############################
            if args.seeds:
                seed_checkpoints = {k: v for k, v in seed_checkpoints.items() if k in requested_seeds}

            ##############################
            # Skip timestep if no seeds found
            ##############################
            if not seed_checkpoints:
                print(f"   ⚠ No seeds have a checkpoint at timestep {timestep:,} – skipping")
                continue

            print(f"\n   Evaluating {len(seed_checkpoints)} seeds at timestep {timestep:,}: "
                  f"{sorted(seed_checkpoints.keys())}")

            ts_seed_results: List[SeedResults] = []

            ##############################
            # Evaluate each seed at this timestep
            ##############################
            for i, (seed_name, (policy_path, vecnorm_path, split_norm_path)) in enumerate(sorted(seed_checkpoints.items())):
                print(f"\n   [{i+1}/{len(seed_checkpoints)}] {seed_name} @ timestep {timestep:,}")
                print(f"      Policy: {policy_path}")
                print(f"      VecNormalize: {vecnorm_path}")
                ##############################
                # Print split normalization info if available
                ##############################
                if split_norm_path:
                    print(f"      SplitNormalization: {split_norm_path}")
                elif args.use_enriched_obs:
                    print(f"      ⚠️ SplitNormalization: NOT FOUND (will use identity normalization)")

                episodes = evaluate_model(
                    model_path=policy_path,
                    vecnorm_path=vecnorm_path,
                    bace_cfg=bace_cfg,
                    n_episodes=args.episodes,
                    training_seed=seed_name,
                    eval_seed=args.eval_seed,
                    obs_mode_override=args.obs_mode,
                    use_enriched_obs=args.use_enriched_obs,
                    feature_config=feature_config,
                    ablation_config=args.ablation_config,
                    verbose=True,
                    strict_validation=not args.no_strict_validation,
                    split_norm_path=split_norm_path,
                )

                seed_result = compute_seed_results(episodes, seed_name, policy_path, vecnorm_path)
                ts_seed_results.append(seed_result)

                print(f"      {seed_name} summary: reward={seed_result.mean_reward:.2f} {PM} {seed_result.std_reward:.2f}, "
                      f"HVAA={seed_result.hvaa_survival_rate*100:.1f}%, Red Kill={seed_result.red_kill_rate*100:.1f}%")

            ##############################
            # Compute aggregate results for this timestep
            ##############################
            if ts_seed_results:
                ts_aggregate = compute_aggregate_results(ts_seed_results)
                timestep_results[timestep] = (ts_seed_results, ts_aggregate)

        ##############################
        # Print crawl-walk-run comparison
        ##############################
        evaluated_timesteps = [ts for ts in requested_timesteps if ts in timestep_results]
        ##############################
        # Print results only if some timesteps were evaluated
        ##############################
        if evaluated_timesteps:
            print_timestep_results(timestep_results, evaluated_timesteps)

        ##############################
        # Save JSON output for timestep mode
        ##############################
        output_data = {
            "experiment_dir": str(experiment_dir),
            "evaluation_mode": "timestep_checkpoints",
            "evaluation_config": {
                "timesteps": requested_timesteps,
                "episodes_per_seed": args.episodes,
                "eval_seed": args.eval_seed,
                "config_path": str(config_path),
                "obs_mode": args.obs_mode,
                "use_enriched_obs": args.use_enriched_obs,
                "ablation_config": args.ablation_config,
                "feature_config": feature_config,
            },
            "per_timestep_results": {},
        }

        ##############################
        # Add per-timestep results to output
        ##############################
        for ts in evaluated_timesteps:
            seed_results, agg = timestep_results[ts]
            phase = _timestep_phase_label(evaluated_timesteps.index(ts), len(evaluated_timesteps))
            output_data["per_timestep_results"][str(ts)] = {
                "phase_label": phase,
                "aggregate": agg,
                "per_seed_results": {
                    sr.seed_name: {
                        "model_path": sr.model_path,
                        "vecnorm_path": sr.vecnorm_path,
                        "n_episodes": sr.n_episodes,
                        "mean_reward": sr.mean_reward,
                        "std_reward": sr.std_reward,
                        "hvaa_survival_rate": sr.hvaa_survival_rate,
                        "red_kill_rate": sr.red_kill_rate,
                        "blue_loss_rate": sr.blue_loss_rate,
                        "mean_length": sr.mean_length,
                        "mean_blue_missiles": sr.mean_blue_missiles,
                        "mean_red_missiles": sr.mean_red_missiles,
                        "mean_blue_pk": sr.mean_blue_pk,
                        "mean_red_pk": sr.mean_red_pk,
                        "blue_shot_first_rate": sr.blue_shot_first_rate,
                        "red_shot_first_rate": sr.red_shot_first_rate,
                        "no_shots_rate": sr.no_shots_rate,
                        "episodes": [asdict(ep) for ep in sr.episodes],
                    }
                    ##############################
                    # Iterate over seed results
                    ##############################
                    for sr in seed_results
                },
            }

        ##############################
        # Determine output file path
        ##############################
        if args.output_file:
            out_path = Path(args.output_file).expanduser().resolve()
        else:
            ts_label = "_".join(str(t) for t in evaluated_timesteps)
            out_path = experiment_dir / f"checkpoint_eval_{ts_label}.json"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        ##############################
        # Write results to JSON file
        ##############################
        with open(out_path, 'w') as f:
            json.dump(_jsonify(output_data), f, indent=2)
        print(f"\nResults saved to: {out_path}")

        return

    ##############################
    # SUPERLATIVE MODE: evaluate best checkpoint
    ##############################
    seed_models = find_seed_models(experiment_dir)
    ##############################
    # Error if no seed models found
    ##############################
    if not seed_models:
        raise SystemExit(f"No seed models found in {experiment_dir}")

    ##############################
    # Filter seeds if requested
    ##############################
    if args.seeds:
        requested_seeds = [s.strip() for s in args.seeds.split(',')]
        filtered_models = {k: v for k, v in seed_models.items() if k in requested_seeds}

        found_seeds = set(filtered_models.keys())
        missing_seeds = set(requested_seeds) - found_seeds
        ##############################
        # Warn about missing seeds
        ##############################
        if missing_seeds:
            print(f"Warning: Seeds not found: {missing_seeds}")

        ##############################
        # Error if no requested seeds found
        ##############################
        if not filtered_models:
            raise SystemExit(f"None of the requested seeds found: {requested_seeds}")

        seed_models = filtered_models
        print(f"\nFiltered to seeds: {list(seed_models.keys())}")

    print(f"\nWill evaluate {len(seed_models)} seeds: {sorted(seed_models.keys())}")

    all_seed_results: List[SeedResults] = []

    ##############################
    # Evaluate each seed
    ##############################
    for i, (seed_name, (model_path, vecnorm_path, split_norm_path)) in enumerate(sorted(seed_models.items())):
        print(f"\n[{i+1}/{len(seed_models)}] Evaluating {seed_name}...")
        print(f"   Model: {model_path}")
        print(f"   VecNormalize: {vecnorm_path}")
        ##############################
        # Print split normalization info if available
        ##############################
        if split_norm_path:
            print(f"   SplitNormalization: {split_norm_path}")
        elif args.use_enriched_obs:
            print(f"   ⚠️ SplitNormalization: NOT FOUND (will use identity normalization)")

        episodes = evaluate_model(
            model_path=model_path,
            vecnorm_path=vecnorm_path,
            bace_cfg=bace_cfg,
            n_episodes=args.episodes,
            training_seed=seed_name,
            eval_seed=args.eval_seed,
            obs_mode_override=args.obs_mode,
            use_enriched_obs=args.use_enriched_obs,
            feature_config=feature_config,
            ablation_config=args.ablation_config,
            verbose=True,
            strict_validation=not args.no_strict_validation,
            split_norm_path=split_norm_path,
        )

        seed_result = compute_seed_results(episodes, seed_name, model_path, vecnorm_path)
        all_seed_results.append(seed_result)

        print(f"   {seed_name} summary: reward={seed_result.mean_reward:.2f} {PM} {seed_result.std_reward:.2f}, "
              f"HVAA={seed_result.hvaa_survival_rate*100:.1f}%, Red Kill={seed_result.red_kill_rate*100:.1f}%")

    aggregate = compute_aggregate_results(all_seed_results)
    print_results(all_seed_results, aggregate)

    ##############################
    # Prepare output data for superlative mode
    ##############################
    output_data = {
        "experiment_dir": str(experiment_dir),
        "evaluation_mode": "superlative",
        "evaluation_config": {
            "episodes_per_seed": args.episodes,
            "eval_seed": args.eval_seed,
            "config_path": str(config_path),
            "obs_mode": args.obs_mode,
            "use_enriched_obs": args.use_enriched_obs,
            "ablation_config": args.ablation_config,
            "feature_config": feature_config,
        },
        "aggregate": aggregate,
        "per_seed_results": {
            sr.seed_name: {
                "model_path": sr.model_path,
                "vecnorm_path": sr.vecnorm_path,
                "n_episodes": sr.n_episodes,
                "mean_reward": sr.mean_reward,
                "std_reward": sr.std_reward,
                "hvaa_survival_rate": sr.hvaa_survival_rate,
                "red_kill_rate": sr.red_kill_rate,
                "blue_loss_rate": sr.blue_loss_rate,
                "mean_length": sr.mean_length,
                "mean_blue_missiles": sr.mean_blue_missiles,
                "mean_red_missiles": sr.mean_red_missiles,
                "mean_blue_pk": sr.mean_blue_pk,
                "mean_red_pk": sr.mean_red_pk,
                "blue_shot_first_rate": sr.blue_shot_first_rate,
                "red_shot_first_rate": sr.red_shot_first_rate,
                "no_shots_rate": sr.no_shots_rate,
                "episodes": [asdict(ep) for ep in sr.episodes],
            }
            ##############################
            # Iterate over all seed results
            ##############################
            for sr in all_seed_results
        },
    }

    ##############################
    # Determine output file path
    ##############################
    if args.output_file:
        out_path = Path(args.output_file).expanduser().resolve()
    else:
        out_path = experiment_dir / "superlative_eval_results.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    ##############################
    # Write results to JSON file
    ##############################
    with open(out_path, 'w') as f:
        json.dump(_jsonify(output_data), f, indent=2)

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
