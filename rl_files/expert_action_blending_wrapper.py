#!/usr/bin/env python3
"""
This file implements a gymnasium wrapper that blends the PPO agent's action with a
rule-based expert's action before execution. Unlike the expert alignment wrapper (which
only uses the expert for reward computation), this wrapper actually modifies the action
that gets sent to the environment using the formula: blended = alpha * expert + (1 - alpha)
* ppo. The alpha parameter decays over training, creating a behavioral cloning handoff
where the expert dominates early and the PPO policy gradually takes full control.

Within the training pipeline, this wrapper is optionally applied when the --expert-blending
flag is passed to train_bace_clean.py. It wraps the environment so that the agent
experiences expert-quality trajectories early in training while still learning from its own
reward signals. The blend_dims parameter controls which action dimensions are blended
(typically turn, altitude, and g-force but not fire), and fire_mode provides several
strategies for handling the discrete fire action separately.

Key configurable parameters: The alpha decay function (created by create_alpha_decay_fn)
takes initial_alpha (default: 0.8), final_alpha (default: 0.05), decay_start (default:
500,000), and decay_end (default: 5,000,000) timesteps. blend_dims specifies which action
indices to blend (default: [0, 1, 2] for turn, altitude, g_force). fire_mode can be 'ppo'
(always use agent's fire), 'expert', 'blend', 'gate' (expert when alpha > 0.3), or 'or'
(fire if either wants to). decay_type supports 'linear', 'exponential', or 'cosine'.
"""

import numpy as np
import gymnasium as gym
from typing import Dict, Any, Optional, Callable, List


class ExpertActionBlendingWrapper(gym.Wrapper):
    """
    Blends agent action with expert action before execution.
    
    The BLENDED action is executed, not the agent's raw action.
    
    This approach:
    - Provides expert-quality trajectories early in training
    - Gradually hands control to PPO as alpha decays
    - Creates smooth curriculum from imitation to reinforcement learning
    
    Args:
        env: The environment to wrap
        expert: Expert policy with get_action(obs, obs_indices) method
        obs_indices: Dictionary mapping observation names to indices for the expert
        alpha_fn: Function(global_step) -> alpha coefficient [0, 1]
                  alpha=1 means 100% expert, alpha=0 means 100% PPO
        blend_dims: List of action dimensions to blend (default: [0, 1, 2] for turn, alt, g)
                    Fire (dim 3) is often handled separately
        fire_blend_mode: How to handle fire action:
                         'ppo' - always use PPO's fire decision
                         'expert' - always use expert's fire decision  
                         'blend' - blend like other dimensions
                         'gate' - use expert fire when alpha > threshold
        fire_gate_threshold: For 'gate' mode, use expert fire if alpha > this
        debug: Enable debug printing
    """
    
    def __init__(
        self,
        env: gym.Env,
        expert: Any,
        obs_indices: Dict[str, int],
        alpha_fn: Callable[[int], float],
        blend_dims: Optional[List[int]] = None,
        fire_blend_mode: str = 'gate',
        fire_gate_threshold: float = 0.3,
        debug: bool = False,
    ):
        super().__init__(env)
        self.expert = expert
        self.obs_indices = obs_indices
        self.alpha_fn = alpha_fn
        self.fire_blend_mode = fire_blend_mode
        self.fire_gate_threshold = fire_gate_threshold
        self.debug = debug

        ##############################
        # Check if blend_dims was provided, use default if not
        ##############################
        if blend_dims is None:
            self.blend_dims = [0, 1, 2]
        else:
            self.blend_dims = list(blend_dims)

        self._current_alpha = 1.0

        self._last_expert_action = None
        self._last_ppo_action = None
        self._last_blended_action = None
        self._episode_expert_contribution = 0.0
        self._episode_steps = 0
        self._total_steps = 0
        self._global_step = 0

        self._last_original_obs = None
        
    def set_global_step(self, t: int):
        """Update global training step (for alpha scheduling)."""
        self._global_step = int(t)
        self._current_alpha = self.alpha_fn(self._global_step)

        ##############################
        # If wrapped environment supports global step updates, propagate it
        ##############################
        if hasattr(self.env, "set_global_step"):
            self.env.set_global_step(t)
    
    def reset(self, **kwargs):
        """Reset environment and expert state."""
        obs, info = self.env.reset(**kwargs)

        ##############################
        # If expert policy has state to reset (e.g., memory), call its reset method
        ##############################
        if hasattr(self.expert, "reset"):
            self.expert.reset()

        self._episode_expert_contribution = 0.0
        self._episode_steps = 0

        self._cache_original_obs()

        return obs, info
    
    def step(self, action: np.ndarray):
        """
        Blend agent's action with expert's action, then execute.

        The BLENDED action is what gets executed in the environment.
        """
        alpha = self._current_alpha

        obs_for_expert = self._get_obs_for_expert()

        expert_action = self.expert.get_action(obs_for_expert, self.obs_indices)
        self._last_expert_action = expert_action.copy()
        self._last_ppo_action = np.array(action).copy()

        blended_action = self._blend_actions(action, expert_action, alpha)
        self._last_blended_action = blended_action.copy()

        obs, reward, terminated, truncated, info = self.env.step(blended_action)

        self._cache_original_obs()

        self._episode_expert_contribution += alpha
        self._episode_steps += 1
        self._total_steps += 1

        info["blend_alpha"] = float(alpha)
        info["expert_action"] = expert_action.tolist()
        info["ppo_action"] = self._last_ppo_action.tolist()
        info["blended_action"] = blended_action.tolist()

        divergence = np.mean(np.abs(self._last_ppo_action - expert_action))
        info["ppo_expert_divergence"] = float(divergence)

        ##############################
        # If debug mode is enabled and we've completed a logging interval, print action details
        ##############################
        if self.debug and self._total_steps % 500 == 0:
            print(f"[BLEND] step={self._total_steps} alpha={alpha:.3f} "
                  f"ppo=[{action[0]:.2f},{action[1]:.2f},{action[2]:.2f},{action[3]:.1f}] "
                  f"expert=[{expert_action[0]:.2f},{expert_action[1]:.2f},{expert_action[2]:.2f},{expert_action[3]:.1f}] "
                  f"blended=[{blended_action[0]:.2f},{blended_action[1]:.2f},{blended_action[2]:.2f},{blended_action[3]:.1f}]")

        return obs, reward, terminated, truncated, info
    
    def _blend_actions(
        self,
        ppo_action: np.ndarray,
        expert_action: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        """
        Blend PPO and expert actions.

        blended = alpha * expert + (1 - alpha) * ppo

        Args:
            ppo_action: Action from PPO policy
            expert_action: Action from expert policy
            alpha: Blending coefficient (1 = all expert, 0 = all PPO)

        Returns:
            Blended action
        """
        ppo_a = np.asarray(ppo_action, dtype=np.float32).copy()
        expert_a = np.asarray(expert_action, dtype=np.float32).copy()

        blended = ppo_a.copy()

        ##############################
        # For each dimension in blend_dims, interpolate between expert and PPO actions
        ##############################
        for dim in self.blend_dims:
            ##############################
            # Check if dimension is valid in both action arrays
            ##############################
            if dim < len(blended) and dim < len(expert_a):
                blended[dim] = alpha * expert_a[dim] + (1 - alpha) * ppo_a[dim]

        fire_dim = 3
        ##############################
        # If fire dimension exists in the action, apply fire blending strategy
        ##############################
        if fire_dim < len(blended):
            ##############################
            # Determine which fire action to use based on blend mode
            ##############################
            if self.fire_blend_mode == 'ppo':
                pass
            ##############################
            # Use expert fire decision
            ##############################
            elif self.fire_blend_mode == 'expert':
                blended[fire_dim] = expert_a[fire_dim]
            ##############################
            # Blend fire like other continuous actions
            ##############################
            elif self.fire_blend_mode == 'blend':
                blended[fire_dim] = alpha * expert_a[fire_dim] + (1 - alpha) * ppo_a[fire_dim]
            ##############################
            # Use gating logic: expert fire if alpha above threshold, else PPO
            ##############################
            elif self.fire_blend_mode == 'gate':
                ##############################
                # If expert influence is strong, prioritize expert fire decision
                ##############################
                if alpha > self.fire_gate_threshold:
                    blended[fire_dim] = expert_a[fire_dim]
            ##############################
            # Fire if either expert or PPO wants to fire
            ##############################
            elif self.fire_blend_mode == 'or':
                ##############################
                # Check if either action has fire signal above threshold
                ##############################
                if expert_a[fire_dim] > 0.5 or ppo_a[fire_dim] > 0.5:
                    blended[fire_dim] = 1.0
                else:
                    blended[fire_dim] = -1.0

        blended = np.clip(blended, -1.0, 1.0)

        return blended
    
    def _flatten_obs_dict(self, obs_dict: dict) -> np.ndarray:
        """Recursively flatten observation dict to array."""
        flat = []
        ##############################
        # Iterate through all values in observation dictionary
        ##############################
        for v in obs_dict.values():
            ##############################
            # Handle numpy array values by flattening them
            ##############################
            if isinstance(v, np.ndarray):
                flat.extend(v.flatten().tolist())
            ##############################
            # Handle list or tuple values by converting to floats
            ##############################
            elif isinstance(v, (list, tuple)):
                flat.extend([float(x) for x in v])
            ##############################
            # Handle nested dictionaries by recursively flattening
            ##############################
            elif isinstance(v, dict):
                flat.extend(self._flatten_obs_dict(v).tolist())
            ##############################
            # Handle scalar numeric values
            ##############################
            elif isinstance(v, (int, float)):
                flat.append(float(v))
        return np.asarray(flat, dtype=np.float32)

    def _cache_original_obs(self):
        """Cache the original observation from the base environment."""
        e = self.env
        ##############################
        # Walk up the wrapper chain to find observation data
        ##############################
        while hasattr(e, 'env'):
            ##############################
            # Check for cached observation array in current wrapper
            ##############################
            if hasattr(e, '_last_obs') and e._last_obs is not None:
                ##############################
                # If observation is numpy array, cache it and return
                ##############################
                if isinstance(e._last_obs, np.ndarray):
                    self._last_original_obs = np.asarray(e._last_obs, dtype=np.float32)
                    return
            ##############################
            # Check for observation dictionary in current wrapper
            ##############################
            if hasattr(e, '_last_obs_dict'):
                agent_id = getattr(e, 'controlled_agent_id', 'agent_0')
                ##############################
                # If controlled agent ID exists in observation dict
                ##############################
                if agent_id in e._last_obs_dict:
                    obs = e._last_obs_dict[agent_id]
                    ##############################
                    # Handle observation if it is a numpy array
                    ##############################
                    if isinstance(obs, np.ndarray):
                        self._last_original_obs = np.asarray(obs, dtype=np.float32)
                        return
                    ##############################
                    # Handle observation if it is a dictionary
                    ##############################
                    elif isinstance(obs, dict):
                        ##############################
                        # Check for 'observation' key with numpy array
                        ##############################
                        if 'observation' in obs and isinstance(obs['observation'], np.ndarray):
                            self._last_original_obs = np.asarray(obs['observation'], dtype=np.float32)
                        else:
                            self._last_original_obs = self._flatten_obs_dict(obs)
                        return
            e = e.env

        ##############################
        # Check innermost environment for observation dictionary
        ##############################
        if hasattr(e, '_last_obs_dict'):
            agent_id = getattr(e, 'controlled_agent_id', 'agent_0')
            ##############################
            # If controlled agent ID exists in observation dict
            ##############################
            if agent_id in e._last_obs_dict:
                obs = e._last_obs_dict[agent_id]
                ##############################
                # Handle observation if it is a numpy array
                ##############################
                if isinstance(obs, np.ndarray):
                    self._last_original_obs = np.asarray(obs, dtype=np.float32)
                ##############################
                # Handle observation if it is a dictionary
                ##############################
                elif isinstance(obs, dict):
                    ##############################
                    # Check for 'observation' key with numpy array
                    ##############################
                    if 'observation' in obs and isinstance(obs['observation'], np.ndarray):
                        self._last_original_obs = np.asarray(obs['observation'], dtype=np.float32)
                    else:
                        self._last_original_obs = self._flatten_obs_dict(obs)
    
    def _get_obs_for_expert(self) -> np.ndarray:
        """Get observation suitable for expert policy (base, unnormalized)."""
        ##############################
        # Return cached observation if available
        ##############################
        if self._last_original_obs is not None:
            return self._last_original_obs

        ##############################
        # Check if wrapped environment has get_original_obs method
        ##############################
        if hasattr(self.env, "get_original_obs"):
            orig = self.env.get_original_obs()
            ##############################
            # If method returns valid numpy array, use it
            ##############################
            if orig is not None and isinstance(orig, np.ndarray):
                return np.asarray(orig, dtype=np.float32)

        ##############################
        # Walk up wrapper chain looking for observation data
        ##############################
        e = self.env
        ##############################
        # Traverse through nested wrapper layers
        ##############################
        while hasattr(e, "env"):
            ##############################
            # Check for cached last observation in wrapper
            ##############################
            if hasattr(e, "_last_obs") and e._last_obs is not None:
                ##############################
                # If observation is numpy array, return it
                ##############################
                if isinstance(e._last_obs, np.ndarray):
                    return np.asarray(e._last_obs, dtype=np.float32)
            ##############################
            # Check for cached original observation in wrapper
            ##############################
            if hasattr(e, "_last_original_obs") and e._last_original_obs is not None:
                ##############################
                # If cached original is numpy array, return it
                ##############################
                if isinstance(e._last_original_obs, np.ndarray):
                    return np.asarray(e._last_original_obs, dtype=np.float32)
            e = e.env

        ##############################
        # Check innermost environment for observation dictionary
        ##############################
        if hasattr(e, '_last_obs_dict'):
            agent_id = getattr(e, 'controlled_agent_id', 'agent_0')
            ##############################
            # If controlled agent has an observation
            ##############################
            if agent_id in e._last_obs_dict:
                obs = e._last_obs_dict[agent_id]
                ##############################
                # If observation is numpy array, return it
                ##############################
                if isinstance(obs, np.ndarray):
                    return np.asarray(obs, dtype=np.float32)
                ##############################
                # If observation is dictionary with 'observation' key
                ##############################
                elif isinstance(obs, dict) and 'observation' in obs:
                    return np.asarray(obs['observation'], dtype=np.float32)

        print("[WARN] ExpertActionBlendingWrapper: Could not find original observation")
        return np.zeros(27, dtype=np.float32)
    
    def get_episode_stats(self) -> Dict[str, float]:
        """Get blending statistics for current episode."""
        ##############################
        # If no steps taken in episode, return default stats
        ##############################
        if self._episode_steps == 0:
            return {"mean_alpha": 0.0, "episode_steps": 0}

        return {
            "mean_alpha": self._episode_expert_contribution / self._episode_steps,
            "episode_steps": self._episode_steps,
        }
    
    def get_total_stats(self) -> Dict[str, float]:
        """Get blending statistics across all steps."""
        return {
            "total_steps": self._total_steps,
            "current_alpha": self._current_alpha,
        }
    
    def get_last_actions(self) -> Dict[str, Optional[np.ndarray]]:
        """Get the last PPO, expert, and blended actions."""
        return {
            "ppo": self._last_ppo_action,
            "expert": self._last_expert_action,
            "blended": self._last_blended_action,
        }


def create_alpha_decay_fn(
    initial_alpha: float = 0.7,
    final_alpha: float = 0.1,
    decay_start: int = 500_000,
    decay_end: int = 4_000_000,
    decay_type: str = 'linear',
) -> Callable[[int], float]:
    """
    Create a decay function for alpha coefficient.

    Args:
        initial_alpha: Starting alpha (expert influence) - typically high (0.5-0.8)
        final_alpha: Ending alpha - typically low but nonzero (0.05-0.2)
        decay_start: Step to begin decay (hold at initial_alpha before this)
        decay_end: Step to reach final_alpha
        decay_type: 'linear', 'exponential', or 'cosine'

    Returns:
        Function mapping global_step -> alpha
    """
    def decay_fn(step: int) -> float:
        ##############################
        # If step is before decay start, return initial alpha
        ##############################
        if step < decay_start:
            return initial_alpha
        ##############################
        # If step is at or after decay end, return final alpha
        ##############################
        if step >= decay_end:
            return final_alpha

        progress = (step - decay_start) / (decay_end - decay_start)

        ##############################
        # Use linear interpolation decay
        ##############################
        if decay_type == 'linear':
            return initial_alpha - progress * (initial_alpha - final_alpha)
        ##############################
        # Use exponential decay (faster early, slower late)
        ##############################
        elif decay_type == 'exponential':
            import math
            return final_alpha + (initial_alpha - final_alpha) * math.exp(-3 * progress)
        ##############################
        # Use cosine annealing for smooth decay
        ##############################
        elif decay_type == 'cosine':
            import math
            return final_alpha + 0.5 * (initial_alpha - final_alpha) * (1 + math.cos(math.pi * progress))
        ##############################
        # Default to linear decay
        ##############################
        else:
            return initial_alpha - progress * (initial_alpha - final_alpha)

    return decay_fn


def create_staged_alpha_fn(
    stages: List[Dict[str, Any]],
) -> Callable[[int], float]:
    """
    Create a multi-stage alpha schedule.

    Example:
        stages = [
            {'end_step': 500_000, 'alpha': 0.8},      # Phase 1: High expert
            {'end_step': 2_000_000, 'alpha': 0.4},    # Phase 2: Medium blend
            {'end_step': 5_000_000, 'alpha': 0.1},    # Phase 3: Low expert
        ]

    Alpha transitions linearly between stages.
    """
    def staged_fn(step: int) -> float:
        prev_end = 0
        prev_alpha = stages[0]['alpha'] if stages else 0.5

        ##############################
        # Iterate through each stage to find which one the current step falls into
        ##############################
        for stage in stages:
            ##############################
            # If step falls before this stage's end, interpolate within this stage
            ##############################
            if step < stage['end_step']:
                progress = (step - prev_end) / (stage['end_step'] - prev_end)
                return prev_alpha + progress * (stage['alpha'] - prev_alpha)
            prev_end = stage['end_step']
            prev_alpha = stage['alpha']

        ##############################
        # After all stages completed, return the final stage's alpha
        ##############################
        return stages[-1]['alpha'] if stages else 0.1

    return staged_fn
