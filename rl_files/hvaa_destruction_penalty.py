"""
This file implements gymnasium wrappers that apply an immediate penalty at the exact
timestep when the HVAA (High Value Airborne Asset) is destroyed. The Godot environment
reports hvaa_alive status in the info dictionary each step, and these wrappers detect the
transition from alive to destroyed and inject a large negative reward. This enables proper
credit assignment for PPO, which would otherwise only see the HVAA loss as a delayed
terminal signal hundreds of steps later. Two variants are provided: HVAADestructionPenalty
with a fixed penalty, and HVAADestructionPenaltyAnnealed with a penalty that can change
over training.

Within the training pipeline, one of these wrappers is applied in the environment wrapper
stack inside train_bace_clean.py. It processes the info dictionary returned by the
environment each step, checking for the hvaa_alive flag transitioning from True to False.
The only_once parameter ensures the penalty fires at most once per episode to avoid
double-counting.

Key configurable parameters: penalty sets the reward subtracted when HVAA destruction is
detected (default: -10.0). only_once ensures the penalty is applied at most once per
episode (default: True). For the annealed variant, penalty_start and penalty_end define
the penalty range, with anneal_steps and hold_steps controlling the ramp schedule from
start to end values over training.
"""

import numpy as np
import gymnasium as gym


class HVAADestructionPenalty(gym.Wrapper):
    """
    Detects HVAA destruction and applies immediate penalty.

    The key insight: PPO needs the penalty at the exact timestep of destruction,
    not as a delayed terminal reward hundreds/thousands of steps later.

    Parameters:
    -----------
    penalty : float
        Negative reward applied when HVAA transitions from alive to destroyed.
        Default -10.0 matches the Godot hvaa_loss_factor.
    only_once : bool
        If True, only applies penalty once per episode (first destruction).
        If False, applies if HVAA somehow respawns and dies again.
    """

    def __init__(
        self,
        env,
        penalty: float = -10.0,
        only_once: bool = True,
    ):
        super().__init__(env)
        self.penalty = float(penalty)
        self.only_once = only_once

        self._prev_hvaa_alive = True
        self._already_penalized = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_hvaa_alive = info.get("hvaa_alive", True)
        self._already_penalized = False
        return obs, info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)

        hvaa_alive = info.get("hvaa_alive", True)
        hvaa_destroyed_now = (self._prev_hvaa_alive and not hvaa_alive)

        penalty_applied = 0.0
        ##############################
        # Check if HVAA was destroyed in this step (transitioned from alive to not alive)
        ##############################
        if hvaa_destroyed_now:
            ##############################
            # Apply penalty only if we should apply it once and haven't already, or if we allow multiple applications
            ##############################
            if not self.only_once or not self._already_penalized:
                penalty_applied = self.penalty
                rew = float(rew) + penalty_applied
                self._already_penalized = True

        self._prev_hvaa_alive = hvaa_alive

        info["hvaa_destruction_penalty"] = float(penalty_applied)
        info["hvaa_destruction_event"] = hvaa_destroyed_now

        return obs, rew, terminated, truncated, info


class HVAADestructionPenaltyWithDecay(gym.Wrapper):
    """
    Same as above but with optional penalty decay over training.

    Use case: Start with high penalty to establish behavior, then
    reduce it so the agent can focus on other objectives.

    Parameters:
    -----------
    penalty_start : float
        Initial penalty (e.g., -15.0)
    penalty_end : float
        Final penalty after annealing (e.g., -5.0)
    anneal_steps : int
        Steps over which to anneal
    hold_steps : int
        Steps to hold at initial value before annealing
    """

    def __init__(
        self,
        env,
        penalty_start: float = -10.0,
        penalty_end: float = -10.0,
        anneal_steps: int = 2_000_000,
        hold_steps: int = 0,
    ):
        super().__init__(env)
        self.penalty_start = float(penalty_start)
        self.penalty_end = float(penalty_end)
        self.anneal_steps = int(anneal_steps)
        self.hold_steps = int(hold_steps)

        self.penalty = float(penalty_start)
        self._prev_hvaa_alive = True
        self._already_penalized = False
        self._global_step = 0

    def set_global_step(self, t: int):
        self._global_step = int(t)
        self.penalty = self._compute_penalty(t)
        ##############################
        # Check if the underlying environment also supports setting global step
        ##############################
        if hasattr(self.env, "set_global_step"):
            try:
                self.env.set_global_step(t)
            except Exception:
                pass

    def _compute_penalty(self, t: int) -> float:
        ##############################
        # If current step is within the hold period, return the starting penalty without annealing
        ##############################
        if t <= self.hold_steps:
            return self.penalty_start
        ##############################
        # If annealing steps is zero or negative, return the end penalty immediately
        ##############################
        if self.anneal_steps <= 0:
            return self.penalty_end
        x = min(1.0, max(0.0, (t - self.hold_steps) / float(self.anneal_steps)))
        return self.penalty_start + x * (self.penalty_end - self.penalty_start)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_hvaa_alive = info.get("hvaa_alive", True)
        self._already_penalized = False
        return obs, info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)

        hvaa_alive = info.get("hvaa_alive", True)
        hvaa_destroyed_now = (self._prev_hvaa_alive and not hvaa_alive)

        penalty_applied = 0.0
        ##############################
        # Check if HVAA was destroyed in this step and we haven't already applied a penalty
        ##############################
        if hvaa_destroyed_now and not self._already_penalized:
            penalty_applied = self.penalty
            rew = float(rew) + penalty_applied
            self._already_penalized = True

        self._prev_hvaa_alive = hvaa_alive

        info["hvaa_destruction_penalty"] = float(penalty_applied)
        info["hvaa_destruction_event"] = hvaa_destroyed_now
        info["hvaa_penalty_coef"] = float(self.penalty)

        return obs, rew, terminated, truncated, info
