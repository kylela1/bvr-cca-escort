"""
This file implements gymnasium wrappers that add per-step mission tempo incentives to the
reward signal. While the red (enemy) aircraft is alive, a small penalty is applied every
step to create urgency for the agent to engage quickly. After red is killed (and optionally
only if the HVAA is still alive), a small bonus is applied every step to reward continued
mission success. Two variants are provided: MissionTempoShapingPerStep with fixed
coefficients, and MissionTempoShapingScheduled with coefficients that anneal over training.

Within the training pipeline, MissionTempoShapingPerStep is applied in the environment
wrapper stack inside train_bace_clean.py. It reads the red_killed and hvaa_alive flags from
the info dictionary each step. The per-step nature of this shaping provides a much denser
reward signal than episode-level penalties, helping PPO learn temporal urgency earlier in
training.

Key configurable parameters for MissionTempoShapingPerStep: penalty_per_step is the
reward subtracted each step while red is alive (default: -0.005, producing up to -67.5
over a full 13,500-step episode). bonus_per_step is the reward added each step after red
is killed (default: 0.001). require_hvaa_alive gates the bonus on the HVAA still being
alive (default: True). For MissionTempoShapingScheduled: penalty_start/-end define the
annealing range, anneal_steps and hold_steps control the ramp schedule.
"""

import gymnasium as gym


class MissionTempoShapingPerStep(gym.Wrapper):
    """
    Per-step mission tempo shaping.

    While red alive: -penalty_per_step each step
    After red dead:  +bonus_per_step each step (if HVAA alive)

    Example magnitudes (for 13,500 step episodes):
        penalty_per_step = -0.005  → -67.5 total if red never killed
        bonus_per_step   = +0.001  → +13.5 total if red killed at step 0

    If red killed at step 5000:
        penalty = 5000 * -0.005 = -25.0
        bonus   = 8500 * +0.001 = +8.5
        net tempo = -16.5

    If red killed at step 1000:
        penalty = 1000 * -0.005 = -5.0
        bonus   = 12500 * +0.001 = +12.5
        net tempo = +7.5  ← Faster kill = positive tempo!

    Args:
        env: The environment to wrap
        penalty_per_step: Penalty each step while red is alive (negative value)
        bonus_per_step: Bonus each step after red is killed (positive value)
        require_hvaa_alive: Only give bonus if HVAA is still alive
    """
    def __init__(
        self,
        env,
        penalty_per_step: float = -0.005,
        bonus_per_step: float = 0.001,
        require_hvaa_alive: bool = True,
    ):
        super().__init__(env)
        self.penalty_per_step = float(penalty_per_step)
        self.bonus_per_step = float(bonus_per_step)
        self.require_hvaa_alive = require_hvaa_alive

        self._red_killed = False
        self._kill_step = None
        self._step_count = 0
        self._total_penalty = 0.0
        self._total_bonus = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._red_killed = False
        self._kill_step = None
        self._step_count = 0
        self._total_penalty = 0.0
        self._total_bonus = 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self._step_count += 1

        ##############################
        # Check if red has been killed in this step and latch the kill event
        ##############################
        if not self._red_killed and info.get("red_killed_event", False):
            self._red_killed = True
            self._kill_step = self._step_count

        shaping = 0.0

        ##############################
        # If red is still alive, apply penalty for each step; otherwise prepare bonus calculation
        ##############################
        if not self._red_killed:
            shaping = self.penalty_per_step
            self._total_penalty += abs(shaping)
        else:
            hvaa_ok = True
            ##############################
            # If HVAA alive requirement is enabled, check if HVAA is still alive
            ##############################
            if self.require_hvaa_alive:
                hvaa_ok = info.get("hvaa_alive", True)

            ##############################
            # If HVAA is acceptable, apply bonus for surviving after red's death
            ##############################
            if hvaa_ok:
                shaping = self.bonus_per_step
                self._total_bonus += shaping

        reward = float(reward) + shaping

        info["tempo_shaping"] = float(shaping)
        info["tempo_penalty_total"] = float(self._total_penalty)
        info["tempo_bonus_total"] = float(self._total_bonus)
        info["tempo_net"] = float(self._total_bonus - self._total_penalty)
        info["red_killed"] = bool(self._red_killed)
        info["kill_step"] = self._kill_step if self._kill_step else 0
        info["steps_red_alive"] = self._kill_step if self._kill_step else self._step_count

        return obs, reward, terminated, truncated, info


class MissionTempoShapingScheduled(gym.Wrapper):
    """
    Per-step tempo shaping with optional scheduling.

    Can start with lower penalties and ramp up as training progresses.
    This allows early exploration without harsh penalties, then enforces
    discipline later in training.

    Args:
        env: The environment to wrap
        penalty_start: Initial penalty per step (e.g., -0.001)
        penalty_end: Final penalty per step (e.g., -0.005)
        bonus_per_step: Bonus per step after kill
        anneal_steps: Steps over which to anneal penalty
        hold_steps: Steps to hold at start value before annealing
    """
    def __init__(
        self,
        env,
        penalty_start: float = -0.001,
        penalty_end: float = -0.005,
        bonus_per_step: float = 0.001,
        anneal_steps: int = 3_000_000,
        hold_steps: int = 500_000,
        require_hvaa_alive: bool = True,
    ):
        super().__init__(env)
        self.penalty_start = float(penalty_start)
        self.penalty_end = float(penalty_end)
        self.bonus_per_step = float(bonus_per_step)
        self.anneal_steps = int(anneal_steps)
        self.hold_steps = int(hold_steps)
        self.require_hvaa_alive = require_hvaa_alive

        self._global_step = 0
        self._red_killed = False
        self._step_count = 0
        self._total_penalty = 0.0
        self._total_bonus = 0.0
        self._current_penalty = penalty_start

    def set_global_step(self, t: int):
        """Called by training loop to update global step."""
        self._global_step = int(t)
        ##############################
        # Propagate global step to wrapped environment if it supports this method
        ##############################
        if hasattr(self.env, "set_global_step"):
            self.env.set_global_step(t)

    def _get_current_penalty(self):
        """Calculate penalty based on schedule."""
        t = self._global_step
        ##############################
        # During the hold period, return the starting penalty without any annealing
        ##############################
        if t <= self.hold_steps:
            return self.penalty_start
        ##############################
        # If anneal_steps is zero or negative, return the end penalty immediately
        ##############################
        if self.anneal_steps <= 0:
            return self.penalty_end
        progress = min(1.0, (t - self.hold_steps) / self.anneal_steps)
        return self.penalty_start + progress * (self.penalty_end - self.penalty_start)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._red_killed = False
        self._step_count = 0
        self._total_penalty = 0.0
        self._total_bonus = 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self._step_count += 1
        self._current_penalty = self._get_current_penalty()

        ##############################
        # Check if red has been killed and latch the kill event
        ##############################
        if not self._red_killed and info.get("red_killed_event", False):
            self._red_killed = True

        shaping = 0.0

        ##############################
        # If red is still alive, apply current penalty; otherwise prepare bonus
        ##############################
        if not self._red_killed:
            shaping = self._current_penalty
            self._total_penalty += abs(shaping)
        else:
            hvaa_ok = info.get("hvaa_alive", True) if self.require_hvaa_alive else True
            ##############################
            # If HVAA is acceptable, apply bonus for surviving after red's death
            ##############################
            if hvaa_ok:
                shaping = self.bonus_per_step
                self._total_bonus += shaping

        reward = float(reward) + shaping

        info["tempo_shaping"] = float(shaping)
        info["tempo_penalty_total"] = float(self._total_penalty)
        info["tempo_bonus_total"] = float(self._total_bonus)
        info["tempo_penalty_coef"] = float(self._current_penalty)
        info["red_killed"] = bool(self._red_killed)

        return obs, reward, terminated, truncated, info
