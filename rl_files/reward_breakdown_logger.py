"""
This file implements CompactRewardLogger, a diagnostic gymnasium wrapper that tracks reward
components flowing through the wrapper stack and prints a single-line human-readable summary
at episode boundaries. It reads specific keys from the info dictionary that other reward
shaping wrappers populate (such as tempo_shaping, track_shaping, and pen_sustained_spin).

Within the training pipeline, this logger is placed as the outermost wrapper in the
environment stack (just before the Monitor wrapper) inside train_bace_clean.py. Because it
sits outermost, it captures the cumulative effect of all inner reward shaping wrappers. The
logged output is useful for diagnosing whether reward components are firing as expected and
at appropriate magnitudes during training. It has no effect on the reward value itself --
purely a pass-through diagnostic.

Key configurable parameters: print_every_n controls how often episode summaries are printed
(default: 1, meaning every episode).
"""

import gymnasium as gym
import numpy as np
from collections import defaultdict


class CompactRewardLogger(gym.Wrapper):
    """
    Lighter-weight version that just prints one line per episode.
    """
    def __init__(self, env, print_every_n: int = 1):
        super().__init__(env)
        self.print_every_n = max(1, int(print_every_n))
        self._episode_count = 0
        self._reset()

    def _reset(self):
        self._steps = 0
        self._total = 0.0
        self._track_loss = 0
        self._tempo_penalty = 0.0
        self._tempo_bonus = 0.0
        self._close_rew = 0.0
        self._open_pen = 0.0
        self._killed = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reset()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self._steps += 1
        self._total += float(reward)

        self._track_loss = info.get("track_loss_count", self._track_loss)
        self._tempo_penalty = info.get("tempo_penalty_total", self._tempo_penalty)
        self._tempo_bonus = info.get("tempo_bonus_total", self._tempo_bonus)
        self._close_rew = info.get("close_reward_total", self._close_rew)
        self._open_pen = info.get("open_penalty_total", self._open_pen)

        ##############################
        # Check if the red agent was killed in this step
        ##############################
        if info.get("red_killed_event", False):
            self._killed = True

        ##############################
        # At episode end, print a one-line summary if it matches the print frequency
        ##############################
        if terminated or truncated:
            self._episode_count += 1
            ##############################
            # Print summary only if this episode number matches the print frequency
            ##############################
            if self._episode_count % self.print_every_n == 0:
                kill_str = "🎯" if self._killed else "  "
                print(f"[EP {self._episode_count:4d}] {kill_str} R={self._total:+7.1f} | "
                      f"tempo={-self._tempo_penalty + self._tempo_bonus:+6.1f} | "
                      f"range={self._close_rew - self._open_pen:+6.1f} | "
                      f"trk_loss={self._track_loss:2d} | "
                      f"steps={self._steps}")

        return obs, reward, terminated, truncated, info
