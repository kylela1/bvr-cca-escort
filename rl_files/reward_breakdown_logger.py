"""
This file implements diagnostic gymnasium wrappers that track all reward components flowing
through the wrapper stack and print human-readable summaries at episode boundaries. Two
variants are provided: RewardBreakdownLogger prints a detailed multi-line breakdown of
every reward component, and CompactRewardLogger prints a single-line summary. Both read
specific keys from the info dictionary that other reward shaping wrappers populate (such as
alignment_reward, turn_penalty, hvaa_penalty, tempo_shaping, and track_shaping).

Within the training pipeline, one of these loggers is placed as the outermost wrapper in
the environment stack (just before the Monitor wrapper) inside train_bace_clean.py. Because
it sits outermost, it captures the cumulative effect of all inner reward shaping wrappers.
The logged output is useful for diagnosing whether reward components are firing as expected
and at appropriate magnitudes during training.

Key configurable parameters: print_every_n controls how often episode summaries are printed
(default: 1, meaning every episode). verbose enables additional per-step detail in
RewardBreakdownLogger (default: False). The info dictionary keys that are tracked are
hardcoded to match the keys set by the other wrappers in this codebase (alignment_reward,
expert_blending_alpha, turn_penalty, hvaa_penalty, tempo_shaping, track_shaping, and
range_shaping).
"""

import gymnasium as gym
import numpy as np
from collections import defaultdict


class RewardBreakdownLogger(gym.Wrapper):
    """
    Tracks reward components from info dict and prints episode summaries.

    This wrapper looks for specific keys in the info dict that other wrappers
    add, and accumulates them over the episode. At episode end, it prints
    a breakdown showing where the reward came from.

    Args:
        env: The environment to wrap
        print_every_n: Print breakdown every N episodes (1 = every episode)
        reward_keys: List of info keys to track. If None, uses default set.
        verbose: If True, prints every episode. If False, only prints every N.
    """

    DEFAULT_KEYS = [
        "track_aware_shaping",
        "track_shaping",
        "range_shaping",
        "track_loss_count",
        "close_reward_total",
        "open_penalty_total",
        "tempo_shaping",
        "tempo_penalty",
        "tempo_bonus",
        "tempo_penalty_total",
        "tempo_bonus_total",
        "steps_red_alive",
        "range_close_shaping",
        "hvaa_survival_bonus",
        "red_killed_event",
        "pen_turn_total",
        "pen_turn_jerk",
        "pen_turn_mag",
        "sustained_turn_penalty",
        "expert_alignment_reward",
        "hvaa_destruction_penalty",
    ]
    
    def __init__(
        self,
        env,
        print_every_n: int = 1,
        reward_keys: list = None,
        verbose: bool = True,
        print_width: int = 60,
    ):
        super().__init__(env)
        self.print_every_n = max(1, int(print_every_n))
        self.reward_keys = reward_keys or self.DEFAULT_KEYS
        self.verbose = verbose
        self.print_width = print_width
        
        self._episode_count = 0
        self._reset_accumulators()
        
    def _reset_accumulators(self):
        """Reset all episode tracking."""
        self._step_count = 0
        self._total_reward = 0.0
        self._component_sums = defaultdict(float)
        self._component_counts = defaultdict(int)
        self._events = defaultdict(int)
        self._red_killed = False
        self._kill_step = None
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reset_accumulators()
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self._step_count += 1
        self._total_reward += float(reward)

        ##############################
        # Loop through each reward key to track components from the info dictionary
        ##############################
        for key in self.reward_keys:
            ##############################
            # Check if this key exists in the current step's info
            ##############################
            if key in info:
                val = info[key]
                ##############################
                # Handle boolean values as event counters
                ##############################
                if isinstance(val, bool):
                    ##############################
                    # If the boolean is True, increment the event counter for this key
                    ##############################
                    if val:
                        self._events[key] += 1
                        ##############################
                        # Track the kill event specially: record that red was killed and when
                        ##############################
                        if key == "red_killed_event" and not self._red_killed:
                            self._red_killed = True
                            self._kill_step = self._step_count
                ##############################
                # Handle numeric values as cumulative reward components
                ##############################
                elif isinstance(val, (int, float)):
                    self._component_sums[key] += float(val)
                    self._component_counts[key] += 1

        ##############################
        # At the end of an episode, print the reward breakdown if it's time to do so
        ##############################
        if terminated or truncated:
            self._episode_count += 1
            ##############################
            # Check if this episode matches the print frequency
            ##############################
            if self.verbose and (self._episode_count % self.print_every_n == 0):
                self._print_breakdown(info)

        return obs, reward, terminated, truncated, info
    
    def _print_breakdown(self, final_info):
        """Print episode reward breakdown."""
        w = self.print_width

        print("\n" + "=" * w)
        print(f"📊 EPISODE {self._episode_count} REWARD BREAKDOWN")
        print("=" * w)
        print(f"Steps: {self._step_count:,}")
        print(f"Total Reward: {self._total_reward:.2f}")
        ##############################
        # If red was killed during this episode, show the step where it happened
        ##############################
        if self._red_killed:
            print(f"🎯 Red Killed at step {self._kill_step}")
        print("-" * w)

        categories = {
            "Track-Aware Shaping": [
                ("track_shaping", "Track maintenance"),
                ("range_shaping", "Range shaping"),
                ("track_aware_shaping", "Combined (per-step)"),
                ("close_reward_total", "Close rewards (cumulative)"),
                ("open_penalty_total", "Open penalties (cumulative)"),
                ("track_loss_count", "Track losses (count)"),
            ],
            "Mission Tempo": [
                ("tempo_penalty", "Red-alive penalty (per-step)"),
                ("tempo_bonus", "Post-kill bonus (per-step)"),
                ("tempo_penalty_total", "Penalty total"),
                ("tempo_bonus_total", "Bonus total"),
                ("steps_red_alive", "Steps with red alive"),
            ],
            "Movement Penalties": [
                ("pen_turn_total", "Turn penalty total"),
                ("pen_turn_jerk", "Jerk penalty"),
                ("pen_turn_mag", "Magnitude penalty"),
                ("sustained_turn_penalty", "Sustained turn"),
            ],
            "Mission Rewards": [
                ("hvaa_survival_bonus", "HVAA survival"),
                ("range_close_shaping", "Range closing (old)"),
            ],
            "Events": [
                ("red_killed_event", "Red kills"),
            ],
        }

        ##############################
        # Iterate through each category of reward components
        ##############################
        for category, keys in categories.items():
            has_data = False
            lines = []

            ##############################
            # Check each key in the category to see if it has recorded values
            ##############################
            for key, label in keys:
                ##############################
                # If this is a numeric component with non-zero value, format it for display
                ##############################
                if key in self._component_sums and self._component_sums[key] != 0:
                    has_data = True
                    total = self._component_sums[key]
                    count = self._component_counts[key]
                    avg = total / count if count > 0 else 0
                    lines.append(f"  {label:30s} sum={total:+8.2f}  (avg={avg:+.4f}/step)")
                ##############################
                # If this is an event counter with at least one occurrence, format it for display
                ##############################
                elif key in self._events and self._events[key] > 0:
                    has_data = True
                    lines.append(f"  {label:30s} count={self._events[key]}")

            ##############################
            # If any data was found in this category, print the category header and all lines
            ##############################
            if has_data:
                print(f"\n[{category}]")
                ##############################
                # Print each formatted line for this category
                ##############################
                for line in lines:
                    print(line)

        print("\n" + "-" * w)
        print("COMPONENT TOTALS:")

        track_aware = self._component_sums.get("close_reward_total", 0) - self._component_sums.get("open_penalty_total", 0)
        track_loss_penalty = self._component_sums.get("track_loss_count", 0) * -0.5
        tempo_net = self._component_sums.get("tempo_bonus_total", 0) - self._component_sums.get("tempo_penalty_total", 0)
        turn_penalty = -self._component_sums.get("pen_turn_total", 0)
        hvaa_bonus = self._component_sums.get("hvaa_survival_bonus", 0)
        ##############################
        # Set kill bonus to 12.0 if red was killed, otherwise 0.0
        ##############################
        kill_bonus = 12.0 if self._red_killed else 0.0

        print(f"  Track-aware (close - open):  {track_aware:+.2f}")
        print(f"  Tempo (bonus - penalty):     {tempo_net:+.2f}")
        print(f"  Turn penalties:              {turn_penalty:+.2f}")
        print(f"  HVAA survival bonus:         {hvaa_bonus:+.2f}")
        print(f"  Kill reward:                 {kill_bonus:+.2f}")

        estimated = track_aware + tempo_net + turn_penalty + hvaa_bonus + kill_bonus
        print(f"\n  Estimated from components:   {estimated:+.2f}")
        print(f"  Actual total reward:         {self._total_reward:+.2f}")
        print(f"  Unaccounted:                 {self._total_reward - estimated:+.2f}")

        print("=" * w + "\n")


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
