#!/usr/bin/env python3
"""
Single, shared definition of the B-ACE environment wrapper stack, used by BOTH
train_bace_clean.py and evaluate_experiment_seeds.py (see
IMPROVEMENT_PLAN_2_WRAPPERS_AND_DEPENDENCIES.md, Part A, Task 2).

Before this file existed, each of those two scripts built its own, independently-maintained
copy of this construction logic. They had drifted apart: HVAASurvivalEveryNStepsBonus and
FireCooldownWrapper were instantiated with different parameters in each script, and
evaluate_experiment_seeds.py's evaluation environment was missing roughly eight wrappers
present in train_bace_clean.py's (HeadingRateLimitWrapper, ActionSmoothnessPenalty,
MissionTempoShapingPerStep, SustainedTurnPenaltySimple, CombinedTrackAwareShaping,
KillRewardWrapper, plus diagnostics). Both scripts now call build_env() below, so there is
exactly one wrapper stack definition to reason about, and it cannot drift between them again.

NOTE ON BEHAVIOR CHANGE: consolidating these two independently-evolved stacks into one is an
intentional, deliberate behavior change to evaluate_experiment_seeds.py specifically (confirmed
with the repo owner before implementing) -- its evaluation environment now includes the full
reward-shaping stack train_bace_clean.py already used for its own internal evaluation, rather
than the smaller subset it used before. Re-evaluating a checkpoint against an old saved result
computed before this change will show different reward numbers for that reason. It does NOT
change train_bace_clean.py's own behavior -- confirmed via tools/equivalence_check.py, which
shows this file produces an identical wrapper-stack structure to what train_bace_clean.py built
inline before this refactor.

Within the broader pipeline: import build_env() from this module and call it once per
environment you need (train, or eval). The wrapper order is fixed and documented inline below;
see IMPROVEMENT_PLAN_2_WRAPPERS_AND_DEPENDENCIES.md for why each wrapper exists.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

from b_ace_py.B_ACE_GodotPettingZooWrapper import B_ACE_GodotPettingZooWrapper
from b_ace_py.contracts import check_info_keys
from b_ace_py.enriched_observation_wrapper_ablation import EnrichedObservationWrapper, SplitNormalizationWrapper
from expert_alignment_wrapper import ExpertAlignmentWrapper, create_alignment_decay_fn
from aggressive_hunter_fixed import AggressiveHunterFixed, get_default_obs_indices
from expert_action_blending_wrapper import ExpertActionBlendingWrapper, create_alpha_decay_fn
from sustained_turn_penalty import SustainedTurnPenaltySimple
from track_aware_shaping import CombinedTrackAwareShaping
from mission_tempo_shaping_perstep import MissionTempoShapingPerStep
from reward_breakdown_logger import CompactRewardLogger

##############################
# Past every schedule's hold+anneal window (ActionSmoothnessPenalty: 1_500_000+4_500_000;
# FireCooldownWrapper: 10_000+5_000_000, rising_edge_after=1_000_000), so passing this as
# freeze_schedule_at_step guarantees both are fully annealed to their end values.
##############################
FULLY_ANNEALED_STEP = 10_000_000


def linear_anneal(t: int, start: float, end: float, steps: int, hold: int = 0) -> float:
    """Hold `start` for `hold` steps, then linearly move to `end` over `steps`."""
    if t <= hold:
        return float(start)
    if steps <= 0:
        return float(end)
    x = min(1.0, max(0.0, (t - hold) / float(steps)))
    return float(start + x * (end - start))


class SingleAgentBACEEnv(gym.Env):
    """
    SB3-compatible single-agent wrapper around B_ACE_GodotPettingZooWrapper.

    Uses "reset first" approach: resets env immediately, builds spaces from actual data.
    This avoids issues with PettingZoo's observation_space() method API.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        bace_config: Dict[str, Any],
        controlled_agent_id: str = "agent_0",
        max_episode_steps: int = 36000,
    ):
        super().__init__()
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0
        self._checked_info_keys = False

        self._ma_env = B_ACE_GodotPettingZooWrapper(device="cpu", **bace_config)
        self.observation_labels = getattr(self._ma_env, "observation_labels", None)

        if hasattr(self._ma_env, "get_hvaa_indices"):
            self.get_hvaa_indices = self._ma_env.get_hvaa_indices

        self.obs_map = getattr(self._ma_env, "obs_map", None)
        self.observation_labels_flat = getattr(self._ma_env, "observation_labels_flat", None)

        self.obs_labels = getattr(self._ma_env, "obs_labels", None)

        obs_dict, info_dict = self._ma_env.reset()
        self.obs_map = getattr(self._ma_env, "obs_map", self.obs_map)
        self.observation_labels_flat = getattr(self._ma_env, "observation_labels_flat", self.observation_labels_flat)
        self._last_obs_dict = obs_dict

        agents = getattr(self._ma_env, "agents", [])
        ##############################
        # Determine which agent to control
        ##############################
        if agents:
            if controlled_agent_id in agents:
                self.controlled_agent_id = controlled_agent_id
            else:
                print(f"WARNING: '{controlled_agent_id}' not in env.agents={agents}, "
                      f"defaulting to '{agents[0]}'")
                self.controlled_agent_id = agents[0]
        else:
            self.controlled_agent_id = controlled_agent_id

        ##############################
        # Build observation space from actual observation
        ##############################
        if self.controlled_agent_id in obs_dict:
            raw_obs = obs_dict[self.controlled_agent_id]
        else:
            raw_obs = next(iter(obs_dict.values()))

        obs_vec = self._to_array(raw_obs)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_vec.shape,
            dtype=np.float32,
        )
        print(f"[DEBUG] Built observation_space from reset: shape={obs_vec.shape}")

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )
        print(f"[DEBUG] Built action_space: shape=(4,)")

    def _to_array(self, obj) -> np.ndarray:
        """Recursively extract numeric vector from nested structures."""
        if isinstance(obj, np.ndarray):
            return obj.astype(np.float32)
        ##############################
        # Convert lists and tuples to numpy arrays
        ##############################
        if isinstance(obj, (list, tuple)):
            return np.asarray(obj, dtype=np.float32)
        ##############################
        # Convert scalar values to single-element arrays
        ##############################
        if isinstance(obj, (float, int)):
            return np.asarray([obj], dtype=np.float32)
        ##############################
        # Recursively extract from dictionaries
        ##############################
        if isinstance(obj, dict):
            if "obs" in obj:
                return self._to_array(obj["obs"])
            for v in obj.values():
                try:
                    arr = self._to_array(v)
                    if arr.dtype.kind in ("f", "i"):
                        return arr
                except (TypeError, ValueError):
                    continue
            raise RuntimeError(f"Could not extract obs from dict: {list(obj.keys())}")
        raise RuntimeError(f"Unsupported observation type {type(obj)}")

    def _extract_single_obs(self, obs_dict: Dict[str, Any]) -> np.ndarray:
        """Extract observation for controlled agent."""
        if self.controlled_agent_id in obs_dict:
            raw_obs = obs_dict[self.controlled_agent_id]
        else:
            raw_obs = next(iter(obs_dict.values()))
        return self._to_array(raw_obs)

    def _build_action_payload(self, agent_action: np.ndarray) -> Any:
        """Build action dict for all agents."""
        clipped = np.clip(agent_action, -1.0, 1.0).astype(np.float32)

        if hasattr(self._ma_env, "agents"):
            return {agent: clipped for agent in self._ma_env.agents}
        else:
            return {self.controlled_agent_id: clipped}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self._elapsed_steps = 0

        obs_dict, info_dict = self._ma_env.reset()
        self.obs_map = getattr(self._ma_env, "obs_map", self.obs_map)
        self.observation_labels_flat = getattr(self._ma_env, "observation_labels_flat", self.observation_labels_flat)
        self._last_obs_dict = obs_dict

        obs = self._extract_single_obs(obs_dict)
        info = info_dict.get(self.controlled_agent_id, {}) if isinstance(info_dict, dict) else {}
        return obs, info

    def step(self, action: np.ndarray):
        self._elapsed_steps += 1

        action_dict = self._build_action_payload(action)
        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self._ma_env.step(action_dict)
        self._last_obs_dict = obs_dict

        obs = self._extract_single_obs(obs_dict)

        ##############################
        # Extract reward for controlled agent
        ##############################
        if isinstance(rew_dict, dict):
            reward = float(rew_dict.get(self.controlled_agent_id, 0.0))
        else:
            reward = float(rew_dict)

        ##############################
        # Extract termination signal for controlled agent
        ##############################
        if isinstance(term_dict, dict):
            terminated = bool(term_dict.get(self.controlled_agent_id, False)) or bool(term_dict.get("__all__", False))
        else:
            terminated = bool(term_dict)

        ##############################
        # Extract truncation signal for controlled agent
        ##############################
        if isinstance(trunc_dict, dict):
            truncated = bool(trunc_dict.get(self.controlled_agent_id, False)) or bool(trunc_dict.get("__all__", False))
        else:
            truncated = bool(trunc_dict)

        ##############################
        # Check if episode exceeded max steps
        ##############################
        if terminated or truncated:
            print(f"[SINGLE-AGENT] step={self._elapsed_steps} terminated={terminated} truncated={truncated} "
                  f"raw_term={term_dict} raw_trunc={trunc_dict}")

        if self._elapsed_steps >= self._max_episode_steps and not terminated:
            truncated = True

        info = info_dict.get(self.controlled_agent_id, {}) if isinstance(info_dict, dict) else {}

        ##############################
        # One-time (not per-step) check that Godot isn't emitting info keys this repo doesn't
        # know about -- see b_ace_py/contracts.py. Checked once rather than every step since
        # this is diagnostic, not correctness-critical, and a per-step set-diff over a
        # multi-million-step run would add needless overhead.
        ##############################
        if not self._checked_info_keys:
            check_info_keys(info, source="SingleAgentBACEEnv.step() info")
            self._checked_info_keys = True

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        if hasattr(self._ma_env, "close"):
            self._ma_env.close()


class HVAASurvivalEveryNStepsBonus(gym.Wrapper):
    def __init__(self, env, every_n_steps: int = 50, bonus: float = 0.05):
        super().__init__(env)
        self.every_n_steps = int(every_n_steps)
        self.bonus = float(bonus)
        self._t = 0

    def reset(self, **kwargs):
        self._t = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)

        hvaa_alive = bool(info.get("hvaa_alive", True))

        bonus_now = 0.0
        ##############################
        # Check if HVAA is alive and if it's time to give bonus
        ##############################
        if hvaa_alive and (self._t % self.every_n_steps == 0):
            bonus_now = self.bonus
            rew = float(rew) + bonus_now

        info["hvaa_survival_bonus"] = float(bonus_now)

        self._t += 1
        return obs, rew, terminated, truncated, info


class KillRewardWrapper(gym.Wrapper):
    """Add reward when red_killed_event FIRST becomes True."""
    def __init__(self, env, kill_reward: float = 10.0):
        super().__init__(env)
        self.kill_reward = kill_reward
        self._already_awarded = False

    def reset(self, **kwargs):
        self._already_awarded = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)

        ##############################
        # Award kill bonus only once when red_killed_event first becomes true
        ##############################
        if info.get("red_killed_event", False) and not self._already_awarded:
            reward += self.kill_reward
            self._already_awarded = True
            print(f"[KILL REWARD] +{self.kill_reward} for killing red!")

        return obs, reward, term, trunc, info


class MissionOutcomeBonus(gym.Wrapper):
    """
    One-time terminal rewards tied directly to mission outcome, used by reward_mode=
    "mission_aligned" (see REWARD_SHAPING_STEPS_FORWARD.md Sec 4.2/4.4 for the full writeup).

    Two gaps this closes, both entirely at the Python wrapper level (no Godot/GDScript change,
    no re-export required):

    1. HVAA loss currently goes unpenalized. `SimpleExample_B_ACE_config.json` sets
       hvaa_loss_factor=0.0, and even if it didn't, `SimManager.gd`'s termination logic never
       triggers the "HVAA_Destroyed" branch unless the CCA is *also* dead (CODEBASE_GUIDE.md
       Sec 5.2) -- so an HVAA lost while the CCA survives currently applies no explicit penalty
       at all. This wrapper applies one, the first time `hvaa_alive` is reported False each
       episode, regardless of whether the episode actually ends at that moment. -15.0 is the
       thesis's own stated magnitude for this term (Sec 3.2.3's r_env equation) -- present in
       the intended design, never actually wired up in this codebase.
    2. The one-time mission-success reward (Godot's mission_accomplished_factor, +8.0) is small
       relative to what the per-step shaping terms it competes against can accumulate over a
       long episode (REWARD_SHAPING_STEPS_FORWARD.md Sec 1 quantifies this: killing red early
       and then loitering can out-earn efficient mission completion). This wrapper adds a
       second, larger one-time bonus specifically on a successful mission-ending episode, sized
       so that completing the mission is clearly worth more than any single kill (+12 from
       KillRewardWrapper, +10 from Godot's hit_enemy_factor = 22 total) -- 8 (Godot) + 25
       (here) = 33, a deliberate margin above 22, not an arbitrary round number.
    """
    def __init__(self, env, hvaa_loss_penalty: float = -15.0, mission_success_bonus: float = 25.0):
        super().__init__(env)
        self.hvaa_loss_penalty = float(hvaa_loss_penalty)
        self.mission_success_bonus = float(mission_success_bonus)
        self._hvaa_loss_applied = False

    def reset(self, **kwargs):
        self._hvaa_loss_applied = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        ##############################
        # One-time penalty the first time the HVAA is reported destroyed this episode, whether
        # or not that also ends the episode this step.
        ##############################
        if not self._hvaa_loss_applied and not info.get("hvaa_alive", True):
            reward += self.hvaa_loss_penalty
            self._hvaa_loss_applied = True
            info["mission_outcome_hvaa_loss_penalty"] = self.hvaa_loss_penalty
            print(f"[MISSION OUTCOME] {self.hvaa_loss_penalty} for HVAA destroyed")

        ##############################
        # One-time bonus on a successful mission-ending episode. mission_success is only
        # populated at episode end (b_ace_py/contracts.py's EXPECTED_INFO_KEYS), so this can
        # only ever fire on the terminating step.
        ##############################
        if (terminated or truncated) and info.get("mission_success", False):
            reward += self.mission_success_bonus
            info["mission_outcome_success_bonus"] = self.mission_success_bonus
            print(f"[MISSION OUTCOME] +{self.mission_success_bonus} for mission success")

        return obs, reward, terminated, truncated, info


class ActionSmoothnessPenalty(gym.Wrapper):
    """
    Scheduled smoothness penalty (jerk + magnitude), increases over training.
    """
    def __init__(
        self,
        env,
        hdg_idx=0,
        jerk_start=0.0,
        jerk_end=0.002,
        mag_start=0.0,
        mag_end=0.001,
        anneal_steps=2_000_000,
        hold_steps=0,
    ):
        super().__init__(env)
        self.hdg_idx = int(hdg_idx)

        self.jerk_start = float(jerk_start)
        self.jerk_end = float(jerk_end)
        self.mag_start = float(mag_start)
        self.mag_end = float(mag_end)
        self.anneal_steps = int(anneal_steps)
        self.hold_steps = int(hold_steps)

        self.jerk_coef = float(jerk_start)
        self.mag_coef = float(mag_start)

        self.prev = None
        self._global_step = 0

    def set_global_step(self, t: int):
        self._global_step = int(t)
        if hasattr(self.env, "set_global_step"):
            try:
                self.env.set_global_step(t)
            except Exception:
                pass

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev = None
        return obs, info

    def step(self, action):
        self.jerk_coef = linear_anneal(self._global_step, self.jerk_start, self.jerk_end, self.anneal_steps, hold=self.hold_steps)
        self.mag_coef  = linear_anneal(self._global_step, self.mag_start,  self.mag_end,  self.anneal_steps, hold=self.hold_steps)

        a = np.asarray(action, dtype=np.float32)
        jerk_pen = 0.0
        mag_pen = 0.0

        ##############################
        # If there was a previous action, compute jerk penalty
        ##############################
        if self.prev is not None:
            d = float(a[self.hdg_idx] - self.prev[self.hdg_idx])
            jerk_pen = d * d
        m = float(a[self.hdg_idx])
        mag_pen = m * m

        obs, rew, terminated, truncated, info = self.env.step(action)

        penalty = self.jerk_coef * jerk_pen + self.mag_coef * mag_pen
        rew = float(rew) - penalty

        info["pen_turn_jerk"] = float(self.jerk_coef * jerk_pen)
        info["pen_turn_mag"]  = float(self.mag_coef * mag_pen)
        info["pen_turn_total"] = float(penalty)
        info["pen_turn_jerk_coef"] = float(self.jerk_coef)
        info["pen_turn_mag_coef"]  = float(self.mag_coef)

        self.prev = a.copy()
        return obs, rew, terminated, truncated, info


class FireCooldownWrapper(gym.Wrapper):
    """
    Fire gating with a schedule:
      - early training: low cooldown, no rising-edge requirement (easier to learn kills)
      - later training: higher cooldown + rising-edge requirement (realistic discipline)
    """
    def __init__(
        self,
        env,
        fire_idx: int = 3,
        threshold: float = 0.5,
        cd_start: int = 20,
        cd_end: int = 240,
        cd_steps: int = 2_000_000,
        cd_hold: int = 0,
        rising_edge_after: int = 1_000_000,
    ):
        super().__init__(env)
        self.fire_idx = int(fire_idx)
        self.threshold = float(threshold)

        self.cd_start = int(cd_start)
        self.cd_end = int(cd_end)
        self.cd_steps = int(cd_steps)
        self.cd_hold = int(cd_hold)
        self.rising_edge_after = int(rising_edge_after)

        self._t = 0
        self._last_fire_t = -10**9
        self._prev_want_fire = False
        self._global_step = 0

        self.cooldown_steps = int(cd_start)

    def set_global_step(self, t: int):
        self._global_step = int(t)
        if hasattr(self.env, "set_global_step"):
            try:
                self.env.set_global_step(t)
            except Exception:
                pass

    def reset(self, **kwargs):
        self._t = 0
        self._last_fire_t = -10**9
        self._prev_want_fire = False
        return self.env.reset(**kwargs)

    def step(self, action):
        self.cooldown_steps = int(round(linear_anneal(
            self._global_step, self.cd_start, self.cd_end, self.cd_steps, hold=self.cd_hold
        )))
        self.cooldown_steps = max(1, self.cooldown_steps)

        a = np.array(action, dtype=np.float32, copy=True)
        want_fire = float(a[self.fire_idx]) > self.threshold

        ##############################
        # Determine if rising-edge requirement should be enabled based on training progress
        ##############################
        use_rising_edge = (self._global_step >= self.rising_edge_after)
        rising_edge = want_fire and (not self._prev_want_fire)

        allow = False
        ##############################
        # Check if fire action is allowed based on cooldown and rising-edge requirements
        ##############################
        if want_fire:
            ##############################
            # Verify enough time has passed since last fire action
            ##############################
            if (self._t - self._last_fire_t) >= self.cooldown_steps:
                ##############################
                # Check rising-edge constraint if it's enabled
                ##############################
                if (not use_rising_edge) or rising_edge:
                    allow = True

        ##############################
        # Set fire action based on whether it was allowed
        ##############################
        if allow:
            a[self.fire_idx] = 1.0
            self._last_fire_t = self._t
        else:
            a[self.fire_idx] = 0.0

        self._prev_want_fire = want_fire

        obs, rew, terminated, truncated, info = self.env.step(a)
        self._t += 1

        info["fire_cd_current"] = int(self.cooldown_steps)
        info["fire_use_rising_edge"] = bool(use_rising_edge)
        info["fire_want"] = bool(want_fire)
        info["fire_allowed"] = bool(allow)
        info["fire_cooldown_remaining_steps"] = int(max(0, self.cooldown_steps - (self._t - 1 - self._last_fire_t)))
        return obs, rew, terminated, truncated, info


class HeadingRateLimitWrapper(gym.Wrapper):
    def __init__(self, env, hdg_idx=0, max_delta=0.15):
        super().__init__(env)
        self.hdg_idx = int(hdg_idx)
        self.max_delta = float(max_delta)
        self.prev_hdg_cmd = 0.0

    def reset(self, **kwargs):
        self.prev_hdg_cmd = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        a = np.array(action, dtype=np.float32, copy=True)
        target = float(a[self.hdg_idx])
        delta = np.clip(target - self.prev_hdg_cmd, -self.max_delta, self.max_delta)
        a[self.hdg_idx] = self.prev_hdg_cmd + delta
        self.prev_hdg_cmd = float(a[self.hdg_idx])
        return self.env.step(a)


class NanGuardWrapper(gym.Wrapper):
    """
    Raises immediately if obs/reward becomes NaN/Inf.
    Also prints useful context to map the bad value back to a feature.
    """
    def __init__(self, env, name="env", print_first_n=20):
        super().__init__(env)
        self.name = name
        self.step_i = 0
        self.print_first_n = int(print_first_n)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.step_i = 0
        self._check_obs(obs, where="reset", info=info)
        return obs, info

    def step(self, action):
        self.step_i += 1
        obs, reward, terminated, truncated, info = self.env.step(action)

        self._check_obs(obs, where="step", info=info)
        self._check_reward(reward, info=info)

        return obs, reward, terminated, truncated, info

    def _check_obs(self, obs, where, info):
        x = obs["obs"] if isinstance(obs, dict) and "obs" in obs else obs
        arr = np.asarray(x, dtype=np.float32)

        ##############################
        # Check if any observation values are non-finite (NaN or Inf)
        ##############################
        if not np.all(np.isfinite(arr)):
            bad = np.where(~np.isfinite(arr))[0]
            print(f"\n[NAN-GUARD:{self.name}] Non-finite OBS at {where} step={self.step_i}")
            print(f"  bad indices (first {self.print_first_n}): {bad[:self.print_first_n].tolist()}")
            print(f"  bad values  (first {self.print_first_n}): {arr[bad[:self.print_first_n]]}")

            ##############################
            # Try to print feature labels if available for context
            ##############################
            labels = getattr(self.env, "observation_labels_flat", getattr(self.env, "observation_labels", None))
            ##############################
            # Print labels for bad observation indices
            ##############################
            if labels is not None and isinstance(labels, (list, tuple)) and len(labels) == len(arr):
                for i in bad[:self.print_first_n]:
                    print(f"  label[{i}] = {labels[i]}")

            ##############################
            # Print info keys for debugging context
            ##############################
            if isinstance(info, dict):
                print("  info keys:", list(info.keys())[:50])

            raise RuntimeError("Non-finite observation detected (NaN/Inf).")

    def _check_reward(self, reward, info):
        r = float(reward)
        ##############################
        # Check if reward value is non-finite (NaN or Inf)
        ##############################
        if not np.isfinite(r):
            print(f"\n[NAN-GUARD:{self.name}] Non-finite REWARD at step={self.step_i}: {r}")
            if isinstance(info, dict):
                print("  info keys:", list(info.keys())[:50])
            raise RuntimeError("Non-finite reward detected (NaN/Inf).")


class _TermTrace(gym.Wrapper):
    """Prints a one-line trace whenever an episode ends. Debug-only, no effect on reward."""
    def __init__(self, env):
        super().__init__(env)
        self._steps = 0

    def reset(self, **kw):
        self._steps = 0
        return self.env.reset(**kw)

    def step(self, action):
        o, r, term, trunc, i = self.env.step(action)
        self._steps += 1
        if term or trunc:
            print(f"[PRE-MONITOR] step={self._steps} term={term} trunc={trunc}")
        return o, r, term, trunc, i


def build_env(
    bace_config: Dict[str, Any],
    *,
    is_eval: bool,
    log_dir_str: str,
    seed: int,
    use_enriched_obs: bool = False,
    ablation_config: str = "all",
    expert_alignment: bool = False,
    alignment_coef: float = 0.002,
    alignment_decay_start: int = 500_000,
    alignment_decay_end: int = 2_000_000,
    expert_blending: bool = False,
    blend_initial_alpha: float = 0.8,
    blend_final_alpha: float = 0.05,
    blend_decay_start: int = 500_000,
    blend_decay_end: int = 5_000_000,
    blend_decay_type: str = "linear",
    blend_fire_mode: str = "gate",
    blend_debug: bool = False,
    max_episode_steps: int = 13_500,
    controlled_agent_id: str = "agent_0",
    apply_split_normalization: bool = True,
    freeze_schedule_at_step: Optional[int] = None,
    reward_mode: str = "baseline",
) -> gym.Env:
    """
    Build one fully-wrapped B-ACE environment instance.

    This is the ONLY place the wrapper stack is assembled -- train_bace_clean.py and
    evaluate_experiment_seeds.py both call this function rather than each building their own
    copy. See this module's docstring for why that matters and what changed as a result.

    Args:
        bace_config: the loaded B-ACE JSON scenario config (EnvConfig/AgentsConfig).
        reward_mode: "baseline" (default) uses REWARD_SHAPING_STACK unchanged -- the exact
            reward this codebase already trains with, unmodified, for reproducibility.
            "mission_aligned" uses MISSION_ALIGNED_REWARD_STACK instead: the per-step
            mission-tempo survival bonus is zeroed out (it currently pays more for loitering
            with red dead than for finishing the mission -- see
            REWARD_SHAPING_STEPS_FORWARD.md Sec 1 for the quantified argument) and
            MissionOutcomeBonus is added (an explicit HVAA-loss penalty and a mission-success
            bonus sized to matter -- Sec 4.2/4.4 of the same doc). Both modes share every other
            wrapper unchanged, so a baseline-vs-mission-aligned comparison isolates exactly
            this change.
        is_eval: if True, skips ExpertAlignmentWrapper/ExpertActionBlendingWrapper (Factors D/E
            are training-only techniques; evaluation always uses the raw learned policy).
        log_dir_str: directory Monitor writes monitor.csv into.
        seed: seed passed to the final env.reset(seed=seed) call.
        apply_split_normalization: whether to add SplitNormalizationWrapper when
            use_enriched_obs=True. train_bace_clean.py always pairs enriched obs with split
            normalization (leave this at the default True). evaluate_experiment_seeds.py can
            evaluate older checkpoints trained before split normalization existed, where
            VecNormalize's own norm_obs=True was used instead -- for those, it passes False so
            it doesn't add a second, untrained normalization layer on top.
        freeze_schedule_at_step: if set, immediately calls set_global_step(freeze_schedule_at_step)
            on the constructed chain, so schedule-dependent wrappers (ActionSmoothnessPenalty,
            FireCooldownWrapper) start at that point in their anneal curve instead of step 0.
            train_bace_clean.py leaves this None (its callback advances the schedule live during
            training). evaluate_experiment_seeds.py passes FULLY_ANNEALED_STEP so evaluation
            reflects the trained policy's realistic, fully-annealed final discipline rather than
            the schedules' un-annealed starting values, which eval would otherwise never advance
            past (see IMPROVEMENT_PLAN_2_WRAPPERS_AND_DEPENDENCIES.md Task 2 discussion).
        (remaining args mirror train_bace_clean.py's CLI flags of the same name; see
        train_bace_clean.py's _parse_args() for the authoritative descriptions/defaults.)

    Returns:
        A single gym.Env with the full wrapper chain applied, already reset once.
    """
    e = SingleAgentBACEEnv(
        bace_config=bace_config,
        controlled_agent_id=controlled_agent_id,
        max_episode_steps=max_episode_steps,
    )

    ##############################
    # Apply enriched observation wrapper if enabled
    ##############################
    if use_enriched_obs:
        e = EnrichedObservationWrapper(
            e,
            ablation_config=ablation_config,
            normalize_features=False,
            smooth_threat_loss=True,
            debug=False
        )

        base_obs_dim = e.original_obs_dim
        print(f"  [Enriched Obs] Config: {ablation_config}")
        print(f"  [Enriched Obs] Base dim: {base_obs_dim}, Enriched dim: {e.observation_space.shape[0]}")
        print(f"  [Enriched Obs] Added features: {e.num_added_features}")

        if apply_split_normalization:
            e = SplitNormalizationWrapper(e, base_obs_dim=base_obs_dim)
            print(f"  [Split Normalization] Applied - base features normalized, enriched features passed through")
    else:
        print("\n" + "=" * 60)
        print("USING BASELINE OBSERVATIONS (NO THEORETICAL FEATURES)")
        print("=" * 60 + "\n")

    ##############################
    # The reward-shaping stack, as an explicit ordered (WrapperClass, kwargs) list rather than
    # imperative code -- this is the exact stack that had silently drifted apart between
    # train_bace_clean.py and evaluate_experiment_seeds.py before this refactor (see this
    # module's docstring). "What's the current order and configuration" is now a single
    # readable list instead of interleaved logic split across two files.
    ##############################
    REWARD_SHAPING_STACK = [
        (HeadingRateLimitWrapper, dict(hdg_idx=0, max_delta=0.15)),
        (KillRewardWrapper, dict(kill_reward=12.0)),
        (ActionSmoothnessPenalty, dict(
            hdg_idx=0,
            jerk_start=0.0, jerk_end=0.0025,
            mag_start=0.0, mag_end=0.012,
            hold_steps=1_500_000,
            anneal_steps=4_500_000,
        )),
        (MissionTempoShapingPerStep, dict(
            penalty_per_step=-0.003,
            bonus_per_step=0.006,
            require_hvaa_alive=True,
        )),
        (SustainedTurnPenaltySimple, dict(
            hdg_idx=0,
            window_size=15,
            turn_threshold=0.5,
            penalty_coef=0.01,
        )),
        (CombinedTrackAwareShaping, dict(
            range_idx=17,
            track_idx=26,
            track_loss_penalty=-0.3,
            k_close=40.0,
            asymmetry_ratio=2.0,
            clip_delta=0.02,
            max_valid_delta=0.05,
            min_range_for_penalty=0.05,
        )),
        (FireCooldownWrapper, dict(
            fire_idx=3,
            cd_start=70,
            cd_end=120,
            cd_hold=10_000,
            cd_steps=5_000_000,
        )),
        (HVAASurvivalEveryNStepsBonus, dict(every_n_steps=25, bonus=0.01)),
    ]

    ##############################
    # reward_mode="mission_aligned": identical to REWARD_SHAPING_STACK above except
    # MissionTempoShapingPerStep's bonus_per_step is zeroed (see this function's docstring and
    # REWARD_SHAPING_STEPS_FORWARD.md Sec 1 for why that term specifically), plus
    # MissionOutcomeBonus appended to reward the actual mission outcome directly. Everything
    # else -- action smoothness, turn penalty, track-aware shaping, fire cooldown, the small
    # per-N-step HVAA survival tick, and the per-step tempo *penalty* while red is alive (kept,
    # since it targets urgency-to-engage, not the loitering-after-kill behavior being removed --
    # is unchanged from baseline, so a comparison between the two modes isolates this one change.
    ##############################
    MISSION_ALIGNED_REWARD_STACK = [
        (HeadingRateLimitWrapper, dict(hdg_idx=0, max_delta=0.15)),
        (KillRewardWrapper, dict(kill_reward=12.0)),
        (ActionSmoothnessPenalty, dict(
            hdg_idx=0,
            jerk_start=0.0, jerk_end=0.0025,
            mag_start=0.0, mag_end=0.012,
            hold_steps=1_500_000,
            anneal_steps=4_500_000,
        )),
        (MissionTempoShapingPerStep, dict(
            penalty_per_step=-0.003,
            bonus_per_step=0.0,  # <-- the change: was 0.006 (see docstring above)
            require_hvaa_alive=True,
        )),
        (SustainedTurnPenaltySimple, dict(
            hdg_idx=0,
            window_size=15,
            turn_threshold=0.5,
            penalty_coef=0.01,
        )),
        (CombinedTrackAwareShaping, dict(
            range_idx=17,
            track_idx=26,
            track_loss_penalty=-0.3,
            k_close=40.0,
            asymmetry_ratio=2.0,
            clip_delta=0.02,
            max_valid_delta=0.05,
            min_range_for_penalty=0.05,
        )),
        (FireCooldownWrapper, dict(
            fire_idx=3,
            cd_start=70,
            cd_end=120,
            cd_hold=10_000,
            cd_steps=5_000_000,
        )),
        (HVAASurvivalEveryNStepsBonus, dict(every_n_steps=25, bonus=0.01)),
        (MissionOutcomeBonus, dict(hvaa_loss_penalty=-15.0, mission_success_bonus=25.0)),
    ]

    if reward_mode == "baseline":
        active_stack = REWARD_SHAPING_STACK
    elif reward_mode == "mission_aligned":
        active_stack = MISSION_ALIGNED_REWARD_STACK
    else:
        raise ValueError(f"Unknown reward_mode={reward_mode!r}; expected 'baseline' or 'mission_aligned'")

    print(f"  [Reward Mode] {reward_mode}")
    for wrapper_cls, wrapper_kwargs in active_stack:
        e = wrapper_cls(e, **wrapper_kwargs)

    ##############################
    # Apply expert alignment wrapper if enabled and not in eval mode
    ##############################
    if expert_alignment and not is_eval:
        expert = AggressiveHunterFixed(debug=False)
        obs_indices = get_default_obs_indices()
        alignment_decay = create_alignment_decay_fn(
            decay_start=alignment_decay_start,
            decay_end=alignment_decay_end,
            final_multiplier=0.3,
        )
        e = ExpertAlignmentWrapper(
            e, expert=expert, obs_indices=obs_indices,
            alignment_coef=alignment_coef,
            action_weights=np.array([1.0, 0.3, 0.6, 0.5]),
            decay_fn=alignment_decay,
        )
        print(f"Expert alignment wrapper added (TRAINING ONLY)")
    elif expert_alignment and is_eval:
        print(f"Expert alignment SKIPPED for evaluation environment")

    ##############################
    # Apply expert action blending wrapper if enabled and not in eval mode
    ##############################
    if expert_blending and not is_eval:
        if expert_alignment:
            print("WARNING: Both --expert-alignment and --expert-blending are enabled!")
            print("   These are different approaches and should not be combined.")
            print("   Proceeding with both, but results may be unpredictable.")

        expert_blend = AggressiveHunterFixed(debug=blend_debug)
        obs_indices_blend = get_default_obs_indices()

        alpha_decay = create_alpha_decay_fn(
            initial_alpha=blend_initial_alpha,
            final_alpha=blend_final_alpha,
            decay_start=blend_decay_start,
            decay_end=blend_decay_end,
            decay_type=blend_decay_type,
        )

        e = ExpertActionBlendingWrapper(
            e,
            expert=expert_blend,
            obs_indices=obs_indices_blend,
            alpha_fn=alpha_decay,
            blend_dims=[0, 1, 2],
            fire_blend_mode=blend_fire_mode,
            fire_gate_threshold=0.3,
            debug=blend_debug,
        )
        print(f"Expert action blending wrapper added (TRAINING ONLY)")
        print(f"   Blending formula: action = alpha*expert + (1-alpha)*PPO")
        print(f"   Alpha schedule: {blend_initial_alpha} -> {blend_final_alpha} ({blend_decay_type})")
        print(f"   Decay window: {blend_decay_start:,} -> {blend_decay_end:,} steps")
        print(f"   Fire mode: {blend_fire_mode}")
    elif expert_blending and is_eval:
        print(f"Expert action blending SKIPPED for evaluation environment")

    obs, _ = e.reset()
    print(f"\nEnvironment created")
    print(f"   Observation dimension: {obs.shape[0]}")
    print(f"   obs[0:10] = {np.round(obs[:10], 3)}")

    if use_enriched_obs and obs.shape[0] > 27:
        print(f"   Enriched features [27:]: {np.round(obs[27:], 3)}")

    e = CompactRewardLogger(e, print_every_n=10)
    e = NanGuardWrapper(e, name=("eval_env" if is_eval else "train_env"))
    e = _TermTrace(e)

    e = Monitor(e, filename=str(Path(log_dir_str) / "monitor.csv"))

    ##############################
    # Freeze schedule-dependent wrappers (ActionSmoothnessPenalty, FireCooldownWrapper) at a
    # fixed point in their anneal curve, e.g. for evaluation where nothing ever advances the
    # schedule live via set_global_step()
    ##############################
    if freeze_schedule_at_step is not None and hasattr(e, "set_global_step"):
        e.set_global_step(int(freeze_schedule_at_step))

    obs, _ = e.reset(seed=seed)
    print(f"[DEBUG] Final observation shape: {obs.shape}")
    return e


def get_split_norm_wrapper_from_vec_env(vec_env):
    """
    Extract the SplitNormalizationWrapper from a VecEnv wrapper stack.

    The wrapper stack is typically:
    VecNormalize -> DummyVecEnv -> [shaping wrappers] -> SplitNormalizationWrapper -> EnrichedObservationWrapper -> base_env

    Returns None if no SplitNormalizationWrapper is found.
    """
    ##############################
    # Get the base env from VecNormalize/DummyVecEnv
    ##############################
    if hasattr(vec_env, 'venv'):
        dummy_vec = vec_env.venv
    else:
        dummy_vec = vec_env

    ##############################
    # DummyVecEnv stores envs in .envs list
    ##############################
    if hasattr(dummy_vec, 'envs') and len(dummy_vec.envs) > 0:
        env = dummy_vec.envs[0]
    else:
        return None

    ##############################
    # Traverse the wrapper stack to find SplitNormalizationWrapper
    ##############################
    while env is not None:
        if type(env).__name__ == 'SplitNormalizationWrapper':
            return env
        ##############################
        # Move to the wrapped environment
        ##############################
        if hasattr(env, 'env'):
            env = env.env
        else:
            break

    return None
