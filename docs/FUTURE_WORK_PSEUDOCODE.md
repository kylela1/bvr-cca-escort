# Potential Extensions: Pseudocode, Locations, and Time Estimates

Design sketches for seven possible follow-on additions, grounded in the actual current code
(verified by reading the relevant files, not assumed). **Nothing here has been implemented or
applied** — every code block is illustrative pseudocode, and no existing file has been modified
by writing this document. Time estimates assume one person, one machine, sequential (not
clustered) training runs, building on everything already working in this repo.

---

## 1. Action blending (Factor E) — drop it, or replace it?

**Recommendation: drop it, and if you still want expert guidance, replace it with a
training-loss term instead of an action-level blend.**

**Why it currently hurts** (confirmed by reading `src/rl_files/expert_action_blending_wrapper.py`,
not just inferred from the pilot results): its own docstring says *"The BLENDED action is
executed, not the agent's raw action."* SB3's PPO samples an action, computes its log-prob/value
from that raw action, and stores `(obs, action, reward, next_obs)` in the rollout buffer —
but because the wrapper substitutes a different (expert-blended) action before it reaches the
simulator, the `reward`/`next_obs` actually came from an action PPO never evaluated the log-prob
of. This silently violates the on-policy assumption PPO's surrogate objective depends on, which
is a well-documented way to bias the policy gradient. This is a plausible, code-level explanation
for why Factor E underperformed baseline on every metric in the pilot — not just "it didn't
help," but a specific mechanism for why it would actively hurt.

**Where it lives**: `src/rl_files/expert_action_blending_wrapper.py` (the wrapper class),
`src/rl_files/env_factory.py` (applied only when `--expert-blending`, training-env only), and
the `--expert-blending`/`--blend-*` flags in `src/rl_files/train_bace_clean.py`.

**Option A — drop entirely** (no code sketch needed): delete the wrapper application and its
CLI flags. Lowest risk, consistent with the pilot evidence, and costs nothing to validate since
it's already off by default and no new training run is required to confirm "removing it doesn't
hurt" — it's already not in your baseline-vs-redesign comparison.

**Option B — replace with a behavior-cloning auxiliary loss** (keeps using the expert, but never
touches what action is actually executed, so PPO's on-policy bookkeeping stays correct):

```pseudocode
# New file: src/rl_files/expert_bc_auxiliary_ppo.py
# Subclasses SkippableTrainPPO (rollback_eval_callback.py) the same way that file
# already subclasses PPO -- NOT a gym.Wrapper, because this needs access to the
# policy's loss computation, not the environment step.

class ExpertGuidedPPO(SkippableTrainPPO):
    def __init__(self, *args, expert, bc_loss_coef_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self.expert = expert                      # AggressiveHunterFixed instance
        self.bc_loss_coef_fn = bc_loss_coef_fn     # same decay-schedule shape as alpha_fn today

    def train(self):
        # ... SB3's normal PPO.train() body, unchanged, EXCEPT:
        for rollout_batch in self.rollout_buffer.get(self.batch_size):
            policy_loss, value_loss, entropy_loss = ...  # existing PPO terms, unchanged

            expert_actions = self.expert.get_action(rollout_batch.observations, obs_indices)
            bc_loss = mse(self.policy.predict_action(rollout_batch.observations), expert_actions)
            coef = self.bc_loss_coef_fn(self.num_timesteps)   # decays to 0, same shape as today

            total_loss = policy_loss + value_loss * vf_coef + entropy_loss * ent_coef \
                         + coef * bc_loss
            # backprop total_loss, unchanged optimizer step
```

Executed action is always PPO's own sampled action — the expert only ever appears inside the
loss, never in the environment transition — so the rollout buffer stays internally consistent.

**Time estimate**: Option A: under an hour (deletion + a doc update), zero new experiments
needed. Option B: ~1 day implementation (new PPO subclass, needs careful testing that gradients
flow correctly and the coefficient schedule doesn't dominate the real PPO loss) + 1-3 full 10M
timestep training runs to see if it actually helps (~9-10 hrs each sequentially), so **~1-4 days
total** depending on how many seeds you run before concluding.

---

## 2. Establish a win-rate ceiling

**Recommendation**: run the existing scripted expert (`AggressiveHunterFixed`) directly as the
blue controller, with **no PPO policy at all**, across many episodes. This gives a concrete,
cheap "non-learned baseline" reference point your trained agents' win rate can be measured
against — not a hard theoretical ceiling, but a meaningful practical one, and it reuses code you
already have (`AggressiveHunterFixed.get_action(obs, obs_indices) -> np.ndarray`, confirmed in
`src/rl_files/aggressive_hunter_fixed.py`).

**Caveat worth stating explicitly in your paper**: your `my_first_run`/`reshaped_rewards1`
re-evaluations this session already hit 95-100% win rate at 20 episodes — if the scripted
expert *also* wins almost every episode, the scenario may already be close to saturated at
seed 42, and a ceiling study would show very little headroom to measure. That's itself a
useful finding (motivates item #4/#5 below — a harder red agent or maneuvering HVAA would be
needed before a ceiling study is actually informative).

**Where it would live**: new script `src/rl_files/evaluate_scripted_baseline.py`, reusing
`env_factory.build_env()` for the exact same wrapper stack/observation space as the real eval
script, swapping only the action source.

```pseudocode
# New file: src/rl_files/evaluate_scripted_baseline.py

def main():
    args = parse_args()   # --episodes, --config, mirrors evaluate_experiment_seeds.py's flags
    env = build_env(bace_config=load_b_ace_config(args.config), is_eval=True, ...)
    expert = AggressiveHunterFixed(...)            # same defaults as the D/E wrappers use

    episodes = []
    for _ in range(args.episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action = expert.get_action(obs, obs_indices)   # no model.predict() at all
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        episodes.append(EpisodeData(...))   # reuse evaluate_experiment_seeds.py's dataclass

    # reuse compute_seed_results()/t_distribution_ci() from stats_utils.py for the same
    # win-rate/HVAA/red-kill/blue-loss table format as the real eval script produces
```

**Time estimate**: ~2-4 hours implementation (mostly plumbing — reusing existing dataclasses
and stats helpers), ~15-30 minutes to run 20-100 episodes. **Well under a day total.**

---

## 3. Addressing the reward-redesign training-instability issue

Two tracks, not mutually exclusive — run both if time allows.

**Track A: it may just need more seeds.** A single seed cannot distinguish "mission-aligned
reward is inherently less stable late in training" from "this one seed happened to destabilize."
**No code changes needed** — this is purely a "run more experiments" fix:

```pseudocode
# No pseudocode needed -- literally:
python src/rl_files/run_experiment.py --experiment-name reshaped_rewards_multiseed \
    --seeds 42 123 456 --total-timesteps 10000000 --reward-mode mission_aligned
```
If the post-~6.6M-step decline reproduces across seeds → genuine code/design issue (go to Track
B). If it only happens on some seeds → still a real finding (the mission-aligned reward is less
seed-robust), worth reporting as-is.

**Track B: if it reproduces, the likely code-level cause and fix.** Hypothesis: the dense
per-step `MissionTempoShapingPerStep` bonus wasn't just causing misalignment — it was also
providing a smooth, continuous gradient signal that made PPO's value-function estimates easier
to keep well-calibrated. Zeroing it entirely (today's `MISSION_ALIGNED_REWARD_STACK`) may have
made the reward too sparse for stable late-training credit assignment. A **potential-based**
replacement would keep the reward dense without reintroducing the original misalignment (Ng et
al. 1999 potential-based shaping provably doesn't change the optimal policy, unlike a flat
survival bonus):

```pseudocode
# Modification sketch for src/rl_files/env_factory.py's MISSION_ALIGNED_REWARD_STACK
# (currently sets MissionTempoShapingPerStep's bonus_per_step=0.0 -- this replaces that
# entry with a new wrapper instead of leaving the reward sparse)

class MissionProgressPotentialShaping(gym.Wrapper):
    """reward += gamma * phi(next_state) - phi(state), phi = a mission-progress potential
    (e.g. -distance_to_mission_complete, or -(time elapsed since kill) capped at the old
    tempo-bonus magnitude) -- dense signal, zero net effect on the optimal policy."""
    def step(self, action):
        phi_before = self._mission_progress_potential(self._last_obs, self._last_info)
        obs, reward, terminated, truncated, info = self.env.step(action)
        phi_after = self._mission_progress_potential(obs, info)
        reward += self.gamma * phi_after - phi_before
        self._last_obs, self._last_info = obs, info
        return obs, reward, terminated, truncated, info
```

**Where it lives**: `src/rl_files/env_factory.py` (new wrapper class + swap it into
`MISSION_ALIGNED_REWARD_STACK`), a new file `src/rl_files/mission_progress_potential_shaping.py`
for the class itself (matching the pattern of the other shaping wrappers already in
`src/rl_files/`).

**Time estimate**: Track A: zero implementation, ~2-3 extra 10M-step runs × ~9-10 hrs each
sequentially = **~1-1.5 days of compute, no engineering time**. Track B: ~1 day implementation
(getting the potential function right is the hard part — a poorly-chosen potential can silently
reintroduce misalignment even though the theory says it shouldn't, so budget real debugging
time), then the same multi-seed validation as Track A on top = **~2-3 days total**.

---

## 4. Improve the scripted Red agent (evasion, weapons) — stay scripted, not learned

**Where it lives**: `Godot_Air_Combat/components/Fighter.gd` — this is GDScript, not Python, so
none of the existing Python tooling touches it. Confirmed by reading the file: `process_behavior()`
(~line 840) is Red's FSM, selected via the `base_behavior` config string (`"baseline1"` by
default). Red already has *some* evasion logic — a `tatic_status == "Evade"` state exists,
triggered by missile-threat detection (`HPT.threat_factor > lBreak * break_error`, ~line 974) —
so "add evasion" really means "make the existing evasion smarter," not building it from scratch.
Also worth knowing: `"baseline2"` is already a recognized, selectable config value, but reading
the code shows it differs from `"baseline1"` by exactly one line (altitude-tracking during
Engage, ~line 955) — it is **not** a meaningfully harder opponent today, despite existing as an
option.

```pseudocode
# Modification sketch inside Godot_Air_Combat/components/Fighter.gd's process_behavior()

# Smarter evasion: react earlier / with more appropriate geometry instead of a flat
# 180-degree-opposite turn (current: hdg_input = clamp_hdg(HPT.radial + 180)):
elif tatic_status == "Evade":
    notch_angle = compute_notch_heading(HPT.radial, own_velocity, missile_closure_rate)
    hdg_input = notch_angle            # beam/notch maneuver instead of a pure 180
    desiredG_input = scale_g_with_missile_range(HPT.range_to_missile)  # harder break when closer

# Better weapon employment: shoot-look-shoot / multiple missiles instead of one-and-done
if HPT.offensive_factor > dShot * shoot_range_error and missiles_remaining > 1:
    if not HPT.is_missile_support or time_since_last_shot > reshoot_cooldown:
        launch_missile_at_target(HPT)   # existing function, called under a new condition
```

**New config option** (`"baseline3"`, following the existing naming convention) rather than
modifying `baseline1` in place, so existing experiments stay reproducible.

**Time estimate**: GDScript changes + in-editor tuning/testing of thresholds (aspect angles, G
limits, cooldowns) is slower to iterate than pure Python — realistically **1-2 days**. Any blue
agent trained against the old red is no longer a fair comparison once red changes, so this
requires a **fresh full 10M-step training run** (and ideally multiple seeds) against the new red
to produce a meaningful before/after — **~1-2 more days of compute** on top. **~2-4 days total.**

---

## 5. Maneuvering HVAA instead of a straight line

**Where it lives**: also `Godot_Air_Combat/components/Fighter.gd`. Confirmed directly from the
code: `is_hvaa` forces a **constant heading/speed** (`const_hdg_enable`/`const_hdg_deg`, ~lines
1157-1226; the variable's own comment at line 143 literally says *"Constant heading (deg) if
is_hvaa"*) — this is exactly the "just going straight" behavior described.

```pseudocode
# Modification sketch inside Fighter.gd, near the existing is_hvaa branches (~line 1157+)

if is_hvaa:
    if hvaa_behavior == "straight":        # today's only option, left as default
        hdg_input = const_hdg_deg
    elif hvaa_behavior == "waypoints":
        hdg_input = get_hdg_2d(global_transform.origin, next_waypoint())
    elif hvaa_behavior == "evasive":
        if nearest_threat.threat_factor > hvaa_evade_threshold:
            hdg_input = clamp_hdg(nearest_threat.radial + 180)   # same evade primitive Red uses
        else:
            hdg_input = const_hdg_deg
```

**This is the costliest item on this list after #6/#7**, for a reason worth stating plainly: it
isn't just a Godot change. CCA's whole policy was trained assuming a stationary-heading HVAA, so
once the HVAA maneuvers, (a) the CCA's existing checkpoints are no longer meaningful and a fresh
training run is mandatory, not optional, and (b) you likely need to check whether the CCA can
even *perceive* the HVAA's heading/turn state — the current 27-dim baseline observation may not
expose it (check `src/b_ace_py/contracts.py`'s `EXPECTED_OBSERVATION_LABELS` and
`enriched_observation_wrapper_ablation.py` before assuming it's already there; if not, that's a
Python-side observation-space change too, on top of the Godot change).

**Time estimate**: Godot-side maneuver logic + tuning: **1-2 days**. Possible observation-space
addition + wrapper updates if HVAA state isn't already exposed: **+0.5-1 day**. Mandatory fresh
training (likely needs more iteration than a drop-in change, since you're changing what "good
behavior" even looks like for CCA): **2-3 days of compute/iteration**. **~1 week total,
realistically — budget more than you think.**

---

## 6. True multi-agent (2v2) capability

**This already has a detailed existing analysis in this repo** — `docs/CODEBASE_GUIDE.md` §3.2
— written by reading `SimManager`, `B_ACE_sync.gd`, and `B_ACE_GodotPettingZooWrapper` directly.
Don't re-derive this; read that section. Summary of what it found:

- The Godot/simulation layer is **already genuinely multi-agent** (scales with `num_agents` in
  the config, keys obs/reward/done by agent name already) — this is not a from-scratch sim
  rebuild.
- The real blocker is `SingleAgentBACEEnv` in `src/rl_files/env_factory.py`: it broadcasts the
  *one* PPO action identically to every agent in the underlying env (confirmed:
  `{agent: clipped for agent in self._ma_env.agents}`) and discards every observation/reward
  except one controlled agent's.
- Every single-track reward wrapper (`track_aware_shaping.py`, `mission_tempo_shaping_perstep.py`)
  assumes exactly one enemy/one protected asset and needs an aggregation rule for N targets.
- `rollback_eval_callback.py`'s eval loop hardcodes a single-env `VecEnv` and would crash outright
  at `n_envs > 1`.
- A real, currently-silent bug (the missile team-check issue, §3.1) gets much more likely to
  bite with 2 co-located blue aircraft.

```pseudocode
# Sketch of the core fix in src/rl_files/env_factory.py (per CODEBASE_GUIDE.md §3.2's
# recommended path: shared-policy CTDE, since that's what the thesis's own lit review suggests)

class MultiAgentBACEEnv:
    """Wraps ONE B_ACE_GodotPettingZooWrapper, exposes a PettingZoo ParallelEnv-style
    interface instead of collapsing to a single controlled_agent_id."""
    def step(self, action_dict):   # {agent_0: action, agent_1: action}, NOT one broadcast action
        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self._ma_env.step(action_dict)
        return obs_dict, rew_dict, term_dict, trunc_dict, info_dict   # no longer discarded

# Training loop (new, e.g. src/rl_files/train_multi_agent.py) needs an actual MARL
# paradigm -- shared policy network across both blue agents, centralized critic sees both
# agents' observations concatenated (CTDE), rather than two fully independent PPO models.
```

**Time estimate**: per CODEBASE_GUIDE.md's own suggested order of attack (fix the missile bug
first, pick the MARL paradigm, then rewrite `SingleAgentBACEEnv`, then the reward wrappers, then
the eval loop last) — this touches 5+ interacting files across both Godot and Python. Realistic
engineering estimate: **1.5-3 weeks**. A 2v2 joint action/observation space is harder to learn
than 1v1, so expect to need more timesteps per run (15-20M+) and more debugging iterations before
anything trains stably — **another 1-2 weeks of compute/iteration**. **~3-5 weeks total — this
is thesis-chapter scope, not a quick class-project addition.**

---

## 7. Self-play / learning Red agent

**The biggest lift on this list.** Confirmed directly from code: `B_ACE_GodotPettingZooWrapper`'s
controllable agent roster is derived *only* from `blue_agents.num_agents`
(`self._num_agents = int(self.agents_config["blue_agents"].get("num_agents", 1))`) — Red is
never in `self.agents`/`self.possible_agents` at all today, and on the Godot side, `Fighter.gd`
only accepts external action commands when `behavior == "external"` — Red is always
`"baseline1"`/`"baseline2"` (FSM-driven), never externally controlled. Making Red learnable means
changing both sides of the Python↔Godot boundary, not just the Python training loop.

```pseudocode
# (1) Godot-side change needed: Fighter.gd must accept behavior == "external" for red
#     agents too (today this works for blue only in practice) -- config change, not shown.

# (2) B_ACE_GodotPettingZooWrapper (src/b_ace_py/B_ACE_GodotPettingZooWrapper.py):
self._num_agents = (
    int(self.agents_config["blue_agents"].get("num_agents", 1))
    + int(self.agents_config["red_agents"].get("num_agents", 1))   # NEW: red included too
)
# self.agents now contains both sides' IDs -- every self.possible_agents loop in this file
# (there are ~15 of them, per the earlier grep of this file) needs to stay agent-agnostic,
# which it already mostly is, EXCEPT wherever blue/red are currently assumed implicitly.

# (3) New self-play training loop, e.g. src/rl_files/train_self_play.py:
blue_policy = PPO(...)
red_policy = PPO(...)                     # or: red_policy = snapshot of blue_policy, mirrored
opponent_pool = [red_policy.get_parameters()]   # league-style snapshotting, not just latest

for iteration in range(n_iterations):
    opponent = sample_from(opponent_pool)          # fixed for this rollout batch
    rollout = collect_rollout(env, blue_policy, opponent)   # both sides act each step
    blue_policy.train_on(rollout)                   # standard PPO update, blue's perspective only
    if iteration % snapshot_every == 0:
        opponent_pool.append(blue_policy.get_parameters())   # mirror/self-play snapshot
```

**Time estimate**: Godot protocol change + Python wrapper change to make Red controllable:
**~1 week** (touches the socket protocol, not just a config flag). Self-play training loop +
opponent-pool management: **~1-2 weeks** (competitive self-play is notoriously prone to
non-stationarity/cyclic strategies — "get it running" and "get it to converge to something
sensible" are very different bars). Compute: self-play typically needs substantially more
timesteps and multiple full iteration-and-retune cycles than single-opponent training — budget
**2-3+ weeks of compute/iteration**, likely more. **~4-6+ weeks total — this is squarely
thesis-scale work**, not a class-project addition; frame it in your paper as a named future-work
direction rather than something attempted this quarter.

---

## Summary table

| # | Change | Implementation | Experiments (compute) | Total (rough) |
|---|---|---|---|---|
| 1a | Drop action blending | <1 hr | none needed | **<1 day** |
| 1b | Replace with BC auxiliary loss | ~1 day | 1-3 runs × ~9-10 hrs | **1-4 days** |
| 2 | Win-rate ceiling (scripted baseline) | ~2-4 hrs | ~30 min | **<1 day** |
| 3a | More seeds on reward redesign | none | 2-3 runs × ~9-10 hrs | **~1-1.5 days** |
| 3b | Potential-based shaping fix | ~1 day | +multi-seed validation | **~2-3 days** |
| 4 | Improve scripted Red | ~1-2 days (GDScript) | 1+ fresh 10M runs | **~2-4 days** |
| 5 | Maneuvering HVAA | ~1.5-3 days (Godot+obs) | 2-3 days iteration | **~1 week** |
| 6 | 2v2 multi-agent | ~1.5-3 weeks | ~1-2 weeks | **~3-5 weeks** |
| 7 | Self-play learning Red | ~2-3 weeks | ~2-3+ weeks | **~4-6+ weeks** |

Items 1-3 are realistic to actually complete and report results for within a class-project
timeline. Item 4 and especially 5 are feasible but tight. Items 6-7 should be scoped as *named,
well-justified future work* in your paper rather than attempted — the guidance rewards a
well-reasoned future-work section, and the CODEBASE_GUIDE.md §3.2 analysis already referenced
above is itself good evidence you understand exactly what 2v2 would require, without needing to
have built it.
