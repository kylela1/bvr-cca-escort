# Reward Shaping: Steps Forward for the Project Proposal

This document is scoped specifically to your project goal: redesign the reward function to
better reflect mission outcomes, compare it against the existing baseline reward, and evaluate
both by win rate. It identifies exactly where to look in this codebase, what's already known
to be wrong, and a concrete architecture for running the comparison cleanly.

Companion docs: [CODEBASE_GUIDE.md](CODEBASE_GUIDE.md) (full file-by-file map and bug audit —
read this first if you haven't), [PILOT_EXPERIMENT_RESULTS.md](PILOT_EXPERIMENT_RESULTS.md)
(reference results from the existing reward, run before this branch's env_factory.py existed).

---

## 1. The headline finding: a concrete, quantified mechanism for the misalignment

I traced through the exact numbers, and there is a specific, identifiable reason the current
reward doesn't track mission outcome well — not just a vague "rewards are messy" concern.

**`env_factory.py`'s `REWARD_SHAPING_STACK` includes `MissionTempoShapingPerStep` with
`penalty_per_step=-0.003, bonus_per_step=0.006`** (`rl_files/mission_tempo_shaping_perstep.py`).
While red is alive, the agent loses 0.003 every step. Once red is killed, the agent gains
0.006 every step *for as long as the episode continues and the HVAA is alive* — regardless of
whether the mission is actually progressing toward completion.

Do the arithmetic over a full ~13,500-step episode (the length baseline episodes consistently
ran during this session's pilot):

- **Never killing red**: -0.003 x 13,500 = **-40.5**. This alone accounts for almost the
  entirety of the baseline configuration's observed reward of -38.85 in this session's pilot —
  the per-step tempo penalty *is* most of what "baseline reward" currently measures.
- **Killing red at step 1,000, then surviving to the end**: -0.003 x 1,000 + 0.006 x 12,500 =
  **+72.0**, before even counting the one-time `KillRewardWrapper` bonus (+12) or Godot's own
  `hit_enemy_factor` (+10).
- **Compare to actually completing the mission**: `mission_accomplished_factor` in
  `b_ace_py/SimpleExample_B_ACE_config.json` is **+8.0**, a *one-time* terminal reward — and
  critically, completing the mission *ends the episode early* (`"stop_mission":1`), which
  **cuts off further tempo-bonus accumulation**.

**The reward function currently pays an agent more for killing red early and then loitering
for the rest of a long episode than for efficiently completing the mission.** A policy that
delays mission completion while farming the per-step survival bonus is, under this reward,
doing the *right* thing — even though it's the opposite of what "win rate" (HVAA reaches its
objective) actually measures. This is a very plausible root cause of the reward/win-rate
misalignment this session's pilot observed independently (see `PILOT_EXPERIMENT_RESULTS.md`
§2 — the ACD configuration had by far the best raw reward but a lower win-rate proxy than the
D configuration).

## 2. A second, compounding gap: HVAA loss isn't penalized at all right now

`CODEBASE_GUIDE.md` §5.2 documents this as a known bug, still present on this branch (it's in
the Godot-side GDScript, untouched by the Python refactor): `Sim_assets.gd`'s
`RewardsControl.add_final_episode_reward()` has a fully-implemented `"HVAA_Destroyed"` branch,
but nothing in `SimManager.gd` ever triggers it — the only termination condition that checks
HVAA status also requires the CCA itself to be dead. If the HVAA alone is destroyed, the
episode just continues (reporting `hvaa_alive=false` every step) until `Max_Cycles` eventually
ends it.

Two more compounding facts, verified directly against this branch's current files:
- `SimpleExample_B_ACE_config.json` sets `hvaa_loss_factor: 0.0` — even if the termination path
  were fixed, the configured penalty is currently zero.
- The Python-side compensating wrapper mentioned in older audits
  (`hvaa_destruction_penalty.py`) **no longer exists on this branch at all** — it was removed
  as confirmed-dead code during the `new-code` refactor (it was imported but never
  instantiated on `mostly-original` either, so nothing is lost). There is currently **no
  mechanism anywhere, Python or Godot, that penalizes losing the HVAA** while the CCA survives.

This is arguably the single most direct piece of prior work relevant to your proposal: fixing
this is close to *literally* "redesigning the reward to better reflect mission outcomes."

## 3. The metric gap: nothing computes win rate

Confirmed on this branch (`rl_files/evaluate_experiment_seeds.py`): there is no `win_rate`
computation anywhere in the file. It reports `HVAA Survival Rate`, `Red Kill Rate`, and
`Blue Loss Rate` as separate per-episode aggregates, but never their conjunction (HVAA
survives *and* CCA survives — the thesis's actual win-rate definition). This session's pilot
derived it by hand from the other two metrics. Since win rate is your project's primary
comparison criterion, this needs to be a first-class computed metric, not something inferred
after the fact — see §5.3 for where to add it.

---

## 4. Recommended architecture for your reward-redesign comparison

`env_factory.py`'s docstring makes the intent explicit: *"This is the ONLY place the wrapper
stack is assembled."* Follow that same pattern for your contribution rather than editing the
existing stack in place — you want to be able to run both the existing reward and your
redesigned one from the same codebase, on demand, for a clean ablation.

### 4.1 Add a reward-mode switch, mirroring the existing `--ablation-config` pattern

`build_env()` in `rl_files/env_factory.py` (around line 703, `REWARD_SHAPING_STACK`) currently
hardcodes one fixed list of `(WrapperClass, kwargs)` tuples. Introduce a parameter — e.g.
`reward_mode: str = "baseline"` — and branch to a second list, e.g. `MISSION_ALIGNED_STACK`,
when `reward_mode == "mission_aligned"`. Thread it through:
- `build_env()`'s signature (new parameter)
- `train_bace_clean.py`'s `_build_arg_parser()` (new `--reward-mode {baseline,mission_aligned}`
  flag, forwarded to `build_env()` in `main()`'s `make_env()` closure)
- `evaluate_experiment_seeds.py`'s equivalent construction call, so evaluation uses the same
  reward mode the checkpoint was trained under

This gets you `python rl_files/train_bace_clean.py --reward-mode mission_aligned ...` as a
direct, symmetric counterpart to the existing `--ablation-config` flags — consistent with how
this codebase already lets you toggle observation factors, and trivially compatible with the
existing YAML launcher (`configs/*.yaml`) for tracking your two conditions as named,
reproducible configs (e.g. `configs/reward_baseline.yaml`, `configs/reward_mission_aligned.yaml`).

### 4.2 What to actually change in the mission-aligned variant

Based on §1-2's findings, concrete candidates, roughly in order of expected impact:

1. **Rebalance or remove the per-step tempo bonus**, or cap its total accumulation, so it can't
   out-earn actually completing the mission. A simple first experiment: set `bonus_per_step` to
   0 (keep only the pre-kill urgency penalty) and see whether win rate changes.
2. **Fix the HVAA-destruction termination bug on the Godot side**
   (`Godot_Air_Combat/SimManager.gd`) so `"HVAA_Destroyed"` alone ends the episode and actually
   applies `hvaa_loss_factor`, and set that config value to something meaningfully negative in
   your mission-aligned config (or a copy of `SimpleExample_B_ACE_config.json`). This is a
   GDScript change, not Python — budget time to re-export `bin/B_ACE_v0.1.exe` after editing
   it (see `CODEBASE_GUIDE.md` §4.1 for the export command).
3. **Increase the relative weight of terminal, mission-outcome rewards** (`mission_accomplished_factor`,
   `hit_enemy_factor`) relative to dense per-step shaping, so the terminal signal isn't
   drowned out over a ~13,500-step episode.
4. Consider whether `stop_mission: 1` (episode ends immediately on mission success) is still
   the right setting once tempo-bonus accumulation is fixed — right now it's working *against*
   quick mission completion by cutting off the bonus stream early; once that perverse incentive
   is removed, ending the episode on success is exactly correct and should stay.

**Recommendation on scope**: change one thing at a time and re-measure, rather than redesigning
everything simultaneously — you'll want to be able to attribute a win-rate change to a specific
mechanism in your writeup, which is much harder if three reward terms change at once.

### 4.3 Add win rate as a first-class metric

In `evaluate_experiment_seeds.py`, wherever per-episode outcomes are aggregated (search for
where `hvaa_survival`/`blue_loss`-equivalent per-episode booleans are computed — look for the
`EpisodeData`-style structure `CODEBASE_GUIDE.md` §5.6 references), add a `win = hvaa_survived
and not blue_destroyed` boolean per episode, and report its mean alongside the existing
HVAA/Red-Kill/Blue-Loss rates in both the per-seed and cross-seed aggregate tables. This
directly matches the thesis's own win-rate definition (both survive) and makes it the
headline number in your comparison rather than something reconstructed after the fact.

---

## 4.4 Bug found and fixed while investigating `my_first_run`'s results (2026-08-31)

`my_first_run`'s `.zip` checkpoints (`ppo_bace_blue_lead.zip`, `best_checkpoint/best_model.zip`)
were 0 bytes — the same `TypeError: cannot pickle '_thread.lock' object` failure
`SkippableTrainPPO` (`rollback_eval_callback.py`) was specifically built to eliminate, but
through a *different* path than the one it fixed.

**Root cause**: `RollbackEvaluationCallbackTopK._create_scaled_lr_schedule()` returns a closure
assigned directly onto `model.learning_rate`/`model.lr_schedule` whenever the rollback
mechanism adjusts the learning rate — and unlike the old `train`-monkey-patch bug
`SkippableTrainPPO` fixed, **this assignment is never undone**. The closure read `self.min_lr`
directly, so it captured `self` (the callback) — which holds `self.eval_env`, a live Godot
socket — permanently reachable from `model.__dict__` for the rest of training.

**Why nothing caught this earlier**: it only manifests once at least one rollback actually
fires. `RollbackEvaluationCallbackTopK`'s `min_timesteps_before_rollback` is 4,000,000 — none
of this session's 2,000,000-timestep pilot runs (§7 below / `PILOT_EXPERIMENT_RESULTS.md`)
could have triggered it. Only `my_first_run`'s full 10,000,000-timestep run did (5 times,
`rollback_history.json`). This is directly relevant to your plan: **any full-scale
baseline-vs-mission-aligned comparison run is likely to hit this exact path**, since real
training instability (which is what triggers rollback) is common in this environment (Figure 1).

**Fix applied**: capture `min_lr` and `base_schedule` as plain local values before defining the
closure, instead of reading `self.min_lr`/`self._base_lr_schedule` from inside it — so the
returned closure has no reference to the callback or anything it holds. Verified with an
isolated reproduction (constructed the exact old/new closure patterns against a fake object
holding a real `threading.Lock`, `cloudpickle`'d both): the old pattern fails with the exact
same error observed in `my_first_run`; the new pattern serializes cleanly. Not yet confirmed
against a real multi-hour run that actually triggers rollback — worth watching for on your
next full-scale run.

**Why this matters before you start reward-shaping specifically**: if you're running full-scale
comparisons (§6's scope question), you want your baseline-vs-mission-aligned runs to produce
*loadable* `.zip` checkpoints, not just `.pt` fallbacks — a clean `PPO.load()` is what you'd
want for any follow-on work (continued training, deployment, or your actual thesis building on
this project's output). Confirming this fix holds under a real rollback-triggering run is worth
doing once, deliberately, before treating it as settled.

## 5. Suggested order of attack

1. Read `env_factory.py`'s `build_env()` and `REWARD_SHAPING_STACK` top to bottom (§4.1) —
   you'll be modifying this file's structure directly.
2. Read `Godot_Air_Combat/assets/Sim_assets.gd`'s `RewardsControl` class and
   `Godot_Air_Combat/SimManager.gd`'s termination-condition logic — this is the other half of
   the reward, and where the HVAA-destruction bug (§2) actually lives.
3. Add the `win_rate` metric to `evaluate_experiment_seeds.py` (§4.3) — do this *before* running
   any comparison, so both your baseline and mission-aligned runs are measured against it from
   the start rather than needing re-evaluation later.
4. Establish a fresh baseline run under this branch (`--reward-mode baseline`, once implemented,
   or simply the current unmodified stack) — don't reuse the `mostly-original`-branch pilot
   numbers as your official baseline; see `PILOT_EXPERIMENT_RESULTS.md` §3 for why.
5. Implement the reward-mode switch (§4.1) and your first mission-aligned variant (§4.2, one
   change at a time), starting with the tempo-bonus rebalance — it's a pure Python/config
   change (no Godot re-export needed) and directly targets the largest, most clearly-quantified
   misalignment found in §1.
6. Run the baseline-vs-mission-aligned comparison, multiple seeds each, using `win_rate` as the
   primary comparison metric (not raw PPO reward — see §1 for why that's specifically the
   metric your redesign should not be judged by).

---

## 6. Plan check: "redesign the reward, retrain, compare results" as the final project

Confirmed as of 2026-08-31: this matches the project proposal almost verbatim ("redesign the
agent's reward function... compare the existing baseline reward with a more mission-aligned
reward and test the win rate"), and it's a sound, appropriately-scoped final project — narrower
than the full thesis (as the proposal itself says), but a complete, self-contained
contribution with a clear before/after comparison and a headline metric that isn't reward
itself. `figures/fig4_reward_vs_winrate_rank.png` and `fig5_reward_misalignment_illustration.png`
already build the motivating case for *why* this redesign is needed; that pairs well with a
final write-up structured as: motivate (Figures 1-5) → redesign (this section) → compare
(new figures, same format).

**To make "then compare the results" actually convincing, not just directionally
suggestive, three things need to happen before the retrain**, all already detailed above —
restated here as a checklist since they're easy to skip in the rush to "just retrain and see":

- [ ] **`win_rate` exists as a computed metric** (§4.3) before either run happens. Without it,
      "compare the results" reduces to reward numbers again — the exact thing Figure 4 shows
      can't be trusted on its own.
- [ ] **The reward-mode switch is implemented in `env_factory.py`** (§4.1), not a hand-edited
      copy of the file per condition — otherwise "baseline" and "mission-aligned" risk silently
      diverging on something other than the reward (an easy mistake to make and hard to notice
      after the fact).
- [ ] **The baseline run being compared against is a fresh one on this branch**
      (`my_first_run`, Figure 1, or a new equivalent run under the reward-mode switch) — not the
      `mostly-original`-branch pilot numbers, which used a different, narrower evaluation stack
      (`PILOT_EXPERIMENT_RESULTS.md` §3).

**One scope decision worth making explicitly, not by default**: `my_first_run` took ~10.4 hours
for 10,000,000 timesteps. If both the baseline and mission-aligned runs need to be full-scale
for the comparison to carry weight, that's ~21 hours of unattended compute minimum for 1 seed
each — plausible for a final project timeline, but worth deciding now rather than discovering
mid-run. The alternative, consistent with how this session validated the approach before
committing to full scale: repeat the earlier pilot pattern (2,000,000 timesteps, ~100 minutes
per run) to get a fast first read on whether the redesigned reward moves `win_rate` in the
expected direction, then commit to a full-scale run only once that signal looks real.

**Recommended first change, concretely**: implement just the tempo-bonus rebalance from §4.2
item 1 (e.g. `bonus_per_step: 0.006 -> 0.0`, or a hard cap on total accumulated bonus) as the
entire first mission-aligned variant. It's a one-line change in `env_factory.py`'s
`REWARD_SHAPING_STACK`, it's the change §1's arithmetic most directly targets, and isolating it
means a `win_rate` change afterward can be attributed to a specific, namable mechanism in the
final write-up — rather than "we changed several things and it got better."
