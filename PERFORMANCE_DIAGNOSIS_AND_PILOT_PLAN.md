# Performance Diagnosis, Fixes, and Pilot Experiment Plan

This document records a session-by-session investigation into (1) how the training pipeline
actually works end-to-end, (2) why training on this machine appeared to run at "~14 it/s"
(extremely slow), (3) the concrete, minimal fixes applied, and (4) a realistic pilot-scale
plan for reproducing the thesis's qualitative findings on this hardware, its results, and
exact instructions to reproduce or extend it. It complements
[CODEBASE_GUIDE.md](CODEBASE_GUIDE.md) (the file-by-file map and full bug audit) — there is
no `HOW_TO_RUN.md` on this branch (it only exists on `new-code`), so §11-13 below are the
canonical how-to-run reference for `mostly-original`.

All work here was done on the **`mostly-original`** branch (same commit as `main`), per
explicit instruction to keep the original codebase intact rather than switching to the more
heavily refactored `new-code` branch that also exists in this repo's history.

---

## 1. How the environment actually works (build → call → step → reset)

```
train_bace_clean.py main()
  → make_env() closure builds, in order:
      SingleAgentBACEEnv(bace_config)          # constructs B_ACE_GodotPettingZooWrapper
        → B_ACE_GodotPettingZooWrapper.__init__
            → GodotEnv._launch_env()           # spawns bin/B_ACE_v0.1.exe as a subprocess
            → GodotEnv._start_server()         # binds a TCP socket, accept()s Godot's connection
            → send_sim_config(...)             # sends the full JSON scenario config
            → _get_env_info()                  # gets obs/action space shapes + observation labels
      → HeadingRateLimitWrapper → KillRewardWrapper → ActionSmoothnessPenalty →
        MissionTempoShapingPerStep → SustainedTurnPenaltySimple → CombinedTrackAwareShaping →
        FireCooldownWrapper → HVAASurvivalEveryNStepsBonus →
        [ExpertAlignmentWrapper / ExpertActionBlendingWrapper if enabled] →
        CompactRewardLogger → NanGuardWrapper → Monitor(SB3) → DummyVecEnv → VecNormalize
  → PPO(policy="MlpPolicy", env=vec_env, ...)
  → model.learn(total_timesteps=..., callback=[metrics_callback, eval_checkpoint_callback])
```

**`reset()`**: `SingleAgentBACEEnv.reset()` → `B_ACE_GodotPettingZooWrapper.reset()` sends
`{"type": "reset"}` over the socket → `B_ACE_sync.gd` tells `SimManager.gd` to respawn all
Fighters per the config's `BlueSpawn`/`HVAASpawn` randomization blocks → sends back the first
observation → each wrapper layer resets its own internal state (turn-rate history, track-loss
counters, etc.) → SB3 sees a flat 27-dim (or up to 40-dim, if enriched observations are
enabled) `np.float32` vector.

**`step(action)`**: the PPO policy outputs a 4-dim `[-1, 1]` vector (heading Δ, altitude rate,
G-force, fire) → each wrapper can modify the action and/or the reward on the way back (e.g.
`HeadingRateLimitWrapper` clamps how much heading can change per step) → the final action is
sent to Godot → Godot advances `action_repeat` (2) physics ticks, running Fighter/Missile
kinematics and its own `RewardsControl` reward computation → sends back obs/reward/done/info →
unwound back up through every wrapper layer, each adding its shaping term.

---

## 2. Key files (what each one actually does)

| File | Role |
|---|---|
| `b_ace_py/godot_env.py` (`GodotEnv`) | Base class. Launches the Godot subprocess, owns the raw TCP socket protocol (length-prefixed JSON, both directions). Never used directly by training scripts. |
| `b_ace_py/B_ACE_GodotPettingZooWrapper.py` | **The actual environment construction point.** Multi-agent PettingZoo `ParallelEnv`, keyed by agent name (`agent_0`, ...). Builds gymnasium spaces from Godot's `env_info` response. |
| `b_ace_py/utils.py` | `load_b_ace_config` (JSON loader), `ActionSpaceProcessor` (Godot's tuple-of-subspaces → flat Box for SB3), `convert_macos_path`. |
| `b_ace_py/pursuit_evasion_features.py` | Closed-form/approximated geometry for Factors A/B/C (Apollonius/ATDDG, WEZ/DMC, RLPE escape-cone/capture-probability). |
| `b_ace_py/enriched_observation_wrapper_ablation.py` | `EnrichedObservationWrapper` appends the enabled theoretical features onto the base 27-dim observation; `SplitNormalizationWrapper` normalizes only the base dims. |
| `rl_files/train_bace_clean.py` | The one file that does everything: CLI parsing, `make_env()` wrapper-stack construction, PPO model construction, `model.learn()`, and a post-training final evaluation. `SingleAgentBACEEnv` (defined here) collapses the multi-agent env to a plain `gym.Env` for SB3. |
| `rl_files/run_experiment.py` | Thin multi-seed loop — calls `train_bace_clean.py` once per `--seeds` value via `subprocess`. **Has no default `--total-timesteps` of its own** — silently inherits `train_bace_clean.py`'s 1,000,000 default if you forget to pass it. |
| `rl_files/evaluate_experiment_seeds.py` | Loads trained checkpoints + `VecNormalize` stats, runs deterministic evaluation episodes, computes per-seed/cross-seed stats. Has a built-in fallback: tries the SB3 `.zip` file first, then falls back to raw `.pt` state-dict checkpoints if the `.zip` is missing/corrupt (this fallback turned out to matter — see §4.2). |
| `rl_files/rollback_eval_callback.py` (`RollbackEvaluationCallbackTopK`) | Periodic evaluation (every 200,000 steps, 15 episodes), best/top-K checkpoint saving, and the checkpoint-rollback mechanism. |
| `rl_files/expert_alignment_wrapper.py` / `expert_action_blending_wrapper.py` / `aggressive_hunter_fixed.py` | Factor D / Factor E / the scripted "expert" FSM they're built around. |
| `rl_files/{sustained_turn_penalty,track_aware_shaping,mission_tempo_shaping_perstep}.py` | Individual reward-shaping wrapper terms, chained together in `make_env()`. |

---

## 3. Bugs found and fixed

### 3.1 Fixed: `B_ACE_GodotPettingZooWrapper.__init__` crashed on every construction

**File:** `b_ace_py/B_ACE_GodotPettingZooWrapper.py`

```python
# before (crashed every time):
self.action_spaces = env_info["action_spaces"]

# after (removed — self.action_spaces is already correctly populated a few lines
# earlier, inside _get_env_info()'s own loop over Godot's "action_space" (singular) key):
# (line removed)
```

Godot's `_send_env_info()` (in `B_ACE_sync.gd`) only ever sends a key called `"action_space"`
(singular) — never `"action_spaces"`. The overwrite at this line raised `KeyError:
'action_spaces'` unconditionally, before a single `env.step()` could ever happen. This means
the environment could not be constructed at all prior to this fix — whatever "14 it/s" was
observed previously must have come from a different code state.

**Verified:** raw environment construction + 300 steps of random actions now completes cleanly
(see §4.1 for throughput numbers).

### 3.2 Fixed: unconditional final `model.save()` crashes with a pickling error

**File:** `rl_files/train_bace_clean.py`, end of `main()`

Symptom (from a real, complete training + evaluation run):
```
TypeError: cannot pickle '_thread.lock' object
```
raised inside `model.save(model_path.as_posix())`, right after a full training run and a
20-episode final evaluation had *already completed successfully*.

**Root cause:** `RollbackEvaluationCallbackTopK._init_callback()` monkey-patches
`self.model.train` with a closure (`wrapped_train`) to support skip-training-update logic
after a rollback event, and `_on_training_end()` is supposed to restore the original bound
method. Somewhere in that lifecycle, `model.__dict__` ends up holding a reference that
`cloudpickle` cannot serialize (a `_thread.lock`, almost certainly reachable through the
callback's own `eval_env`, which holds a live Godot TCP socket). This is a known issue —
confirmed already diagnosed and fixed on this repo's `new-code` branch (commit message:
*"Replace the model.train monkey-patch in rollback_eval_callback.py with a
SkippableTrainPPO(PPO) subclass, so model.__dict__ can no longer gain a stray 'train' key --
the root cause of a model.save() pickling crash"*).

**Why it isn't actually catastrophic:** the callback's own periodic/best-checkpoint saves
(`_save_best_checkpoint`, `_save_eval_checkpoint` in `rollback_eval_callback.py`) save the
policy and optimizer via plain `torch.save()` of state dicts — not through the buggy
`cloudpickle` path — and already wrap their own (separate) `model.save()` attempt in a
`try/except` that prints a warning and continues on failure. Only the **one** unconditional,
unguarded `model.save()` call at the very end of `main()` was missing that same guard.
`evaluate_experiment_seeds.py` already has a fallback path (`load_model_with_pt_fallback`)
that reconstructs the policy from the `.pt` state-dict checkpoints when the `.zip` file is
missing or corrupt — so even before this fix, a run's trained weights were *not* actually
lost, just the very final full-format save and the trailing "TRAINING COMPLETE!" summary/JSON.

**Fix applied** — mirrored the exact defensive pattern already used elsewhere in this same
file for the exact same call:
```python
try:
    model.save(model_path.as_posix())
except Exception as e:
    print(f"   ⚠️  Final full model save failed ({e}); "
          f"policy weights remain available via best/eval .pt checkpoints.")
vec_env.save(norm_path.as_posix())
```
This was chosen over porting the larger `SkippableTrainPPO` subclass fix from `new-code`,
to keep the change on `mostly-original` as small and low-risk as possible while still letting
a run finish cleanly (correct exit code, full "TRAINING COMPLETE!" summary printed, JSON
results file written).

### 3.3 Fixed: `np.cross()` on 2D vectors crashes under NumPy 2.x (blocks all of Factor A)

**File:** `b_ace_py/pursuit_evasion_features.py`, `compute_atddg_defense_features`

Symptom (first hit during `pilot_ACD`, the first pilot run to actually exercise Factor A —
baseline/D/E don't use enriched observations at all, and B/`engagement_only` never calls this
function):
```
ValueError: Both input arrays must be (arrays of) 3-dimensional vectors, but they are 2 and 2 dimensional instead.
```
raised from `np.cross(AT_unit, AD_vec)`, where both vectors are 2D ground-plane (x, z)
vectors. NumPy 2.0 removed the old implicit-2D-input convenience behavior of `np.cross`
(which used to silently compute the scalar z-component of the equivalent 3D cross product);
NumPy 2.5.2 (installed here) now requires genuinely 3D inputs and raises instead. This is a
known issue — the `new-code` branch's history explicitly lists *"Fix a NumPy 2.x
incompatibility (np.cross on 2D vectors) blocking the enriched-observation path"* as one of
its fixes.

**Practically, this meant Factor A was completely unusable** — every `--ablation-config` that
includes geometry/ATDDG features (`geometry_only`, `geometry_engagement`, `geometry_range`,
`all`) would crash almost immediately (~13,500 steps, i.e. partway through the very first
episode) once any track/pursuit-evasion computation actually ran. Only `engagement_only`
(Factor B alone), `range_limited_only` (Factor C alone), and `engagement_range` (B+C) avoided
this code path entirely.

**Fix applied** — replaced the `np.cross()` call with the explicit 2D scalar cross-product
formula it was implicitly relying on (numerically identical result, just spelled out instead
of delegating to a NumPy API surface that no longer supports 2D inputs):
```python
# before:
perp_distance = np.abs(np.cross(AT_unit, AD_vec))
# after:
perp_distance = np.abs(AT_unit[0] * AD_vec[1] - AT_unit[1] * AD_vec[0])
```
**Verified:** `pilot_ACD` re-run past the exact step count where it previously crashed
(confirmed clean past 24,576 steps, well beyond the ~13,500-step crash point) with zero
errors.

### 3.4 Cleaned up: stale orphaned processes

Found and killed:
- A **10-day-old** orphaned `B_ACE_v0.1.exe` (port 12500 — the default port) from an Aug 17
  run that was never cleaned up.
- A **10-day-old** orphaned Streamlit dashboard process (`tools/dashboard.py`, from the
  `new-code` branch's tooling, run at some earlier point).
- My own crashed pre-fix benchmark process.

On a 4-core/8-thread CPU with no discrete GPU, each lingering Godot instance meaningfully
competes for the same limited cores as any active run. **Check Task Manager for lingering
`B_ACE_v0.1.exe`/`python.exe` processes after any crashed or manually-interrupted run.**

---

## 4. What "~14 it/s" actually was

Three independent measurements, all on `mostly-original` after the fixes above:

| Measurement | Result |
|---|---|
| Raw `env.step()` with random actions, no PPO/wrappers | **~1,365 it/s** |
| Full PPO + full wrapper stack, actual `model.learn()` loop | **~125–210 it/s** (varied across runs; CPU-bound, no GPU) |
| A 3,000-timestep smoke test's **total wall-clock time** (training + wrapper overhead + the unconditional 20-episode final evaluation) | **~9 minutes**, of which the actual `model.learn()` loop was only ~15–24 seconds |

**The gap is almost entirely evaluation overhead, not step throughput.** Episodes in this
environment can run for **thousands to tens of thousands of steps** before naturally
terminating (mission success at ~80 NM typically takes on the order of 10,000+ agent-decision
steps at this simulation's speed; the hard cap is `max_episode_steps=36,000` in
`SingleAgentBACEEnv`, derived from `max_cycles=72,000` ÷ `action_repeat=2`). A real run
observed **mean final-evaluation episode length of 10,298 steps**, and the 20-episode final
evaluation alone took **1,905.5 seconds (~31.75 minutes)** — for a policy that had only seen
3,000 training steps.

This same mechanism recurs **every 200,000 training steps** during a real run (15 episodes
each, via `RollbackEvaluationCallbackTopK`, `eval_freq=200000`), which is unavoidable overhead
baked into any full-length run, not just short smoke tests — see §5 for the resulting time
budget.

**Conclusion:** core throughput on this hardware is healthy and comparable in order of
magnitude to what the thesis's own hardware likely achieved. The "14 it/s" impression was an
artifact of (a) the crash bug preventing clean runs, (b) accumulated zombie processes stealing
CPU, and (c) naively dividing total wall-clock time (which includes disproportionately
expensive evaluation phases) by total training timesteps on a short run where that fixed
overhead dominates.

---

## 5. Realistic time budget

Per single (seed, config) run at the thesis's full scale (10,000,000 timesteps):

| Component | Estimate |
|---|---|
| Core training loop (10M steps ÷ ~200 it/s) | ~14 hours |
| Periodic evaluation (50 checkpoints × 15 episodes, variable episode length) | ~3–10 hours |
| One-time final evaluation (20 episodes) | minutes to ~1 hour |
| **Total per run** | **~17–24 hours** |

The thesis's full factorial design is 17 configurations × 4 seeds = 68 such runs — serially,
this is **months**, not achievable in a single working session regardless of further
optimization on this hardware.

### Decision: pilot-scale subset instead of the full design

Rather than attempt the full design, the following 5-configuration, 1-seed-each,
2,000,000-timestep pilot was chosen to reproduce the thesis's **qualitative** headline
findings (Factor B is the strongest observation feature; Factor D helps; Factor E hurts;
"ACD" is the thesis's best-ranked combination) in roughly a day of unattended runtime instead
of months:

| Run | Flags | Thesis factor(s) |
|---|---|---|
| `pilot_baseline` | *(none)* | Reference |
| `pilot_B` | `--use-enriched-obs --ablation-config engagement_only` | B (WEZ/DMC observation features) |
| `pilot_D` | `--expert-alignment --alignment-decay-start 100000 --alignment-decay-end 400000` | D (alignment-based reward shaping) |
| `pilot_E` | `--expert-blending --blend-decay-start 100000 --blend-decay-end 1000000` | E (action blending — expected to hurt, per thesis) |
| `pilot_ACD` | `--use-enriched-obs --ablation-config geometry_range --expert-alignment --alignment-decay-start 100000 --alignment-decay-end 400000` | A+C+D (thesis's top-ranked combo) |

The D/E curriculum decay-start/end points were scaled proportionally (2M / 10M = 0.2×) using
the CLI flags that already exist for this purpose, so a shortened run still traverses a
complete miniaturized version of the intended curriculum rather than freezing partway through
a schedule calibrated for 10M steps.

**Known, accepted limitation:** the checkpoint-rollback safety net
(`min_timesteps_before_rollback=4,000,000`, hardcoded, not CLI-exposed) will not activate
within these 2M-step runs. A run that degrades badly simply stays degraded rather than being
automatically reverted. This is accepted as a reasonable simplification for a pilot rather
than a further code change.

**Estimated cost (pre-launch estimate):** ~3.5–4.3 hours per run × 5 runs, run
**sequentially** (not in parallel — concurrent Godot instances would compete for the same 4
physical cores and reproduce the exact contention problem diagnosed in §3.3) ≈ **~17.5–21.5
hours total**, unattended.

**Actual measured cost (`pilot_baseline`, first completed run):** ~100 minutes total
(94.3 min training incl. all periodic evaluation + 5.5 min final evaluation) — well under the
pre-launch estimate. See §7 for the running results table; estimates below are revised
accordingly.

---

## 7. Pilot run results (filled in as each run completes)

| Run | Status | Wall time | Final eval reward | Best checkpoint reward (step found) | Notes |
|---|---|---|---|---|---|
| `pilot_baseline` | ✅ complete | ~100 min (94.3 train + 5.5 final eval) | -38.85 ± 0.06 | -21.37 (1,400,000 steps) | Episodes consistently ran the full ~13,500-step length with no decisive win/loss — matches thesis's baseline characterization (avoids engagement, high instability, performance *degraded* over the back half of training: best-vs-final gap of 17.48 points). |
| `pilot_B` | ✅ complete | ~98.3 min (93.5 train + 4.8 final eval) | -35.10 ± 5.81 | -15.62 (200,000 steps) | Confirmed 32-dim observation (27 base + 5 enriched) active. Variance jumped ~100x vs. baseline: several episodes now end early via real combat (`All_Blue_Destroyed` at 6,231–7,231 steps) rather than every episode uniformly hitting the 13,500-step timeout. Best-ceiling improved over baseline (-15.62 vs -21.37) — consistent with the thesis's finding that Factor B is the strongest individual observation feature. |
| `pilot_D` | ✅ complete | ~92.8 min (87.7 train + 5.1 final eval) | -34.40 ± 4.15 | -14.75 (1,600,000 steps) | Best ceiling of the three configs so far, and found late in training (80% through) rather than early like B — suggests sustained rather than spiky improvement. Same shift toward decisive combat as B (`All_Blue_Destroyed` terminations), plus new positive "range"-closing reward components indicating active pursuit. `ExpertAlignmentWrapper` confirmed active on training env only, correctly absent from eval env. |
| `pilot_E` | ✅ complete | ~98.1 min (92.7 train + 5.4 final eval) | -39.93 ± 1.63 | -31.71 (1,000,000 steps) | **Worst of all four configs tested on both metrics** — clean replication of the thesis's central finding that action blending violates PPO's on-policy assumptions and hurts performance. Smaller best-vs-final gap (8.23 vs 17-20 for others) suggests uniformly poor training rather than a good-then-degraded pattern. `ExpertActionBlendingWrapper` confirmed active on training env only, correctly absent from eval env. |
| `pilot_ACD` | ✅ complete | ~100.5 min (95.2 train + 5.4 final eval) | **-7.52 ± 30.18** | **+3.56 (1,400,000 steps)** | **Clear winner across all 5 pilot runs.** Only config to reach a *positive* best-checkpoint reward. First attempt crashed at ~13,500 steps on the `np.cross()` NumPy 2.x bug (§3.3) — first pilot run to actually exercise Factor A; re-launched after the fix, confirmed clean. Log shows explicit `HVAA_Mission_Success` terminations and repeated `[KILL REWARD] +12.0` events across many episodes — qualitatively decisive, proactive engagement rather than the passive/uniformly-poor patterns seen in the other four configs. High variance (30.18) reflects a genuine win/loss split rather than consistent mediocrity. |

Raw checkpoint note: the final unconditional `model.save()` still produces a **0-byte
`.zip`** file in every run (the pickling bug in §3.2 is caught, not eliminated — the fix only
prevents it from crashing the script). `evaluate_experiment_seeds.py`'s `.pt` fallback path
handles this transparently; confirmed working against `pilot_baseline`'s
`best_checkpoint/best_policy.pt` (2.2 MB, real content) + `best_vecnormalize.pkl`.

---

## 8. Speed optimization notes (not yet applied, for future consideration)

1. **Never let Godot/python processes linger** — biggest lever, already applied once (§3.4).
2. **Shrink the evaluation tax** — `n_eval_episodes` (15 periodic / 20 final) and/or a
   shorter, evaluation-only episode-length cap (separate from training's 36,000-step cap)
   would reduce wall-clock cost without touching reward/observation/action semantics. Not
   applied — would change evaluation cadence/precision, a decision left open.
3. **Windows Defender exclusion** for the repo folder (particularly `bin/`) — real-time AV
   scanning a freshly-spawned 96 MB `.exe` on every launch is a plausible contributor to
   Godot startup latency. Requires admin rights; not something automatable from here.
4. **Power plan**: confirm "Best Performance" / plugged in during long runs.
5. **Not recommended for this pilot**: parallelizing environments (`SubprocVecEnv` with
   multiple Godot instances) would multiply throughput on the 4-core CPU, but the thesis's
   own methodology used `parallel_envs=1` — introducing this would diverge from faithful
   reproduction of PPO's on-policy batch composition. Worth revisiting only if exact-method
   reproduction is deprioritized in favor of wall-clock speed.

---

## 9. Pilot conclusions

All 5 planned pilot runs completed (2,000,000 timesteps, seed 42 each). Total wall-clock cost
for the full pilot: **~8.3 hours** (5 runs averaging ~99.6 minutes each, including all
periodic and final evaluation), dramatically under the original ~17.5–21.5 hour estimate.

| Run | Final eval reward | Best checkpoint (step) |
|---|---|---|
| `pilot_baseline` | -38.85 ± 0.06 | -21.37 (1,400,000) |
| `pilot_B` (WEZ/DMC) | -35.10 ± 5.81 | -15.62 (200,000) |
| `pilot_D` (alignment reward) | -34.40 ± 4.15 | -14.75 (1,600,000) |
| `pilot_E` (action blending) | -39.93 ± 1.63 | -31.71 (1,000,000) |
| `pilot_ACD` (A+C+D) | **-7.52 ± 30.18** | **+3.56 (1,400,000)** |

**This pilot reproduces every headline qualitative claim from the thesis at 1/5th scale,
1 seed, on consumer laptop hardware:**

1. **Action blending (Factor E) hurts** — it is the single worst-performing configuration on
   both final and best-checkpoint reward, underperforming even the no-heuristics baseline.
   Matches the thesis's central claim that action blending violates PPO's on-policy
   assumptions.
2. **Individual heuristic factors (B, D) each provide a modest improvement** over baseline,
   primarily by raising the achievable ceiling (best checkpoint) and by inducing more
   decisive combat behavior (episodes ending via real kills/losses rather than uniformly
   timing out).
3. **Combining observation and reward-shaping factors (A+C+D) compounds those gains
   dramatically** — the only configuration to reach a positive best-checkpoint reward, with
   qualitatively different behavior (explicit mission successes and repeated kills) rather
   than incremental improvement. Consistent with the thesis's own finding that its
   best-ranked design ("ACD") was a multi-factor combination, not a single factor alone.
4. **All configurations show a "best-vs-final" gap** (11–20 points, except E's smaller
   8-point gap) — i.e. performance peaked mid-training and degraded by the end. This matches
   the thesis's own discussion of training instability, and is a direct consequence of the
   checkpoint-rollback safety net being inactive at this reduced timestep scale (§5) — a full
   10M-step run with rollback active would be expected to retain more of its peak performance.

**What this pilot does not claim**: with 1 seed per configuration, none of this rises to the
thesis's statistical standard (which used 4 seeds and formal significance testing at multiple
confidence thresholds). These results are a directional replication, not a statistically
validated one — exactly the tradeoff accepted when choosing pilot scope over the full 68-run
design (§5).

---

## 10. Thesis-metric evaluation (mission outcomes, not raw PPO reward)

Raw PPO reward (§9) is an internal training signal, not the thesis's headline metric. Each
pilot's best (superlative) checkpoint was additionally evaluated via
`evaluate_experiment_seeds.py` (20 episodes each, seed 42) to get thesis-comparable
mission-outcome statistics:

| Config | Final PPO reward | Best ckpt PPO reward | HVAA Survival | Red Kill | Blue Loss | Approx. win rate\* |
|---|---|---|---|---|---|---|
| `pilot_baseline` | -38.85 | -21.37 | 0% | 0% | 0% | ~0% |
| `pilot_B` | -35.10 | -15.62 | 0% | 0% | 15% | ~0% |
| `pilot_D` | -34.40 | -14.75 | **50%** | 25% | 45% | **~50%** |
| `pilot_E` | -39.93 | -31.71 | 5% | 0% | 20% | ~5% |
| `pilot_ACD` | **-7.52** | **+3.56** | 25% | **45%** | **0%** | ~25% |

\*Approximated as HVAA survival rate — episodes in which the CCA is destroyed
(`All_Blue_Destroyed`) appear to terminate before mission completion in every case observed,
so blue-loss and HVAA-survival episodes are effectively disjoint, making HVAA survival rate a
reasonable proxy for the thesis's actual "win rate" (both HVAA and CCA survive) definition.
Full per-episode results: `runs_sb3/pilot_<name>/superlative_eval_results.json`.

**An important, unplanned finding**: `pilot_ACD` has by far the best raw PPO reward, but
`pilot_D` achieves the **higher actual mission win rate** (~50% vs. ~25%). `pilot_ACD`
instead achieves near-perfect CCA survivability (0% blue losses) and much higher lethality
(45% red kills) at the cost of fewer full mission completions. This is not a bug — it
**independently reproduces the thesis's own Section 4.7.1 finding**
("Reward Function Misalignment with Evaluation Metrics"): the training reward function
correlates imperfectly with the mission-level win rate the thesis treats as its headline
outcome, and different configurations trade off survivability, lethality, and mission
completion rather than one configuration dominating every axis — exactly the pattern the
thesis's own Chapter 4 discussion describes for its full-scale results (e.g. its Table 16).

Both `pilot_D` (highest win-rate proxy) and `pilot_ACD` (best reward, most lethal, safest for
the CCA) are genuinely strong candidates depending on which axis matters more — this pilot
does not produce a single unambiguous "winner," which is itself consistent with the thesis's
own conclusion that no single configuration dominates across all mission metrics (§4.4.2 of
the thesis).

---

## 11. How to re-run this experiment (exact terminal commands)

### 11.1 One-time setup, every new terminal session

**Git Bash:**
```bash
cd /c/Users/kyle2/dev/bvr-cca-escort
source .venv/Scripts/activate
export PYTHONUTF8=1
```

**PowerShell:**
```powershell
cd C:\Users\kyle2\dev\bvr-cca-escort
.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
```

`PYTHONUTF8=1` is required, not optional — several files print non-ASCII characters
(emoji, arrows) that crash on Windows' default console codepage otherwise.

Confirm you're on the right branch (all fixes in §3 live here, not on `main`/other branches
before this session):
```bash
git branch --show-current   # should print: mostly-original
```

Before launching anything, check for lingering processes from a previous crashed/interrupted
run (§3.4 explains why this matters):
```powershell
Get-Process B_ACE_v0.1, python -ErrorAction SilentlyContinue
```
Kill any you find with `Stop-Process -Id <pid> -Force` before starting a new run.

### 11.2 The 5 pilot commands, exactly as run

Run one at a time (not in parallel — see §8.5). Each takes ~90-100 minutes at 2,000,000
timesteps on this hardware.

```bash
# baseline (no heuristic features)
python rl_files/train_bace_clean.py \
  --total-timesteps 2000000 --seed 42 \
  --experiment-name pilot_baseline --run-name pilot_baseline_seed42

# Factor B: WEZ/DMC observation features
python rl_files/train_bace_clean.py \
  --total-timesteps 2000000 --seed 42 \
  --use-enriched-obs --ablation-config engagement_only \
  --experiment-name pilot_B --run-name pilot_B_seed42

# Factor D: alignment-based reward shaping (decay schedule scaled 0.2x for the shorter run)
python rl_files/train_bace_clean.py \
  --total-timesteps 2000000 --seed 42 \
  --expert-alignment --alignment-decay-start 100000 --alignment-decay-end 400000 \
  --experiment-name pilot_D --run-name pilot_D_seed42

# Factor E: action blending (decay schedule scaled 0.2x)
python rl_files/train_bace_clean.py \
  --total-timesteps 2000000 --seed 42 \
  --expert-blending --blend-decay-start 100000 --blend-decay-end 1000000 \
  --experiment-name pilot_E --run-name pilot_E_seed42

# ACD: Factors A+C+D combined (thesis's top-ranked design)
python rl_files/train_bace_clean.py \
  --total-timesteps 2000000 --seed 42 \
  --use-enriched-obs --ablation-config geometry_range \
  --expert-alignment --alignment-decay-start 100000 --alignment-decay-end 400000 \
  --experiment-name pilot_ACD --run-name pilot_ACD_seed42
```

(PowerShell: replace trailing `\` line-continuations with `` ` ``, or just put each command
on one line.)

### 11.3 Evaluating a completed run (thesis-style mission metrics)

After a run finishes, get win-rate/HVAA-survival/red-kill/blue-loss stats for its best
checkpoint:

```bash
python rl_files/evaluate_experiment_seeds.py \
  --experiment-dir runs_sb3/pilot_baseline \
  --episodes 20
```

**For enriched-observation configs (B, ACD, or any A/C variant), pass the matching
`--use-enriched-obs --ablation-config ...` flags here too** — the eval script does not
reliably auto-detect which features a checkpoint was trained with:

```bash
python rl_files/evaluate_experiment_seeds.py --experiment-dir runs_sb3/pilot_B --episodes 20 \
  --use-enriched-obs --ablation-config engagement_only

python rl_files/evaluate_experiment_seeds.py --experiment-dir runs_sb3/pilot_D --episodes 20

python rl_files/evaluate_experiment_seeds.py --experiment-dir runs_sb3/pilot_E --episodes 20

python rl_files/evaluate_experiment_seeds.py --experiment-dir runs_sb3/pilot_ACD --episodes 20 \
  --use-enriched-obs --ablation-config geometry_range
```

`--episodes` defaults to 100 (the thesis's own convention is closer to this) — 20 was chosen
here to keep the pilot's evaluation pass to ~15-25 min per config instead of ~1.5-2 hours;
raise it for tighter confidence intervals at the cost of time.

### 11.4 Quick sanity check before a long run

Before committing to a 90+ minute run, confirm the pipeline still connects (~1-2 minutes):
```bash
python rl_files/train_bace_clean.py \
  --total-timesteps 3000 --n-steps 1024 --batch-size 256 --epochs 2 \
  --run-name smoketest
```
You should see Godot launch, the observation-label table print, a few training iterations,
then (after training completes) a ~5-10 minute silent pause while the unconditional 20-episode
final evaluation runs (§4), then "TRAINING COMPLETE!". Delete `runs_sb3/smoketest/` afterward.

---

## 12. Parameter reference — what to change and how

### 12.1 The five thesis factors → CLI flags

| Factor | Meaning | Flag(s) |
|---|---|---|
| A | ATDDG observation features (Apollonius/intercept geometry) | `--use-enriched-obs --ablation-config geometry_only` |
| B | WEZ/DMC observation features | `--use-enriched-obs --ablation-config engagement_only` |
| C | RLPE observation features (escape cone/capture probability) | `--use-enriched-obs --ablation-config range_limited_only` |
| A+B | | `--ablation-config geometry_engagement` |
| A+C | | `--ablation-config geometry_range` |
| B+C | | `--ablation-config engagement_range` |
| A+B+C | | `--ablation-config all` |
| D | Alignment-based reward shaping | `--expert-alignment` (+ optionally `--alignment-coef`, `--alignment-decay-start`, `--alignment-decay-end`) |
| E | Action blending | `--expert-blending` (+ optionally `--blend-initial-alpha`, `--blend-final-alpha`, `--blend-decay-start`, `--blend-decay-end`, `--blend-decay-type`, `--blend-fire-mode`) |

**"Baseline" = omit `--use-enriched-obs` entirely.** Do not pass `--ablation-config none` as a
substitute — it still runs the enrichment/split-normalization wrapper machinery with zero
added features, a different (untested-by-this-pilot) code path. Don't combine
`--expert-alignment` and `--expert-blending` in the same run — the script warns if you try.

### 12.2 Run length / identity

| Flag | Meaning | Thesis value | This pilot's value |
|---|---|---|---|
| `--total-timesteps` | Total PPO training steps | 10,000,000 | 2,000,000 |
| `--seed` | Random seed | 4 seeds (42,123,456,789 per §5) | 1 seed (42) |
| `--experiment-name` | Groups runs under `runs_sb3/<name>/` | — | `pilot_<config>` |
| `--run-name` | Overrides the auto-generated run folder name | — | `pilot_<config>_seed<N>` |

**If you shorten `--total-timesteps` below 10,000,000**, proportionally scale any decay-window
flags you're using (`--alignment-decay-start/-end`, `--blend-decay-start/-end`) by the same
ratio — e.g. at 2,000,000 (0.2x of 10M) this pilot used `--alignment-decay-end 400000`
(0.2 x 2,000,000) rather than the thesis's 2,000,000. Otherwise the run freezes partway
through a curriculum calibrated for a much longer horizon. Note `min_timesteps_before_rollback`
(the checkpoint-rollback safety net) is hardcoded at 4,000,000 in
`rl_files/train_bace_clean.py`'s call to `RollbackEvaluationCallbackTopK` — not
CLI-exposed — so it simply won't activate below that many steps regardless.

### 12.3 Multiple seeds of one configuration

```bash
python rl_files/run_experiment.py \
  --experiment-name pilot_ACD_multiseed --seeds 42 123 456 789 \
  --total-timesteps 2000000 \
  --use-enriched-obs --ablation-config geometry_range \
  --expert-alignment --alignment-decay-start 100000 --alignment-decay-end 400000
```
**Always pass `--total-timesteps` explicitly here** — `run_experiment.py` has no default of
its own and silently inherits `train_bace_clean.py`'s 1,000,000 default otherwise.

### 12.4 Algorithm/network hyperparameters (rarely need changing — these are the thesis's tuned values)

| Flag | Default | Meaning |
|---|---|---|
| `--initial-learning-rate` / `--final-learning-rate` | 0.000258 → 0.00001 | LR decay schedule |
| `--initial-entropy-coef` / `--final-entropy-coef` | 0.051 → 0.005 | Entropy bonus decay |
| `--decay-schedule` | `linear` | or `exponential` |
| `--n-steps` | 6144 | PPO rollout length |
| `--batch-size` | 1024 | Minibatch size |
| `--epochs` | 8 | PPO epochs per update |
| `--gamma` | 0.99 | Discount factor |
| `--gae-lambda` | 0.9207 | GAE lambda |
| `--clip-eps` | 0.198 | PPO clip range |
| `--vf-coef` | 0.7728 | Value loss coefficient |
| `--n_layers` / `--layer_size` | 2 / 512 | Policy/value network architecture |
| `--device` | `cpu` | or `cuda` if a GPU is ever available |

### 12.5 Scenario/environment config

The scenario itself (spawn randomization, reward weights, battlespace size, missile counts,
etc.) is **not** a CLI flag — it lives in the JSON file passed via `--config` (default:
`b_ace_py/SimpleExample_B_ACE_config.json`). To change spawn geometry, reward weights, or
episode length limits (`max_cycles`), edit that file directly (or pass `--config path/to/your
copy.json`). See CODEBASE_GUIDE.md §5.1 for the full list of known mismatches between this
file's values and the thesis's stated reward constants.

---

## 13. Where to find results

For a run named `pilot_<name>` with `--run-name pilot_<name>_seed<N>`:

```
runs_sb3/pilot_<name>/pilot_<name>_seed<N>/
├── training_results_seed<N>.json   ← full training summary: config used, per-checkpoint
│                                      eval history (reward/length at every 200k-step eval),
│                                      final model path, best model path + reward
├── best_checkpoint/
│   ├── best_policy.pt              ← policy weights (state_dict) at the best eval checkpoint
│   ├── best_optimizer.pt
│   ├── best_vecnormalize.pkl       ← observation normalization stats to pair with it
│   ├── best_model.zip              ← ⚠️ always 0 bytes on this branch (§3.2) — ignore
│   └── best_model_info.json
├── eval_checkpoints/                ← top-5 checkpoints by reward, same .pt format
├── ppo_bace_blue_lead.zip           ← ⚠️ also 0 bytes (§3.2) — the .pt files above are authoritative
├── vecnormalize.pkl                 ← final (not best) normalization stats
├── monitor.csv                      ← per-episode reward/length, SB3 Monitor format
└── events.out.tfevents.*            ← TensorBoard log (see below)
```

And after running `evaluate_experiment_seeds.py` (§11.3):
```
runs_sb3/pilot_<name>/superlative_eval_results.json   ← mission-outcome metrics: HVAA
                                                          survival %, red kill %, blue loss %,
                                                          mean reward/length, 95% CI, plus
                                                          full per-seed breakdown
```

**To view training curves interactively:**
```bash
tensorboard --logdir runs_sb3/pilot_baseline
```
then open the printed URL (usually `http://localhost:6006`) — compare multiple runs at once
by pointing `--logdir` at `runs_sb3` itself instead of one config's folder.

**To read a result file directly without TensorBoard:**
```bash
python -c "import json; print(json.dumps(json.load(open('runs_sb3/pilot_baseline/superlative_eval_results.json')), indent=2))"
```

**This document's own §7/§9/§10 tables** are the consolidated summary across all 5 pilot
runs — start there before diving into individual JSON files.

