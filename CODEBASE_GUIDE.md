# BVR-CCA-Escort Codebase Guide

A working guide to the code behind AFIT-ENS-MS-26-M-085 (Bieger, "A Factorial Analysis of
Heuristic Feature Design and Training Guidance in Reinforcement Learning for
Beyond-Visual-Range Escort Missions"). This document explains what each part of the repo
does, how data actually flows end-to-end, how to set up the environment, and — in detail —
where the code diverges from what the thesis text describes. It was produced by reading
every Python file in full and auditing the Godot/GDScript simulation source.

> **Update (environment setup):** the environment-setup section (§4) has since been fully
> executed on this machine — Godot exported, Python environment installed, and a real training
> run confirmed working end-to-end. Three small, previously-undetected bugs that only surfaced
> by actually running the code (not visible from reading it) were fixed as part of that; see
> §4.4/§5.2b.
>
> **Update (wrappers/dependencies refactor):** the two follow-on improvement plans this guide's
> findings fed into (`IMPROVEMENT_PLAN_1_VISUALIZATION_AND_RUNNING.md`,
> `IMPROVEMENT_PLAN_2_WRAPPERS_AND_DEPENDENCIES.md`) have since been fully implemented and
> verified. This resolved several findings §5 originally flagged as problems: the wrapper-stack
> is no longer duplicated between training and evaluation (§2.3, §5.3), ~900 lines of dead code
> listed in §5.4 have been deleted rather than just left inert, the `model.train` monkey-patch
> bug's root cause is now structurally impossible (§5.2b), and several §5.7 code-quality gaps
> (no `requirements.txt`, no tests, some docstring mismatches) are closed. **The rest of this
> document — the reward-constant mismatches (§5.1), the functional Godot-side bugs (§5.2), the
> approximated-theory notes (§5.5), and the statistical-machinery gaps (§5.6) — is unaffected**;
> those are analysis/design findings about what the code *computes*, not the tooling refactor
> described here, which deliberately changed no reward, observation, or training behavior
> (verified via a structural equivalence-check tool, `tools/equivalence_check.py`, run before
> and after every change). Sections below are updated in place to reflect the new state, with
> superseded content marked rather than silently deleted where it's still useful context.

> **How to read this document.** Section 1 is the map. Section 2 is "what do I change to do
> X." Section 3 is setup (now done — see above). Section 4 is the full issues list — read it
> before you trust any specific number quoted in the thesis against this code.

---

## 1. Repository Map

Three top-level components, corresponding to three different parts of the thesis:

```
bvr-cca-escort/
├── b_ace_py/              Python <-> Godot bridge + theoretical feature engineering
│                          (thesis §3.2.1 "State and Observation Variables", Appendix B)
├── rl_files/               PPO training, reward shaping, experiment orchestration, evaluation
│                          (thesis §3.3-3.4 "Environment Configuration"/"PPO", Ch. IV)
└── Godot_Air_Combat/       The actual B-ACE simulation (Godot 4.5 project)
                           (thesis §3.1-3.2 "Problem Description", environment mechanics)
```

**How training actually happens, end to end:**

```
run_experiment.py  (loops over seeds, forwards CLI flags)
      │
      ▼
train_bace_clean.py  (builds env, wrapper stack, PPO model, runs model.learn())
      │  SingleAgentBACEEnv wraps →
      ▼
B_ACE_GodotPettingZooWrapper  (PettingZoo ParallelEnv semantics)
      │  inherits from →
      ▼
GodotEnv (b_ace_py/godot_env.py)  — launches Godot subprocess, owns the TCP socket
      │  length-prefixed JSON over 127.0.0.1:<port>, Godot is the CLIENT, Python is the SERVER
      ▼
Godot_Air_Combat/addons/godot_rl_agents/B_ACE_sync.gd  — the real protocol handler
      │
      ▼
Godot_Air_Combat/SimManager.gd  — scenario orchestration (spawns/resets/steps/rewards)
      │
      ▼
Godot_Air_Combat/components/Fighter.gd + Missile.gd  — kinematics, sensors, weapons
```

A critical, non-obvious point that recurs throughout this guide: **the reward function is
not implemented as one equation anywhere.** It is assembled additively across three layers
that were all built/tuned somewhat independently, and (per the audit below) frequently
disagree with each other and with the thesis text:

1. Godot's own `RewardsControl` class (`Godot_Air_Combat/assets/Sim_assets.gd`), which computes
   a substantial, already-shaped scalar reward every physics tick from a `RewardsConfig` block
   in the JSON scenario config.
2. A stack of ~8-10 Python `gym.Wrapper` classes applied in a fixed order inside
   `rl_files/env_factory.py`'s `build_env()` function (both `train_bace_clean.py` and
   `evaluate_experiment_seeds.py` call this one function now — see §2.2/§2.3), each doing
   `reward = reward + shaping`.
3. A handful of hardcoded constants inside `SimManager.gd`/`Fighter.gd` that aren't exposed
   through any config path at all (e.g. the HVAA mission-success distance).

If you want to change "the reward function," you have to touch layers 1 and 2 (and know layer
3 exists), not edit one file.

---

## 2. Folder-by-Folder Breakdown

### 2.1 `b_ace_py/` — Python↔Godot bridge and feature engineering

| File | Purpose |
|---|---|
| `godot_env.py` | `GodotEnv`: launches the Godot binary as a subprocess, opens a TCP socket, and implements the raw `{handshake, reset, action, config, env_info, close}` message protocol. Base class of `B_ACE_GodotPettingZooWrapper`; never used directly by the training scripts. |
| `B_ACE_GodotPettingZooWrapper.py` | `B_ACE_GodotPettingZooWrapper(GodotEnv, ParallelEnv)`: the multi-agent PettingZoo-compatible environment. Sends the JSON scenario config to Godot, parses the returned per-agent observation/action spaces and observation *labels*, and exposes `reset()`/`step()` keyed by agent name (`"agent_0"`, `"agent_1"`, ...). **This class is already genuinely multi-agent** — see §2.4. |
| `utils.py` | `load_b_ace_config()` (JSON config loader — returns `None` and only *prints* on failure, doesn't raise), `ActionSpaceProcessor` (converts Godot's tuple action space into a flat Box/MultiDiscrete for SB3), `convert_macos_path`. |
| `pursuit_evasion_features.py` | `PursuitEvasionFeatures`: the closed-form/approximated geometry for Apollonius circles, Basic Engagement Zones (BEZ/WEZ), Dynamic Maneuvering Cue (DMC), Active Target Defense Differential Game (ATDDG), and Range-Limited Pursuit-Evasion (RLPE). This is where the thesis's Factor A/B/C math lives. The methods actually used are the lower-level ones (`compute_apollonius_intercept`, `compute_bez_boundary`, `compute_dmc`, `compute_atddg_defense_features`, `compute_barrier_hyperbola_value`, `compute_optimal_intercept_heading`, `compute_critical_escape_heading`, `compute_capture_probability_proxy`), called individually by the wrapper below. *(Update: `compute_features_for_agent()`/`get_feature_names()`/`extract_agent_state_from_obs()`, previously dead code, have been deleted — see §5.4. Also fixed: `compute_atddg_defense_features`'s 2D cross-product call, which crashed under NumPy 2.x — see §5.2b.)* |
| `enriched_observation_wrapper_ablation.py` | `EnrichedObservationWrapper`: appends the theoretical features to the base observation. `SplitNormalizationWrapper`: normalizes only the base (non-enriched) observation dims with its own Welford running-mean/var, leaving the enriched dims untouched, so that `VecNormalize` (with `norm_obs=False`) doesn't re-normalize them. *(Update: `FeatureOnlyWrapper`/`create_enriched_env`, previously dead code, have been deleted — see §5.4. `get_ablation_summary`/`print_ablation_matrix` were kept since the file's own `__main__` self-test still exercises them.)* |
| `contracts.py` **(new)** | Declares, as plain data, the contract between Godot and the Python side that was previously implicit: `EXPECTED_OBSERVATION_LABELS` (the 27 raw observation label names, in order) and `EXPECTED_INFO_KEYS` (the `info`-dict keys Godot's `SimManager.gd` actually emits, enumerated from that file directly). `assert_observation_labels()` is called at environment-construction time (`B_ACE_GodotPettingZooWrapper.__init__`) and raises immediately if Godot's actual reported labels don't match — turning "every positional-index feature computation silently reads the wrong field" into a loud failure at construction time instead. `check_info_keys()` is a soft, one-time-per-episode warning (not an assertion) for unrecognized `info` keys — building this surfaced a real bug, see §5.2b/§5.3. |
| `SimpleExample_B_ACE_config.json` | The scenario config `train_bace_clean.py` actually loads by default (`--config` default). 1 blue / 1 red / 1 HVAA, with randomized spawn blocks (`BlueSpawn`, `HVAASpawn`). **Its `RewardsConfig` numbers do not match the thesis** — see §4.2. |
| `Default_B_ACE_config.json` | The upstream/generic B-ACE example config (num_agents: 4 blue / 3 red, no HVAA at all). Not what `train_bace_clean.py` defaults to; useful mainly as a reference for the engine's built-in reward defaults (they match `Sim_assets.gd`'s hardcoded `RewardsControl` defaults almost exactly). |

**Ablation-category naming mismatch worth knowing up front:** the code's internal names for
the three observation-feature ablation categories are `GEOMETRY`, `ENGAGEMENT`, and
`RANGE_LIMITED` — these correspond 1:1 to the thesis's **Factor A (ATDDG)**, **Factor B
(WEZ/DMC)**, and **Factor C (RLPE)** respectively, just under different labels. `--ablation-config
geometry_only` means "Factor A only," not literally "geometry."

### 2.2 `rl_files/` — training, reward shaping, experiment orchestration, evaluation

**Core pipeline scripts:**

| File | Purpose |
|---|---|
| `env_factory.py` (871 lines) **(new)** | **The single file that defines the environment/wrapper stack.** Both `train_bace_clean.py` and `evaluate_experiment_seeds.py` call its one function, `build_env(bace_config, is_eval, ...)`, instead of each independently constructing their own copy the way they used to (see §2.3, §5.3 — this used to be a real, confirmed source of train/eval drift). Contains `SingleAgentBACEEnv` and the reward-shaping wrapper classes that used to be defined inline inside `train_bace_clean.py` (`HeadingRateLimitWrapper`, `KillRewardWrapper`, `ActionSmoothnessPenalty`, `FireCooldownWrapper`, `HVAASurvivalEveryNStepsBonus`, `NanGuardWrapper`, `linear_anneal`), plus `get_split_norm_wrapper_from_vec_env()`. The core reward-shaping portion of the stack is represented as an explicit ordered `(WrapperClass, kwargs)` list (`REWARD_SHAPING_STACK`) applied in a loop, rather than 150 lines of interleaved imperative construction. |
| `train_bace_clean.py` (1,515 lines, down from 2,627) | **The file you'll edit most for hyperparameters/CLI flags.** Parses ~55 CLI flags (including `--use-wandb`/`--wandb-project`/`--wandb-entity`, new — see §4.5), calls `env_factory.build_env()` to construct the environment, constructs a `SkippableTrainPPO` model (see below) with the tuned hyperparameters as CLI defaults, trains via `model.learn()`, and runs a small ad-hoc final evaluation. No longer defines the reward-shaping wrapper classes itself (moved to `env_factory.py`) or the dead `ExpertRolloutBuffer`/`ExpertMixPPO` alternative-PPO implementation (deleted — see §5.4). |
| `run_experiment.py` (523 lines) | A generic multi-seed launcher: loops over `--seeds`, invokes `train_bace_clean.py` once per seed via `subprocess`, forwarding any flags it doesn't itself recognize straight through (this pass-through is also what `run_from_config.py`'s multi-seed configs rely on — see §4.5). **It does not know about the 5 experimental factors or the 16-18 design configurations** — see §4.1, this is an important correction to how you might expect this file to work. |
| `run_from_config.py` (new) | Declarative YAML run launcher (`python rl_files/run_from_config.py configs/factor_BD.yaml`) — see §4.5. Converts YAML keys (which must equal a real `train_bace_clean.py` argparse `dest` name) into the equivalent CLI invocation and shells out to it via `subprocess`, so a YAML-launched run can never diverge from the equivalent hand-typed command. Delegates to `run_experiment.py` for multi-seed configs (a `seeds:` list in the YAML) rather than reimplementing the seed loop. |
| `evaluate_experiment_seeds.py` (1,969 lines, down from 2,173) | Loads trained checkpoints + `VecNormalize` stats per seed, runs deterministic evaluation episodes, and computes per-seed / cross-seed statistics. Implements far less of the thesis's Chapter IV statistical machinery than you'd expect — see §4.1. Its evaluation environment now calls `env_factory.build_env(..., is_eval=True)` — see §2.3/§5.3 for what changed. |

**Wrapper/callback files** (all imported and layered together inside `env_factory.py`'s
`build_env()`, called by both `train_bace_clean.py` and `evaluate_experiment_seeds.py`):

| File | Class(es) actually used | Maps to |
|---|---|---|
| `expert_alignment_wrapper.py` | `ExpertAlignmentWrapper` | Factor D (alignment-based reward shaping) |
| `expert_action_blending_wrapper.py` | `ExpertActionBlendingWrapper` | Factor E (action blending) |
| `aggressive_hunter_fixed.py` | `AggressiveHunterFixed` (the two sibling classes, `AggressiveHunterV3`/`AggressiveHunterTunable`, were confirmed-dead and have been deleted — see §5.4) | The scripted "expert" FSM that Factors D and E are built around |
| `sustained_turn_penalty.py` | `SustainedTurnPenaltySimple` | r_shape sustained-turn penalty |
| `track_aware_shaping.py` | `CombinedTrackAwareShaping` | r_shape track-loss penalty + range-closure bonus |
| `mission_tempo_shaping_perstep.py` | `MissionTempoShapingPerStep` | r_shape "red still alive" / mission-tempo terms |
| `reward_breakdown_logger.py` | `CompactRewardLogger` (`RewardBreakdownLogger`, previously imported-but-dead, has been deleted — see §5.4) | Diagnostics only, no effect on reward |
| `rollback_eval_callback.py` | `RollbackEvaluationCallbackTopK`, `SkippableTrainPPO` (new) | The checkpoint-rollback mechanism (thesis §4.2) |

`hvaa_destruction_penalty.py` — previously listed here as dead code (neither class was ever
instantiated) — **has been deleted entirely**, per §5.4.

`SkippableTrainPPO` (new, `rollback_eval_callback.py`) is a small `PPO` subclass
`train_bace_clean.py` now instantiates instead of vanilla `PPO`. It replaces what used to be a
`model.train` monkey-patch (`RollbackEvaluationCallbackTopK._init_callback` reassigning
`self.model.train` to a wrapped closure) with a normal method override
(`skip_next_n_train_calls(n)` sets an instance counter; `train()` checks it and either no-ops or
calls `super().train()`). This is the fix for the `model.save()` pickling crash documented in
§5.2b — see that section for why the monkey-patch broke saving in the first place.

### 2.3 The actual wrapper stack, in order (as built by `env_factory.build_env()`)

This is the order environment `.step()` calls actually propagate through — reward
contributions accumulate bottom-to-top relative to this list (innermost/first-applied at top).
**Both `train_bace_clean.py` and `evaluate_experiment_seeds.py` now build this exact same
stack** by calling the one shared `env_factory.build_env(bace_config, is_eval, ...)` function —
see §5.3 for what this changed relative to how evaluation used to work.

1. `SingleAgentBACEEnv` — collapses the multi-agent PettingZoo env to one Gym `Env` (see §2.4 for why this matters for 2v2)
2. `EnrichedObservationWrapper` + `SplitNormalizationWrapper` *(only if `--use-enriched-obs`; at eval time, `SplitNormalizationWrapper` is only added if the checkpoint being evaluated actually used it — see `apply_split_normalization` in `env_factory.py`, for compatibility with older/legacy checkpoints trained before split normalization existed)*
3. `HeadingRateLimitWrapper`
4. `KillRewardWrapper`
5. `ActionSmoothnessPenalty`
6. `MissionTempoShapingPerStep`
7. `SustainedTurnPenaltySimple`
8. `CombinedTrackAwareShaping`
9. `FireCooldownWrapper`
10. `HVAASurvivalEveryNStepsBonus`
11. `ExpertAlignmentWrapper` *(only if `--expert-alignment`, training env only — correctly skipped at eval, this was always intentional)*
12. `ExpertActionBlendingWrapper` *(only if `--expert-blending`, training env only — same)*
13. `CompactRewardLogger`
14. `NanGuardWrapper`
15. `Monitor` (SB3)
16. `DummyVecEnv` → `VecNormalize`

**Schedule-dependent wrappers at evaluation time:** `ActionSmoothnessPenalty` and
`FireCooldownWrapper` (items 5 and 9) anneal their coefficients based on a `global_step` counter
that only training's callback ever advances (via `env_method("set_global_step", ...)`).
Evaluation has no such callback, so `env_factory.build_env()` accepts a
`freeze_schedule_at_step` argument; `evaluate_experiment_seeds.py` passes
`env_factory.FULLY_ANNEALED_STEP` (a constant past both schedules' anneal windows) so evaluation
reflects the trained policy's realistic, fully-annealed final discipline (full smoothness
penalty, mature fire-cooldown discipline) rather than each schedule's un-annealed step-0 starting
values, which evaluation would otherwise never advance past.

### 2.4 `Godot_Air_Combat/` — the B-ACE simulation (Godot 4.5)

| File | Purpose |
|---|---|
| `B_ACE.tscn` / `B_ACE.gd` | Root scene. **`B_ACE.gd` is dead/orphaned code** (opens with the comment `#Should be OLD CODE`, not referenced by any scene). The root scene's actual behavior comes from its `B_ACE_sync` child node. |
| `addons/godot_rl_agents/B_ACE_sync.gd` (~1,300 lines) | **The real, live entry point.** Implements the TCP protocol (4-byte little-endian length prefix + UTF-8 JSON, both directions), handles `handshake`/`config`/`reset`/`action`/`close`/`call` messages, and owns spawn randomization. Confusingly, *this* file also opens with a stale `#SHOULD BE OLD CODE` comment despite being the actively-developed core of the project — don't be misled by it the way a first-time reader would be. |
| `SimManager.gd` / `SimManager.tscn` | One instance = one parallel simulation world (its own `SubViewport`/`World3D`, so parallel envs don't interfere). Spawns/owns all `Fighter` instances, computes win/loss conditions every physics tick, and is the single choke point for kill/shot bookkeeping (`inform_state()`). |
| `components/Fighter.gd` (1,661 lines) | Every blue, red, and HVAA unit is this same scene, differentiated by `team_id`/`behavior`/`is_hvaa`. Owns 3DOF kinematics, the radar-track/WEZ pipeline, the scripted-adversary FSM, action ingestion, missile launch, and observation building (`get_obs()`). |
| `components/Missile.gd` / `missile.tscn` | Missile flight model, guidance, and hit/kill resolution. **The most important file for a missile-evasion follow-on** — see §3.1. |
| `components/Detection.gd` | **Dead code.** Intended as a collision-based radar detector; its `Area3D` has no collision shape attached in the scene, so its signals never fire. The *actual* live detection system is the `Track` class in `assets/Sim_assets.gd`, recomputed by pure geometry every step — not collision events. |
| `assets/Calc.gd` | Stateless trig helpers (bearing, aspect angle, angle-off, heading wrap). Clean, no issues found. |
| `assets/Sim_assets.gd` | `SimGroups` (per-parallel-env node-group namespacing), `SConv` (unit conversion constants — 1 Godot unit = 100 real meters, etc.), `Track` (per-contact detection/WEZ state), `RewardsControl` (the actual Godot-side reward engine). |
| `assets/wez/*.json` | WEZ engagement-envelope models: large multivariate quartic polynomial regressions (fit against an offline 3DOF missile-kinematics dataset), evaluated at runtime via Godot's `Expression.execute()`. |
| `gui/`, `ViewPort.gd` | HUD text, pause/speed-up buttons, and a free-fly debug camera. Purely cosmetic/interactive; none of it runs during headless training. |
| `assets/Default_Sim_Config.json` | **Only** used as a standalone-editor fallback when Godot can't reach the Python server (confirmed by an explicit code comment: *"Don't load baked JSON - Python will send everything"*). Not authoritative for real training parameters — several of its values (`action_repeat: 20`, HVAA `init_hdg: 135°`) don't match what the thesis describes, because it's simply never loaded during real Python-driven training. |

---

## 3. How to Attack the Code for Follow-On Work

### 3.1 Missile Evasion

**What already exists that you can build on:**
- A shooter-side "guidance quality" state machine in `Missile.gd`: `upLink_support` /
  `lost_support()` / `recover_support()`, driven by whether the *shooter's* radar track on the
  target is currently live (`Fighter.process_tracks()`). Before the missile goes "pitbull"
  (permanent seeker lock, triggered at 10 NM regardless of uplink state), losing the shooter's
  track causes the missile to dead-reckon off stale position/velocity — this is the *only*
  mechanism by which a target's behavior currently affects missile accuracy.
- A time-varying guidance gain: `turn_speed` ramps from 0.1 up to ~5.1 (asymptotically, as
  flight time increases) via `current_velocity.lerp(direction_to_target, turn_speed*delta)`.
  This is **not** textbook proportional navigation despite the thesis's description — it's a
  pursuit-course homing law with increasing authority over time. A hard maneuver early in a
  missile's flight, while `turn_speed` is still small, is the theoretically most promising
  evasion window under this model.
- Radar cone gating (`radar_hfov`/`radar_vfov`, ±60°/-20°..+40°) on the shooter's track of the
  target, with a 10-second "coast" before a lost track is declared dead — a target that
  maneuvers to exit the shooter's FOV already starts the uplink-degradation clock described
  above.

**What does not exist and must be built from scratch:**
- **Any "a missile is currently inbound at me" observable.** Nothing in `get_obs()` reports
  this today — the only missile-related observation bits are the *shooter's own* in-flight-missile
  flag and whether *its* tracked contact currently has missile support, i.e., purely
  first-person information. The cheapest way to add this: scan
  `tree.get_nodes_in_group(MISSILE)` for missiles whose `target` is `self`, and expose a
  boolean (or time-to-impact estimate) through `get_obs()` and the `info` dict.
- **Any probabilistic Pk-at-intercept model.** Hit/kill resolution
  (`Missile.gd::_on_area_3d_body_entered`) is **100% deterministic geometric collision**: a
  `SphereShape3D` of radius 3.0 Godot units (300 real meters) — if the missile's guidance
  closes to within that radius before its 50-second flight timer expires, it is a guaranteed
  kill, with *zero* dependence on closing velocity, aspect, or range at intercept. The thesis's
  "probabilistic... based on range, aspect, closing velocity" language describes the *pre-shot*
  WEZ/launch-decision gating (whether the scripted adversary chooses to fire), not what happens
  once a missile is actually in flight. If you want a Pk formula at intercept, it has to be
  added inside this function.
- Any notch/beam/Doppler-aspect susceptibility model, or chaff/flare — none exist anywhere.
- **A team check on missile impact.** `_on_area_3d_body_entered` only checks `body != shooter`,
  never `team_id`. Fix this *before* any 2v2 work (see §3.2) — otherwise a friendly-fire hit is
  physically possible and would even reward the shooter as if it had killed an enemy
  (`own_kill()` is unconditional and `add_hit_enemy_rew()` fires on any non-shooter hit).

**Suggested order of attack:** (1) fix the missile team-check bug regardless of whether you do
2v2, since it silently corrupts any multi-unit scenario; (2) add a per-Fighter "missiles
currently targeting me" signal to the info dict and observation vector, following the exact
pattern already used for `blue_pk`/`red_pk`/`blue_killed_total` in `SimManager.gd`'s
`_get_info_from_agents()`; (3) decide whether you want evasion credit purely through the
existing uplink/pitbull geometry, or want to add an explicit Pk-vs-geometry formula at
intercept — the former is "free" (already modeled, just not observable/rewarded), the latter
requires new Godot code; (4) add a matching reward-shaping term (a new small `gym.Wrapper`
following the pattern of the other reward-shaping wrappers in `rl_files/env_factory.py`, e.g.
`KillRewardWrapper` — `hvaa_destruction_penalty.py`, previously a template for this, has since
been deleted as dead code, see §5.4) once the signal exists, and a matching evaluation metric
(following the exact pattern `evaluate_experiment_seeds.py` already uses for
`red_killed`/`blue_killed` via `EpisodeData`).

### 3.2 Extending to 2v2 Multi-Agent RL

**The encouraging news:** the simulation layer is already genuinely multi-agent, not a 1v1
sim you'd need to rebuild. `SimManager._set_agents()` loops over
`agentsConfig["blue_agents"]["num_agents"]` / `red_agents["num_agents"]` generically;
`get_obs()`'s track/allied-track blocks already scale with roster size; `B_ACE_sync.gd`
already keys `obs`/`reward`/`done`/`info` by agent name (`agent_0`, `agent_1`, ...) in a dict,
not flat per-index arrays; and `B_ACE_GodotPettingZooWrapper` is a real PettingZoo `ParallelEnv`
that already iterates `self.possible_agents`. **`Default_Sim_Config.json` even already
specifies `num_agents: 2` for both sides** (though that file is only loaded as an offline
editor fallback, not proof the RL pipeline has been exercised at 2v2).

**Where the single-agent assumption is actually baked in — this is what you have to change:**

1. **`SingleAgentBACEEnv`** (now defined once, in `rl_files/env_factory.py` — previously
   duplicated near-identically in both `train_bace_clean.py` and
   `evaluate_experiment_seeds.py`, since consolidated, see §2.2) is the real blocker, not the
   Godot side. Its `step()`
   broadcasts the *one* action from the single PPO policy identically to every agent in the
   underlying multi-agent env (`{agent: clipped for agent in self._ma_env.agents}`) — for two
   independently-controlled blue aircraft this is simply wrong, both would fly identical
   maneuvers. Its `reset()`/`step()` also discard every observation/reward/termination signal
   except the one `controlled_agent_id`'s. You need either (a) N instances of a
   single-agent-style wrapper, one per controlled agent, sharing one underlying
   `B_ACE_GodotPettingZooWrapper`, feeding a genuinely multi-agent training loop (shared policy
   / CTDE with a centralized critic, which is what the thesis's own literature review
   recommends in §2.1), or (b) a proper multi-agent training library (SB3-adjacent MARL
   extensions, RLlib, etc.) built around the existing PettingZoo interface directly.
2. **Every reward-shaping wrapper that assumes exactly one enemy track / one protected asset**
   (this is the majority of them): `track_aware_shaping.py`'s `track_idx`/`range_idx` scalar
   indices, `mission_tempo_shaping_perstep.py`'s single boolean `red_killed_event`/`hvaa_alive`
   gates. These need either a redesigned per-track schema with
   aggregation logic (e.g., "nearest threat" or "most dangerous threat"), or N wrapper
   instances with an explicit combination rule that doesn't exist today.
3. **The expert wrappers and scripted FSM** (`expert_alignment_wrapper.py`,
   `expert_action_blending_wrapper.py`, `aggressive_hunter_fixed.py`) need one instance per
   controlled agent (workable — they already parameterize on `controlled_agent_id`), but the
   FSM itself has no target-selection logic for multiple enemies (Godot already auto-picks a
   "highest priority target" via `offensive_factor` argmax and never exposes this as an
   RL-controllable choice — worth deciding whether a 2v2 agent should get to choose its own
   target or continue relying on this built-in heuristic).
4. **`rollback_eval_callback.py`'s evaluation loop** hardcodes a single-env `VecEnv`
   (`reward[0]`, `done[0]`, `while not done:` relying on scalar truthiness) — this would raise
   at runtime with `n_envs > 1`, so parallelizing evaluation throughput for a 2v2 setup
   (probably desirable given longer/more complex episodes) requires rewriting this loop.
5. **The missile team-check bug** (§3.1) becomes much more consequential here — with two
   co-located blue aircraft plus an HVAA sharing the engagement volume, the geometric odds of a
   stray friendly-fire kill rise sharply, and each one would currently be recorded as an enemy
   kill for the shooter.
6. Minor, non-blocking things that still need attention: `B_ACE_sync.gd`'s
   `_collect_experiment_result()` hardcodes `fighters[0]`/`fighters[1]` as "blue"/"red" (breaks
   with >1 blue, batch-experiment-mode debug output only); `_send_env_info()` derives the shared
   observation/action space shape from `agents[0]` only (fine as long as all blue units stay
   homogeneous).

**Suggested order of attack:** fix the missile team-check bug first (cheap, prevents silently
corrupted 2v2 results); then decide your MARL training paradigm (shared-policy CTDE is the
path of least resistance given the existing centralized-critic recommendation in the thesis's
own lit review) before touching `SingleAgentBACEEnv`; then redesign the reward-wrapper
single-track assumptions only for the specific shaping terms you actually want to keep (you
may find it simpler to drop some of the more heavily-miscalibrated shaping terms — see §4 —
rather than generalize all of them to N-agent semantics); update `evaluate_experiment_seeds.py`
and `rollback_eval_callback.py`'s evaluation loops last, once the training side works.

---

## 4. Environment Setup

**This section has been fully executed end-to-end on this machine (Windows, Godot 4.5.2,
Python 3.13) as part of writing this guide** — not just researched. `bin/B_ACE_v0.1.exe`
now exists, `.venv/` now exists with all dependencies installed, and a real training run
was confirmed to complete cleanly (env connects, full episodes run, model checkpoint +
`VecNormalize` stats + results JSON all save successfully). Three real, previously-undetected
bugs (found only by actually running the pipeline, not by reading the code) were fixed along
the way — see the end of this section and §5.2b. Nothing else in the codebase was changed.

There is **no `requirements.txt`, `pyproject.toml`, `setup.py`, or root `README`** anywhere in
this repository, and no automated tests or CI configuration, so none of this was discoverable
from an in-repo setup script.

### 4.1 Godot side — done

1. **Godot Engine 4.5.2** was used (matches `project.godot`'s declared
   `config/features=PackedStringArray("4.5")` exactly). You also have Godot 4.6.2 downloaded;
   I deliberately used 4.5.2 instead to avoid any risk of a minor-version behavior change
   silently affecting this thesis's carefully-tuned GDScript. Executable used:
   `C:\Users\kyle2\Downloads\Godot_v4.5.2-stable_win64.exe\Godot_v4.5.2-stable_win64.exe`.
2. Ran a headless import pass once (`godot --headless --path Godot_Air_Combat --import --quit`)
   to generate the local `.godot/` cache — equivalent to opening the project in the editor once.
3. **Downloaded and installed the official Godot 4.5.2 export templates** (~1.35 GB, from
   Godot's official GitHub releases) into
   `%APPDATA%\Godot\export_templates\4.5.2.stable\` — this wasn't present and is required for
   any export to work.
4. **Added a Windows Desktop export preset** to `Godot_Air_Combat/export_presets.cfg` (only
   `macOS` and `Linux` presets existed before — consistent with the thesis being built on a
   Mac). This is a project-config edit, not GDScript logic.
5. **Exported the release binary headlessly**
   (`godot --headless --path Godot_Air_Combat --export-release "Windows Desktop" bin/B_ACE_v0.1.exe`),
   producing `bin/B_ACE_v0.1.exe` (96 MB) + `bin/B_ACE_v0.1.pck`, at exactly the path
   `train_bace_clean.py`'s `_resolve_env_path()` expects on Windows. Verified as a valid PE32+
   executable.

**If you ever need to re-export** (e.g. after a GDScript change): re-run step 5's command from
the repo root, or use the editor (**Project → Export... → Windows Desktop → Export Project**)
now that the preset exists.

### 4.2 Python side — done

- **Python 3.13.14** (already on this machine) — well above the 3.10 floor
  `train_bace_clean.py` requires (it uses PEP 604 `int | None` syntax).
- Created a virtual environment at `<repo_root>/.venv/` and installed:
  ```
  pip install -r requirements.txt
  ```
  **Update: `requirements.txt` now exists at the repo root** (it didn't when this section was
  originally written — see the rest of this note for what it pins). It pins exactly the versions
  confirmed working below: `gymnasium 1.3.0`, `pettingzoo 1.27.0`, `numpy 2.5.2`, `scipy 1.18.0`,
  `torch 2.13.0+cpu`, `stable-baselines3 2.9.0`, `tensorboard 2.21.0`, plus `tqdm`+`rich` (needed
  for SB3's `progress_bar=True`, which `train_bace_clean.py` always passes to `model.learn()` —
  not in the base `stable-baselines3` install, only in its `[extra]` extra), `pytest` (for
  `tests/`, see §4.6), and `pyyaml` (for `run_from_config.py`, see §4.5). All import cleanly; no
  version-compatibility problems encountered. **Did not** install the `godot-rl-agents` PyPI
  package — this repo reimplements its own minimal socket client (`b_ace_py/godot_env.py`)
  instead of depending on that library.
- To activate this environment yourself: `source .venv/Scripts/activate` (Git Bash) or
  `.venv\Scripts\Activate.ps1` (PowerShell).
- Optional, not in `requirements.txt` by default (commented out there, install manually if
  wanted): `wandb` (opt-in training-dashboard mirror, see §4.5), `streamlit`+`matplotlib`+
  `pandas` (the local results dashboard, `tools/dashboard.py`, see §4.5).
- Still missing, confirmed absent repo-wide: a hyperparameter-search (Optuna) script. If you
  need one, you're writing it yourself around the existing training/eval scripts. (Plotting/
  analysis code — previously also missing — now exists: `tools/dashboard.py`, see §4.5.)

### 4.3 Running it — verified commands

**One required environment variable on Windows**, confirmed necessary by an actual crash (see
§5.2b): several files in this codebase `print()` emoji characters, which raises
`UnicodeEncodeError` under Windows' default (non-UTF-8) console codepage and kills the process.
Set this once per shell session before running anything:

```powershell
# PowerShell
$env:PYTHONUTF8 = "1"
```
```bash
# Git Bash
export PYTHONUTF8=1
```

Then, from the repo root, with `.venv` activated:

```bash
# Single training run (uses SimpleExample_B_ACE_config.json by default):
python rl_files/train_bace_clean.py --total-timesteps 10000000 --experiment-name my_test

# Enriched observations, Factor B only (WEZ/DMC), with alignment-based reward shaping:
python rl_files/train_bace_clean.py --total-timesteps 10000000 \
    --use-enriched-obs --ablation-config engagement_only --expert-alignment \
    --experiment-name factorB_D

# Multi-seed sweep of one configuration:
python rl_files/run_experiment.py --experiment-name factorB_D --seeds 42 123 456 789 \
    --total-timesteps 10000000 --use-enriched-obs --ablation-config engagement_only --expert-alignment

# Evaluate a completed run:
python rl_files/evaluate_experiment_seeds.py --experiment-dir runs_sb3/factorB_D --episodes 200
```

**Confirmed working end-to-end**: a 512-timestep smoke run (`--total-timesteps 512 --n-steps 256
--batch-size 64 --epochs 1`) completed with zero errors after the fixes below — Godot launched,
the socket connected, the config was accepted, 27-dim observations and their labels came back
correctly, multiple full episodes ran (terminating on `All_Blue_Destroyed` as expected), PPO
trained, a 20-episode hand-rolled final evaluation ran, and the model checkpoint (`.zip`),
`VecNormalize` stats (`.pkl`), and `training_results_seed<N>.json` all saved to
`runs_sb3/<run_name>/`. A real ~10M-timestep run should behave the same way, just much longer.

Other gotchas confirmed during setup, not documented anywhere in-code:
- `run_experiment.py` has **no default for `--total-timesteps`** of its own — it silently
  inherits `train_bace_clean.py`'s default of 1,000,000, which is 10x too small to reproduce
  anything resembling the thesis's results. Always pass `--total-timesteps` explicitly.
- "Baseline" (no heuristic features at all) means **omitting `--use-enriched-obs` entirely**,
  not passing `--ablation-config none` — the latter still runs the enrichment/split-normalization
  wrapper machinery with zero added features, which is a different code path.
- Runs land under `<repo_root>/runs_sb3/<experiment_name>/<run_name>/`, which is gitignored.
  Reward/observation-mode metadata is not saved anywhere that `evaluate_experiment_seeds.py`
  reads back automatically — if you evaluate a model trained with a non-default
  `--ablation-config`, pass the *matching* `--ablation-config` to the eval script yourself, or
  it will silently reconstruct the wrong features for that checkpoint (see §5.3).

### 4.4 Code bugs found (only by actually running the pipeline) and fixed

Everything else in this guide came from reading the code. These were invisible to static
reading and only surfaced by executing the real training script — each one *fully blocked*
some part of the pipeline before the fix. Full technical detail in §5.2b.

1. **`b_ace_py/B_ACE_GodotPettingZooWrapper.py`** — `__init__` crashed with
   `KeyError: 'action_spaces'` on every single invocation, before an environment could even be
   constructed. A redundant, wrongly-keyed line was reassigning `self.action_spaces` from a
   dict key that doesn't exist; `self.action_spaces` was already correctly populated as a side
   effect of the `_get_env_info()` call one line above it. Fix: deleted the redundant line.
2. **`rl_files/rollback_eval_callback.py`** — `model.save()` crashed with
   `TypeError: cannot pickle '_thread.lock' object` at the very end of every training run
   (meaning, in a real full-length run, **no checkpoint could ever be written**, including the
   periodic best-model checkpoints this same callback is responsible for saving *during*
   training). Root cause: the callback monkey-patches `model.train` during training and its
   restore logic reassigned the bound method instead of removing the instance-level override,
   leaving a `'train'` key in the model's `__dict__` whose value's `__self__` bypasses SB3's
   save-exclusion filtering entirely, dragging in the live environment (and its socket
   connection) at pickle time. Original fix: `_on_training_end` now `del`s the instance
   attribute instead of reassigning it. **Superseded fix:** the monkey-patch itself has since
   been replaced entirely by `SkippableTrainPPO`, a small `PPO` subclass (see §2.2) — the
   post-rollback "skip N training updates" logic is now a normal method override, so
   `model.__dict__` can no longer gain a stray `'train'` key at all, structurally, rather than
   relying on cleanup code remembering to remove it.
3. **`rl_files/train_bace_clean.py`** — the final summary print crashed with
   `ValueError: -inf is not in list` whenever a run ends before its first periodic evaluation
   (any run shorter than `eval_freq`, default 200,000 steps — exactly what any short test run
   looks like). This happened *after* the model was already saved, but *before* the final
   `training_results_seed<N>.json` was written, so that file was silently never produced. Fix:
   guarded the summary print with a check for whether any evaluation actually ran.

A fourth bug of this same "only visible by running it" kind was found later, while building the
observation-label contract check described in §4.6:
`b_ace_py/pursuit_evasion_features.py::compute_atddg_defense_features` called `np.cross()` on
two 2D vectors, which NumPy 2.0+ no longer accepts (this used to implicitly return the
2D cross-product's scalar z-component; newer NumPy requires exactly 3D vectors). This crashed
`--use-enriched-obs` training/eval the first time this exact code path executed. Fix: replaced
with the equivalent explicit scalar formula (`a[0]*b[1] - a[1]*b[0]`) — same math, no `np.cross`
dependency.

### 4.5 New tooling (added since the sections above were written)

Two follow-on improvement plans (`IMPROVEMENT_PLAN_1_VISUALIZATION_AND_RUNNING.md`,
`IMPROVEMENT_PLAN_2_WRAPPERS_AND_DEPENDENCIES.md`, both at the repo root) added the following.
None of it changes training/reward/observation behavior — see §5.3 for the one deliberate,
approved exception (evaluation's wrapper stack now matches training's, where it used to
silently differ).

- **`--use-wandb` / `--wandb-project` / `--wandb-entity`** (opt-in, default off) on
  `train_bace_clean.py` (and forwarded through `run_experiment.py`) mirror the same TensorBoard
  scalars to a Weights & Biases run via `sync_tensorboard=True`. `wandb` is only imported if the
  flag is passed; training without it is unaffected.
- **`tools/dashboard.py`** — a local Streamlit results-browsing dashboard
  (`streamlit run tools/dashboard.py`) that reads only existing output files (`monitor.csv`,
  `training_results_seed<N>.json`) — reward-vs-timestep curves, evaluation-summary with a
  confidence band, and a hyperparameter-config table, for one or more selected runs. No cloud
  account needed; complements, doesn't replace, TensorBoard/W&B.
- **`configs/*.yaml` + `rl_files/run_from_config.py`** — a declarative way to save and re-launch
  a specific run configuration (`python rl_files/run_from_config.py configs/factor_BD.yaml`).
  YAML keys must equal a real `train_bace_clean.py` flag's argparse `dest` name; an unrecognized
  key errors immediately. Five example configs ship in `configs/`, matching the factor table in
  `HOW_TO_RUN.md`. `--dry-run` prints the constructed command without running it.
- **`tools/equivalence_check.py`** — the regression tool built to verify all of the above (and
  the wrapper-stack consolidation in §2.2/§2.3) didn't change training behavior. Captures the
  wrapper-stack's structure (class names, order, config) for a given CLI-flag combination and
  diffs it against a saved baseline (`baselines/*.json`) — structural mismatches are a hard
  failure; trajectory-value differences are informational-only, since Godot's spawn-position
  randomization is not reproducible run-to-run even with a fixed seed (confirmed: the `--seed`
  CLI flag never actually reaches Godot's `EnvConfig.seed`).

### 4.6 Tests

`tests/test_env_factory_regression.py` (run with `pytest`, or `pytest -m "not slow"` to skip the
two tests that launch a real Godot instance) — a minimal regression suite added specifically
because two of the four bugs in §4.4 would have been caught by an automated test if one had
existed. Confirmed by temporarily reintroducing the original `KeyError: 'action_spaces'` bug and
verifying the test catches it (then reverting). Also verifies `RollbackEvaluationCallbackTopK`
rejects a non-`SkippableTrainPPO` model at `_init_callback` time (fail-fast, no Godot needed).

---

## 5. Glaring Issues (Full Audit)

Organized by severity/category. File:line citations are given wherever the underlying agent
audit captured them precisely; a few are cited by function/file only where GDScript line
numbers weren't captured.

### 5.1 The single most important finding: reward-constant mismatches, everywhere

Nearly every numeric constant in the reward-shaping pipeline exists at **three or four layers
that disagree with each other**: the thesis text, a wrapper class's own constructor default,
the actual value passed at its instantiation call site in `train_bace_clean.py`, and sometimes
a CLI `argparse` default one layer further back. This isn't one bug, it's a pattern across
nearly the whole reward system — treat every specific number in the thesis's Chapter III
reward equations as **unconfirmed against this exact repo snapshot** until you've checked which
value was actually used for a specific run.

| Term | Thesis | Class/file default | Actual call site in `train_bace_clean.py` |
|---|---|---|---|
| Mission success reward | +10 | — | `SimpleExample_B_ACE_config.json`: `mission_accomplished_factor=8.0` |
| Red destroyed reward | +8 | `reward_breakdown_logger.py`'s `kill_bonus=12.0` (self-consistent with below) | Godot `hit_enemy_factor=10.0` **+** Python `KillRewardWrapper(kill_reward=12.0)` ≈ **22 combined**, and the 8/10 values for mission-success vs. red-kill look transposed relative to the thesis |
| CCA (blue) destroyed | -3 | — | `SimpleExample_B_ACE_config.json`: `hit_own_factor=0.0` (disabled) |
| HVAA destroyed | -15 | `hvaa_destruction_penalty.py` default -10.0 | **Never applied** — see §5.2 below, this is its own dedicated finding |
| Missile fired | -0.1 | — | `SimpleExample_B_ACE_config.json`: `missile_fire_factor=0.0` (disabled) |
| Missile miss | -0.2 | — | `SimpleExample_B_ACE_config.json`: `missile_miss_factor=0.0` (disabled) |
| Track held / lost | +0.001 / -0.01 | — | Config matches (`keep_track_factor=0.001`, `detect_loss_factor=-0.01`) — **one of the few exact matches found** |
| HVAA alive bonus, "+0.005/25 steps" | +0.005 every 25 steps | `HVAASurvivalEveryNStepsBonus` default `every_n_steps=50, bonus=0.05` | Actual: `every_n_steps=25` (matches), `bonus=0.01` (2x thesis) — **and** a *separate* `MissionTempoShapingPerStep` bonus (`bonus_per_step=0.006`) layered on top, which the thesis's equation doesn't mention as a distinct term |
| Range-closure bonus | +0.005/step | `AsymmetricRangeShaping` default `k_close=20.0` | `CombinedTrackAwareShaping` actual: `k_close=40.0`, can reach **~0.8/step at the clip ceiling — roughly 160x the thesis's stated value** |
| Sustained-turn penalty | -0.01/step | matches | matches (`penalty_coef=0.01`), though the functional form is a windowed-mean-squared-deviation, not a flat per-step deduction |
| "Red still alive" penalty | -0.0005/step | `MissionTempoShapingPerStep` class default `-0.005` (10x) | Actual call site: `-0.003` (**6x thesis**) |
| Alignment coefficient (Factor D) | c(t)=0.1 | `ExpertAlignmentWrapper` class default 0.1 (matches) | CLI default `--alignment-coef 0.002` — **50x smaller** |
| Blending initial/final alpha (Factor E) | 0.7 → 0.0 by 5,000,000 steps | `create_alpha_decay_fn` signature defaults: 0.7 → 0.1 by 4,000,000 | CLI defaults actually wired up: **0.8 → 0.05 by 5,000,000** — three different versions of the same schedule across three places in the code |
| Fire cooldown curriculum | 20 → 150 timesteps over 4,000,000 steps | — | Actual (hardcoded, no CLI flag): `cd_start=70, cd_end=120` over **5,000,000** steps |
| Heading-rate-limit curriculum | ±0.15/step, "decays to 0 over 6,000,000 steps" | — | `HeadingRateLimitWrapper` has **no annealing logic at all** — `max_delta=0.15` is fixed for the entire run, confirmed by reading the full class body (no `set_global_step` method exists) |
| Rollback: eligibility threshold | eval reward > 15 | — | `activation_threshold=12.0` |
| Rollback: patience | "7 consecutive" degrading evals | `rollback_patience` class default 5 | Actual: `rollback_patience=6` (fires one cycle earlier than described) |
| Expert FSM fire envelope | range∈[0.15,0.40], aspect<120° | `AggressiveHunterFixed` actual: `fire_range_min=0.08` (thesis says 0.15), `fire_range_max=0.40` (matches), `fire_cone=0.55` ≈ 99° half-width (narrower than "<120°") | |
| Expert FSM pursuit memory | 500 timesteps | Hardcoded `pursuit_memory_steps=400` | 400 ≠ 500 |
| PPO entropy coefficient | single value 0.0451 (Optuna Trial 0051) | — | Code uses a **decay schedule** `--initial-entropy-coef 0.051 → --final-entropy-coef 0.005`; neither endpoint is 0.0451 |

Two config files exist for the Godot-side reward weights (`SimpleExample_B_ACE_config.json`,
which is what's actually loaded by default, and `Default_B_ACE_config.json`, a 4-blue/3-red
scenario with no HVAA at all, closer to the engine's own hardcoded `RewardsControl` defaults
in `Sim_assets.gd`) — **neither matches the thesis's stated r_env values**, and it is not
possible to determine from this repository alone which config (if either, as-is) was used to
produce the thesis's reported numbers. If exact reproduction matters, this is the first thing
to nail down, ideally against whatever launch logs or scripts exist outside this checkout.

### 5.2 Confirmed functional bugs (not just constant mismatches)

- **The HVAA can be destroyed without the episode ending or any terminal penalty applying, as
  long as the blue CCA is still alive.** `Sim_assets.gd`'s `RewardsControl.add_final_episode_reward()`
  has a fully-implemented `"HVAA_Destroyed"` branch (applies `hvaa_loss_factor`), but **nothing
  in `SimManager.gd` ever calls it** — the only compound termination condition that checks HVAA
  status (`All_Blue_Destroyed`) *also* requires the blue fighter itself to be dead. If the HVAA
  alone is shot down, the episode simply continues (reporting `hvaa_alive=false` every step)
  until an unrelated condition — most likely `Max_Cycles` — eventually ends it, at which point
  `mission_outcome` is retroactively computed as `"mission_failure"`, but the explicit -10/-15
  HVAA-loss terminal reward never fires through this path. This is compounded by two further
  facts: `SimpleExample_B_ACE_config.json` sets `hvaa_loss_factor=0.0` anyway (so even a fixed
  termination path would currently apply zero penalty), and the Python-side
  `hvaa_destruction_penalty.py` wrapper that could compensate for this is imported but **never
  instantiated** in `train_bace_clean.py`. Three independent, compounding reasons why "losing
  the HVAA while the CCA survives" may not be penalized the way the thesis's reward equation
  describes — worth verifying directly against real training logs before trusting this
  mechanism. (The agent does still lose the terminal mission-success bonus in this scenario,
  since the HVAA never reaches its distance goal — the effect isn't literally zero, just not
  the explicit penalty term the thesis describes.)
- **Missile hit resolution has no team/faction check.** `Missile.gd::_on_area_3d_body_entered`
  only checks `body != shooter`, never `team_id`/`team_color`. Friendly fire (blue missile
  hitting the HVAA, or in 2v2, a wingman) is physically possible, and — because `own_kill()` is
  unconditional and the shooter's `add_hit_enemy_rew()` fires on any non-shooter hit — a
  friendly-fire kill would currently **reward the shooter as if it had killed an enemy**. Low
  risk in strict 1v1v1 given the geometry, much higher risk once you have multiple co-located
  friendly units (see §3.2).
- **Missile launch, for the RL-controlled agent specifically, enforces almost none of its own
  computed safety gates.** `Fighter.gd::launch_missile_at_target` computes seven named boolean
  checks (inventory, track validity, range, IFF/team, cooldown, aspect, closure) but only uses
  them for debug printing. Under an explicit `# --- OVERRIDE LOGIC (TEMP) ---` comment, the
  actual enforced gate is just "ammo > 0 and target currently detected" — no cooldown, no range
  gate, no aspect gate. (The scripted adversary FSM self-disciplines via its own state machine
  before ever calling this function, so it behaves reasonably; the RL agent's raw `fire=1`
  action does not.) The WEZ ranges surfaced in the observation vector are informational for the
  RL agent, not physically enforced by the launch gate itself.
- **Reward terms computed at 20 Hz physics-tick cadence bleed into a pipeline meant to
  represent per-action-step reward.** `Fighter.gd::_calculate_hvaa_proximity_reward` (and the
  fire-eligibility check inside the same function) execute every physics tick, not gated by
  `action_repeat` the way the correctly-implemented `process_tracks()`/reward terms are. Since
  the Python side only reads accumulated reward once per env step (every `action_repeat` ticks),
  this term's effective weight is silently coupled to whatever `action_repeat` happens to be
  configured, and a fire command that becomes eligible partway through one action step can
  accumulate a "no fire" penalty multiple times for what the agent experiences as one decision.
- **`Detection.gd` is entirely inert.** Its `Area3D` node has no `CollisionShape3D` attached in
  `Fighter.tscn`, so its `body_entered`/`area_entered` signals never fire — and even if a shape
  were added, its logic conflates array-style and dictionary-style access on
  `radar_track_list` in a way that would throw a GDScript runtime error the first time it *did*
  fire. The system that actually drives detection (`Track.is_track_detectable()` in
  `Sim_assets.gd`) is a separate, unrelated, purely-geometric implementation. Don't build
  missile-evasion detection logic on top of `Detection.gd` without a rewrite; consider deleting
  it to avoid misleading future readers.
- **`get_hvaa_indices()` in `B_ACE_GodotPettingZooWrapper.py` has ~15 lines of unreachable code
  after its `return` statement** (setting `self.agent_idx`, `self.observation_space`,
  `self._cumulative_rewards`, `self.rewards`, `self.terminations`, etc.) — almost certainly
  leftover `__init__` code accidentally left inside this method during a refactor. It never
  executes. Confirmed via grep that `self.agent_idx` and `self._cumulative_rewards` are
  referenced nowhere else in the codebase (truly dead), while `self.observation_space` as a
  singular attribute is separately (and correctly) set by `SingleAgentBACEEnv`, so this doesn't
  currently break the live pipeline — but it means the singular `.observation_space` attribute
  would raise `AttributeError` if anyone ever instantiated `B_ACE_GodotPettingZooWrapper`
  directly and read that attribute before wrapping it.

### 5.2b Runtime-only bugs found by actually executing the pipeline (now fixed in this checkout)

Everything else in this document came from reading code. These were invisible to static
reading — they only surfaced by actually running `train_bace_clean.py` end-to-end on a real
Godot connection — and each one **fully blocked** part of the pipeline. All are now fixed in
this checkout; described here for completeness and in case you're diffing against an upstream
copy of this repo.

1. **`b_ace_py/B_ACE_GodotPettingZooWrapper.py::__init__` — `KeyError: 'action_spaces'`, on
   every construction, no exceptions.** The parent class's `_get_env_info()`
   (`b_ace_py/godot_env.py`) already populates `self.action_spaces` (plural, a list of
   `gymnasium.spaces.Dict`) as a side effect while it runs, and returns the *raw* dict from
   Godot, whose key is `"action_space"` (**singular** — confirmed directly from Godot's actual
   wire response: `action space {'input': {'action_type': 'continuous', 'size': 4}}`).
   `B_ACE_GodotPettingZooWrapper.__init__` then had a redundant, wrongly-keyed line,
   `self.action_spaces = env_info["action_spaces"]` (plural key that was never set on that
   dict), immediately clobbering the correctly-populated attribute with a `KeyError`. **Fix:**
   deleted that one line; `self.action_spaces` was already correct from the inherited call.
2. **`rl_files/rollback_eval_callback.py::RollbackEvaluationCallbackTopK` —
   `TypeError: cannot pickle '_thread.lock' object` inside every `model.save()` call, including
   the periodic best-checkpoint saves this same callback performs *during* training — meaning a
   real full-length run could complete all its training and still end up with **zero usable
   saved checkpoints**.** `_init_callback()` monkey-patches `self.model.train = wrapped_train`,
   a closure that captures the callback instance (and, transitively, the eval `VecEnv`, which
   wraps a live Godot TCP socket). `_on_training_end()` was supposed to undo this
   (`self.model.train = self._original_train_fn`), but *reassigning* the attribute still leaves
   a `'train'` key in `self.model.__dict__` whose value is a bound method — and pickling a bound
   method requires pickling its `__self__`, i.e. the entire live model object, which completely
   bypasses Stable-Baselines3's `_excluded_save_params()` filtering (the mechanism that normally
   keeps unpicklable things like `env` out of a saved checkpoint) and drags in the socket
   connection several layers down. Confirmed root cause with a monkeypatched diagnostic script
   that intercepted SB3's `data_to_json` and probed every model attribute individually — the
   offending key was unambiguously `'train'`. **Fix:** `_on_training_end()` now does
   `del self.model.__dict__["train"]` instead of reassigning, so the instance-level override is
   fully removed and attribute access falls back to the normal class method.
3. **`rl_files/train_bace_clean.py::main()` — `ValueError: -inf is not in list` in the final
   summary print, whenever a run ends before its first periodic evaluation** (any run shorter
   than `eval_freq`, 200,000 steps by default — which is exactly what any short test/smoke run
   looks like, but *not* a real ~10M-step run, where dozens of evaluations occur). This crash
   happens after the model checkpoint is already saved, but *before*
   `training_results_seed<N>.json` is written, so that file silently never gets produced.
   **Fix:** guarded the "Found at: N steps" print with a check for whether
   `eval_summary["mean_rewards"]` is non-empty and actually contains `best_mean_reward`
   (it stays at its `-inf` sentinel, and the list stays empty, if no periodic evaluation ran).
4. **`b_ace_py/pursuit_evasion_features.py::compute_atddg_defense_features` — `ValueError: Both
   input arrays must be (arrays of) 3-dimensional vectors`, the first time this code path
   executed under `--use-enriched-obs`.** `np.cross(AT_unit, AD_vec)` was called on two 2D
   vectors to get the perpendicular-distance scalar (the 2D cross product's implicit
   z-component) — NumPy 2.0 removed `np.cross`'s support for non-3D inputs. Found while building
   the observation-label contract check (§4.6) and running a real `--use-enriched-obs` training
   smoke test. **Fix:** replaced with the equivalent explicit formula,
   `AT_unit[0]*AD_vec[1] - AT_unit[1]*AD_vec[0]` — identical math, no `np.cross` dependency.

None of this changes anything about the reward-shaping/constant-mismatch findings elsewhere in
this document (§5.1, §5.3-§5.6) — those are design/tuning issues, these were pure plumbing bugs
that had nothing to do with reward values.

### 5.3 Train/evaluation mismatches (results from training may not reflect what gets reported)

**Update: the two bullets below describing wrapper-stack drift are now fixed** —
`evaluate_experiment_seeds.py` no longer reimplements any wrapper from scratch; it calls the
same `env_factory.build_env()` function training uses (see §2.2/§2.3). Kept here, marked, for
the historical record and because the specific numeric drift they document is exactly the kind
of silent bug this repo is prone to without a shared wrapper-stack definition.

- ~~`evaluate_experiment_seeds.py` reimplements `HVAASurvivalEveryNStepsBonus` and
  `FireCooldownWrapper` from scratch rather than importing the training versions, with
  **different parameters**: the survival bonus uses `every_n_steps=50, bonus=0.05` at eval vs.
  `every_n_steps=25, bonus=0.01` at training (≈2.5x the average per-step rate at eval); the fire
  cooldown is a fixed 120-step cooldown at eval vs. training's annealed 70→120-step schedule
  (matters most when evaluating early checkpoints, since it enforces end-of-training discipline
  on a policy that was actually trained under a looser cooldown at that point). Net effect:
  **the "mean_reward"/AUC reported by the evaluation script is not pure `r_env`** — it still
  includes at least one shaping term, at a different rate than training used.~~ **FIXED:**
  evaluation now uses training's exact `FireCooldownWrapper`/`HVAASurvivalEveryNStepsBonus`
  parameters (via the shared `build_env()`), annealed to their fully-trained end values (see the
  "Schedule-dependent wrappers at evaluation time" note in §2.3) rather than a separate,
  differently-tuned fixed configuration. **This is an intentional behavior change, made with
  the repo owner's explicit approval**: evaluation numbers for checkpoints re-evaluated after
  this fix will differ from numbers recorded before it (evaluation's reward now includes several
  shaping terms it previously omitted entirely — see the next bullet).
- ~~`HeadingRateLimitWrapper` (a pure action-space transform, not reward shaping) is absent
  entirely at evaluation — raw policy actions reach the simulator unfiltered, even though the
  policy was trained assuming its outputs get rate-limited first.~~ **FIXED** — evaluation's
  stack now includes `HeadingRateLimitWrapper` and every other reward-shaping wrapper training
  uses (`KillRewardWrapper`, `ActionSmoothnessPenalty`, `MissionTempoShapingPerStep`,
  `SustainedTurnPenaltySimple`, `CombinedTrackAwareShaping`), which were all previously entirely
  absent from evaluation. `ExpertAlignmentWrapper`/`ExpertActionBlendingWrapper` remain
  correctly training-only (that was always intentional, not drift).
- The observation-enrichment `ablation_config` used for a given checkpoint is not saved
  anywhere `evaluate_experiment_seeds.py` reads automatically; it guesses "which observation
  mode" a checkpoint used purely from total dimensionality via hardcoded thresholds
  (`obs_dim <= 30` → baseline, `<= 50` → enriched). If you evaluate a `geometry_only`-trained
  model without passing a matching `--ablation-config`, the eval script will silently
  reconstruct **all** enriched features instead — same total dimension, semantically wrong
  content, and nothing will error.

### 5.4 Dead code (large, could waste significant time if you try to trace it as "live")

**Update: every Python item below has been deleted from the codebase** (re-confirmed dead via a
fresh repo-wide grep immediately before each deletion, and the training/eval pipeline was
re-verified byte-identical in behavior afterward via `tools/equivalence_check.py` — see §4.5).
Kept here as a historical record, in case you're diffing against an upstream copy of this repo
or wondering why a symbol this document once mentioned no longer exists:

- ~~`ExpertRolloutBuffer`/`ExpertMixPPO` in `train_bace_clean.py` (~370 lines) — a full
  alternative PPO implementation with an imitation/behavioral-cloning loss and rollout-level
  expert-action mixing. Never instantiated; `main()` builds vanilla `PPO`.~~ **Deleted.** This
  was also the only "manual/alternative PPO implementation" in the repo, so its removal directly
  answers "can we decouple from manual PPO" — there was nothing live to decouple from.
- ~~`hvaa_destruction_penalty.py` — both classes (`HVAADestructionPenalty`,
  `HVAADestructionPenaltyWithDecay`) are imported but never instantiated.~~ **Deleted** (whole
  file removed).
- ~~`RewardBreakdownLogger` (the verbose diagnostic logger, as opposed to `CompactRewardLogger`
  which *is* used) — imported, never instantiated.~~ **Deleted**; `reward_breakdown_logger.py`
  now contains only `CompactRewardLogger`.
- Two of the three `aggressive_hunter_fixed.py` classes (`AggressiveHunterV3`,
  `AggressiveHunterTunable`) — ~95% copy-pasted from the one that's actually used. **Deleted**;
  only `AggressiveHunterFixed` (and `get_default_obs_indices()`) remain. The module docstring,
  which described constructor parameters (`fire_threshold`, `fire_cooldown`, `g_force`,
  `turn_rate`) that don't actually exist on the real class (see §5.7), was also corrected as
  part of this cleanup.
- `pursuit_evasion_features.py`'s `compute_features_for_agent`/`get_feature_names`/
  `extract_agent_state_from_obs` — **deleted** (only exercised by the file's own `__main__`
  self-test, confirmed unused elsewhere). `enriched_observation_wrapper_ablation.py`'s
  `FeatureOnlyWrapper`/`create_enriched_env` — **deleted** for the same reason.
  `get_ablation_summary`/`print_ablation_matrix` in the same file were **kept** (not dead by the
  same standard): the file's own `__main__` self-test still calls them.
- ~~`RangeClosingShaping` (defined inline in `train_bace_clean.py`) — defined, never
  instantiated.~~ **Deleted.**
- ~~`DEBUG_HVAA_OBS` module-level flag in `train_bace_clean.py` — defined, never read.~~
  **Deleted.**

**Not touched (out of scope for the Python-tooling refactor this update describes, still
accurate as originally documented):**
- `Godot_Air_Combat/B_ACE.gd` — orphaned root-scene GDScript, not referenced by any `.tscn`.
- A stray docstring reference to a module `simple_aggressive_expert.py`
  (`expert_alignment_wrapper.py`) that doesn't exist anywhere in the repo — a renamed-module
  reference that was never updated (the live module is `aggressive_hunter_fixed.py`).

### 5.5 Approximated theory vs. the thesis's "closed-form" framing

Worth knowing if you plan to extend or verify the observation-feature math. Cross-checking
`pursuit_evasion_features.py` against the formulas it cites:

- **Exact closed-form matches** (Factor B / WEZ-DMC): `compute_bez_boundary` implements the WEZ
  boundary formula and `compute_dmc` implements the DMC critical-aspect-angle formula exactly
  as derived in the cited papers — no approximation. `compute_critical_escape_heading` (Factor
  C / RLPE) likewise implements its cited closed-form equation directly (though if you need to
  verify the exact algebraic form against the source paper, do so independently — some of the
  thesis PDF's own equation transcription showed OCR artifacts that make hand-checking the
  paraphrased equation risky).
- **Acknowledged approximations** (Factor A / ATDDG): `compute_atddg_defense_features`,
  `compute_barrier_hyperbola_value`, and `compute_optimal_intercept_heading` all contain
  comments that explicitly say things like *"This is a geometric approximation of the full
  barrier surface"* and *"The quartic equation from Eq. 49 gives closed-form optimal defender
  heading. This implementation uses a geometric approximation..."* — i.e., the code openly
  documents that it substitutes heuristic distance/speed-ratio proxies (magic numbers like
  `attacker_speed * 1.2`, `d_DA < d_AT * 0.5`) for the actual closed-form differential-game
  solution the thesis cites. This is a plausible, if speculative, explanation for why the
  thesis's own results rank Factor A below Factor B in effectiveness (Table 20) — the
  most rigorously-implemented feature category also produced the strongest results.
- A minor semantic note: the "time to capture" Apollonius feature is computed as
  `intercept_distance / self.default_pursuer_speed`, where `default_pursuer_speed` is a fixed
  constructor default (1.0 in the actual pipeline, since no `feature_config` is passed) rather
  than the agent's real current speed — so numerically this feature is a distance proxy divided
  by 1.0, not a true elapsed-time estimate. Likely harmless for RL purposes (still a monotonic,
  useful signal) but worth knowing if you're trying to interpret its literal units.

### 5.6 Statistical/experimental-design gaps relative to Chapter IV

- **`run_experiment.py` does not enumerate the 16-18 factorial design configurations anywhere
  in this repo.** It is a generic seed-sweep launcher; each design (e.g. "ACD", "Baseline") would
  have required a separate, manually-constructed command line, and none of those exact
  invocations are recorded in-repo. If you need to reproduce a specific design run, you have to
  reconstruct its flags yourself from the factor→flag mapping in §2.1/§4.3, there's no saved
  master table.
- **`evaluate_experiment_seeds.py` does not compute**: AlgScore, MaxLCB/AUC_LCB, main-effects
  tables, ANOVA-style F-tests, pairwise-vs-Baseline comparisons at the α=0.05/0.10/0.20
  thresholds, or a "superlative ceiling" table distinct from the cross-seed aggregate — none of
  Chapter IV's statistical apparatus exists in this script. It only computes per-seed episode
  stats and a straightforward cross-seed t-distribution confidence interval (which does
  correctly treat each seed's mean as one data point — a genuine strength). **"Win Rate" (both
  CCA and HVAA survive), the thesis's headline metric, is never actually computed anywhere in
  this file**, though it's trivially derivable from two fields (`hvaa_survived`, `blue_killed`)
  that already exist per-episode.
- No Optuna hyperparameter-search script exists anywhere in the repo, despite the thesis's
  detailed two-stage Search One/Search Two description (150 + 70 trials). The winning
  hyperparameters are hardcoded as `argparse` defaults in `train_bace_clean.py`, but the search
  process that produced them isn't included.
- No plotting/analysis code exists anywhere (`matplotlib`/`pandas`/`seaborn`/`plotly` all
  return zero hits repo-wide) — the thesis's learning-curve figures, the hyperparameter
  Pearson-correlation bar chart, and the various summary tables were produced by tooling not
  included in this checkout.

### 5.7 Code-quality / practice issues

- ~~**No `requirements.txt`/environment file, no automated tests, no CI configuration anywhere**
  in the repository.~~ **Partially fixed:** `requirements.txt` now exists (§4.2) and a minimal
  regression test suite exists (`tests/`, §4.6). No CI configuration exists yet — that would
  need a CI provider decision (GitHub Actions, etc.) this repo doesn't currently have wired up.
- **Bare or overly-broad `except` clauses that silently swallow errors** appear repeatedly:
  `load_b_ace_config()` prints and returns `None` on any failure rather than raising;
  `rollback_eval_callback.py` has bare `except:` (no exception type at all) around VecNormalize
  stat syncing; `evaluate_experiment_seeds.py` has several `except: continue`/`except: return 0`
  patterns that convert malformed data into silent defaults rather than surfacing a problem;
  most `set_global_step` propagation methods across the wrapper files end in
  `except Exception: pass`.
- **Windows console portability risk — confirmed, not just theoretical.** `reward_breakdown_logger.py`
  and `rollback_eval_callback.py` (at least — likely more, not exhaustively grepped) print emoji
  characters directly to stdout. This was hit for real during setup on this machine: with the
  default Windows `cp1252` console codepage, `rollback_eval_callback.py`'s
  `print(f"⏭️  Skipping initial evaluation...")` raised `UnicodeEncodeError` and killed
  the run outright. **Workaround (no code change needed): set `PYTHONUTF8=1` in the environment
  before running any script in this repo** — see §4.3. This is the macOS-vs-Windows gap you'd
  expect, since the thesis itself was developed entirely on a Mac.
- **Docstrings that describe different parameter names/values than the actual code**, in
  multiple files: ~~`aggressive_hunter_fixed.py`'s module docstring lists parameters
  (`fire_threshold`, `fire_cooldown`, `g_force`, `turn_rate`) that don't exist in the actual
  constructor at all (the real names are `fire_range_min/max`, `fire_cooldown_steps`,
  `fire_cone`, `pursuit_gain`, `max_turn_rate`)~~ **fixed** as part of the §5.4 dead-code
  cleanup (the docstring now describes the real constructor parameters); ~~`hvaa_destruction_penalty.py`'s
  docstring calls its second class "HVAADestructionPenaltyAnnealed" when the real name is
  `HVAADestructionPenaltyWithDecay`~~ **moot** — that whole file was deleted (§5.4).
  **Still present, not touched:** `rollback_eval_callback.py`'s module docstring states defaults
  for `entropy_decay_factor`/`freeze_vecnormalize_until_new_best` that don't match the actual
  constructor signature (confirmed still true: docstring says `entropy_decay_factor` defaults to
  0.5, the actual default is 0.7).
- **The scripted expert's "Patrol" mode doesn't do what the thesis describes.** Thesis §3.4.1
  states patrol mode "commands a return to HVAA proximity ... and maintains station-keeping by
  flying toward the HVAA's position." The actual `AggressiveHunterFixed` third branch never
  reads any HVAA-related observation field (distance/angle-off/heading to HVAA are all defined
  in `get_default_obs_indices()` but unused in `get_action()`) — it only drives the aircraft's
  absolute heading toward zero, unrelated to the HVAA's actual position.
- **Substantial code duplication**: the observation-caching helper (`_cache_original_obs`/
  `_get_obs_for_expert`, ~150 lines) is copy-pasted near-verbatim between
  `expert_alignment_wrapper.py` and `expert_action_blending_wrapper.py` (not touched by the
  refactor described in the banner at the top of this document — still an open item).
  ~~`HVAASurvivalEveryNStepsBonus` and `FireCooldownWrapper` are independently redefined (with
  different parameters, per §5.3) in both `train_bace_clean.py` and
  `evaluate_experiment_seeds.py` instead of living in one shared module.~~ **Fixed** — both now
  live only in `env_factory.py` (§2.2), imported by both scripts.
- **Fragile unit handling in `Fighter.gd`'s altitude/pitch controller**: a raw altitude-error
  (in scaled Godot-meter units) is clamped directly into a pitch-angle-degree range with no
  explicit unit conversion or named gain constant — it only "works" because the current scale
  factor and pitch limits happen to produce overlapping numeric ranges; changing either
  constant elsewhere would silently break climb behavior with no warning.
- Two independently-written "find the repo root by walking up for a `b_ace_py` folder"
  functions (`run_experiment.py`, `evaluate_experiment_seeds.py`) have silently different
  (and differently wrong) fallback behavior on failure.
- Several large commented-out code blocks left in place rather than deleted (legacy
  termination-condition logic in `SimManager.gd::_physics_process`; a ~35-line commented-out
  duplicate function body in `Fighter.gd`).

---

## 6. Quick-Reference: What to Read First

If you're extending this codebase rather than just reading about it, in priority order:

1. `rl_files/env_factory.py` — read `build_env()` top to bottom; this is where every wrapper
   gets wired together now (both training and evaluation call it), and the file you'll edit if
   you're changing what wrappers exist or their order/parameters.
2. `rl_files/train_bace_clean.py` — read `main()`; this is what you'll edit for CLI flags,
   hyperparameters, and PPO/callback configuration (it no longer builds the wrapper stack
   itself — see #1).
3. `Godot_Air_Combat/components/Fighter.gd` — `get_obs()`, `set_action()`,
   `process_manouvers_action()`, `launch_missile_at_target()`.
4. `Godot_Air_Combat/components/Missile.gd` — the entire file is short enough to read in one
   sitting and is the crux of any missile-evasion work.
5. `Godot_Air_Combat/SimManager.gd` — `_physics_process()`, `inform_state()`,
   `_get_info_from_agents()`.
6. `b_ace_py/enriched_observation_wrapper_ablation.py` and `b_ace_py/pursuit_evasion_features.py`
   — if you're touching Factors A/B/C. If you change the raw observation vector's contents or
   order on the Godot side, also update `b_ace_py/contracts.py`'s `EXPECTED_OBSERVATION_LABELS`
   — it's checked against Godot's actual reported labels at every environment construction and
   will raise a clear error if they no longer match (see §2.1, §4.5).
7. `Godot_Air_Combat/addons/godot_rl_agents/B_ACE_sync.gd` — if you're touching the
   multi-agent protocol for 2v2.
