# BVR-CCA-Escort Codebase Guide

A working guide to the code behind AFIT-ENS-MS-26-M-085 (Bieger, "A Factorial Analysis of
Heuristic Feature Design and Training Guidance in Reinforcement Learning for
Beyond-Visual-Range Escort Missions"). This document explains what each part of the repo
does, how data actually flows end-to-end, how to set up the environment, and — in detail —
where the code diverges from what the thesis text describes. It was produced by reading
every Python file in full and auditing the Godot/GDScript simulation source; nothing in the
repo was modified to produce this guide.

> **How to read this document.** Section 1 is the map. Section 2 is "what do I change to do
> X." Section 3 is setup. Section 4 is the full issues list — read it before you trust any
> specific number quoted in the thesis against this code.

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
   `rl_files/train_bace_clean.py`'s `make_env()` closure, each doing `reward = reward + shaping`.
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
| `pursuit_evasion_features.py` | `PursuitEvasionFeatures`: the closed-form/approximated geometry for Apollonius circles, Basic Engagement Zones (BEZ/WEZ), Dynamic Maneuvering Cue (DMC), Active Target Defense Differential Game (ATDDG), and Range-Limited Pursuit-Evasion (RLPE). This is where the thesis's Factor A/B/C math lives. Its own `compute_features_for_agent()`/`get_feature_names()`/`extract_agent_state_from_obs()` are **dead code** — only used by the file's own `if __name__=="__main__"` self-test, never imported elsewhere. The methods that *are* actually used are the lower-level ones (`compute_apollonius_intercept`, `compute_bez_boundary`, `compute_dmc`, `compute_atddg_defense_features`, `compute_barrier_hyperbola_value`, `compute_optimal_intercept_heading`, `compute_critical_escape_heading`, `compute_capture_probability_proxy`), called individually by the wrapper below. |
| `enriched_observation_wrapper_ablation.py` | `EnrichedObservationWrapper`: appends the theoretical features to the base observation. `SplitNormalizationWrapper`: normalizes only the base (non-enriched) observation dims with its own Welford running-mean/var, leaving the enriched dims untouched, so that `VecNormalize` (with `norm_obs=False`) doesn't re-normalize them. Also defines `FeatureOnlyWrapper`/`create_enriched_env`/`get_ablation_summary`/`print_ablation_matrix` — **all dead code**, only exercised by this file's own demo block. |
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
| `train_bace_clean.py` (2,627 lines) | **The single most important file to understand.** Parses ~40 CLI flags, builds the full environment wrapper stack (hardcoded order, see §2.3 below), constructs a Stable-Baselines3 `PPO` model with the tuned hyperparameters as CLI defaults, trains via `model.learn()`, and runs a small ad-hoc final evaluation. Also defines ~15 custom `gym.Wrapper`/callback classes used nowhere else. |
| `run_experiment.py` (523 lines) | A generic multi-seed launcher: loops over `--seeds`, invokes `train_bace_clean.py` once per seed via `subprocess`, forwarding any flags it doesn't itself recognize straight through. **It does not know about the 5 experimental factors or the 16-18 design configurations** — see §4.1, this is an important correction to how you might expect this file to work. |
| `evaluate_experiment_seeds.py` (2,173 lines) | Loads trained checkpoints + `VecNormalize` stats per seed, runs deterministic evaluation episodes, and computes per-seed / cross-seed statistics. Implements far less of the thesis's Chapter IV statistical machinery than you'd expect — see §4.1. |

**Wrapper/callback files** (all imported and layered together inside `train_bace_clean.py`'s
`make_env()`):

| File | Class(es) actually used | Maps to |
|---|---|---|
| `expert_alignment_wrapper.py` | `ExpertAlignmentWrapper` | Factor D (alignment-based reward shaping) |
| `expert_action_blending_wrapper.py` | `ExpertActionBlendingWrapper` | Factor E (action blending) |
| `aggressive_hunter_fixed.py` | `AggressiveHunterFixed` (2 sibling classes, `AggressiveHunterV3`/`AggressiveHunterTunable`, are dead code) | The scripted "expert" FSM that Factors D and E are built around |
| `sustained_turn_penalty.py` | `SustainedTurnPenaltySimple` | r_shape sustained-turn penalty |
| `track_aware_shaping.py` | `CombinedTrackAwareShaping` | r_shape track-loss penalty + range-closure bonus |
| `mission_tempo_shaping_perstep.py` | `MissionTempoShapingPerStep` | r_shape "red still alive" / mission-tempo terms |
| `hvaa_destruction_penalty.py` | *(neither class is ever instantiated — dead code)* | Would-be r_env "-15 HVAA destroyed" term |
| `reward_breakdown_logger.py` | `CompactRewardLogger` (`RewardBreakdownLogger` is imported but never instantiated) | Diagnostics only, no effect on reward |
| `rollback_eval_callback.py` | `RollbackEvaluationCallbackTopK` | The checkpoint-rollback mechanism (thesis §4.2) |

Several more reward-shaping wrappers are defined **inline inside `train_bace_clean.py` itself**
(not in their own files), which is easy to miss: `HVAASurvivalEveryNStepsBonus`,
`KillRewardWrapper`, `ActionSmoothnessPenalty`, `RangeClosingShaping` (unused),
`FireCooldownWrapper`, `HeadingRateLimitWrapper`, `NanGuardWrapper`, plus two full alternative
PPO implementations, `ExpertRolloutBuffer`/`ExpertMixPPO` (~370 lines) — **which are never
instantiated anywhere in the codebase.** `main()` builds a vanilla `PPO(...)` (line ~2321), not
`ExpertMixPPO`. This is a large, sophisticated, fully-dead alternative implementation of the
same idea Factor E's `ExpertActionBlendingWrapper` implements more simply — don't spend time
trying to trace it into the live training path, it isn't there.

### 2.3 The actual wrapper stack, in order (as built by `make_env()`)

This is the order environment `.step()` calls actually propagate through — reward
contributions accumulate bottom-to-top relative to this list (innermost/first-applied at top):

1. `SingleAgentBACEEnv` — collapses the multi-agent PettingZoo env to one Gym `Env` (see §2.4 for why this matters for 2v2)
2. `EnrichedObservationWrapper` + `SplitNormalizationWrapper` *(only if `--use-enriched-obs`)*
3. `HeadingRateLimitWrapper`
4. `KillRewardWrapper`
5. `ActionSmoothnessPenalty`
6. `MissionTempoShapingPerStep`
7. `SustainedTurnPenaltySimple`
8. `CombinedTrackAwareShaping`
9. `FireCooldownWrapper`
10. `HVAASurvivalEveryNStepsBonus`
11. `ExpertAlignmentWrapper` *(only if `--expert-alignment`, training env only)*
12. `ExpertActionBlendingWrapper` *(only if `--expert-blending`, training env only)*
13. `CompactRewardLogger`
14. `NanGuardWrapper`
15. `Monitor` (SB3)
16. `DummyVecEnv` → `VecNormalize`

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
requires new Godot code; (4) add a matching reward-shaping term (a new small wrapper following
the `hvaa_destruction_penalty.py` pattern) once the signal exists, and a matching evaluation
metric (following the exact pattern `evaluate_experiment_seeds.py` already uses for
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

1. **`SingleAgentBACEEnv`** (defined near-identically in both `train_bace_clean.py` and
   `evaluate_experiment_seeds.py`) is the real blocker, not the Godot side. Its `step()`
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
   indices, `mission_tempo_shaping_perstep.py`/`hvaa_destruction_penalty.py`'s single boolean
   `red_killed_event`/`hvaa_alive` gates. These need either a redesigned per-track schema with
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

There is **no `requirements.txt`, `pyproject.toml`, `setup.py`, or root `README`** anywhere in
this repository, and no automated tests or CI configuration. Everything below was reconstructed
from imports/config files, not from an in-repo setup script — expect to iterate.

### 4.1 Godot side

1. Install **Godot Engine 4.5** (confirmed via `project.godot`'s `config/features=PackedStringArray("4.5")`
   and the presence of `.gd.uid` sidecar files, which Godot only auto-generates from 4.4+).
   Get it from the official Godot site — do not use an older 4.x or a 3.x build, the project
   will not open correctly.
2. Open `Godot_Air_Combat/project.godot` in the editor once, so Godot can regenerate its local
   `.godot/` import cache (this directory is gitignored and must be rebuilt locally).
3. **You need to export a native binary — the repo does not ship one, and `bin/` does not
   exist yet.** `train_bace_clean.py`'s `_resolve_env_path()` expects, on Windows specifically:
   `<repo_root>/bin/B_ACE_v0.1.exe`.
4. **There is no Windows export preset in this project.** `export_presets.cfg` only defines
   `macOS` and `Linux` presets (consistent with the thesis's own statement that all experiments
   were run on an Apple M3 MacBook). In the Godot editor: **Project → Export... → Add... →
   Windows Desktop**, configure it (you'll need to download the Godot export templates matching
   your 4.5 install, via Editor → Manage Export Templates if prompted), then **Export Project**,
   saving to `bin/B_ACE_v0.1.exe` relative to the repo root. Export as a release/non-debug
   build for training throughput.
5. Do this *before* trying to run any Python training script — `train_bace_clean.py` will raise
   a clear `FileNotFoundError` telling you exactly which path it checked and failed to find if
   you skip this step.

### 4.2 Python side

No pinned versions exist in-repo, so the following is inferred from actual code usage, not
guaranteed-exact:

- **Python 3.10+** is required — `train_bace_clean.py` uses PEP 604 union-type syntax
  (`int | None` in a function signature), which is a hard 3.10 floor.
- Install:
  ```
  pip install gymnasium pettingzoo numpy scipy torch stable-baselines3 tensorboard
  ```
  (`tensorboard` is needed because `train_bace_clean.py` configures an SB3 logger with the
  `"tensorboard"` output format.) **Do not** additionally install the `godot-rl-agents` PyPI
  package — this repo reimplements its own minimal socket client
  (`b_ace_py/godot_env.py`) rather than depending on that library; installing it isn't
  necessary and isn't imported anywhere.
- Note the gap called out in §4.1 below: no hyperparameter-search (Optuna) script and no
  plotting/analysis code (matplotlib/pandas/etc.) exist anywhere in this repo, despite both
  being extensively described in the thesis. If you need either, you'll be writing them
  yourself around the existing training/eval scripts.

### 4.3 Running it

```bash
# From the repo root, with the Godot binary already exported to bin/B_ACE_v0.1.exe

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

Important gotchas confirmed during the audit, not documented anywhere in-code:
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
  it will silently reconstruct the wrong features for that checkpoint (see §4.1).

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

### 5.3 Train/evaluation mismatches (results from training may not reflect what gets reported)

- `evaluate_experiment_seeds.py` reimplements `HVAASurvivalEveryNStepsBonus` and
  `FireCooldownWrapper` from scratch rather than importing the training versions, with
  **different parameters**: the survival bonus uses `every_n_steps=50, bonus=0.05` at eval vs.
  `every_n_steps=25, bonus=0.01` at training (≈2.5x the average per-step rate at eval); the fire
  cooldown is a fixed 120-step cooldown at eval vs. training's annealed 70→120-step schedule
  (matters most when evaluating early checkpoints, since it enforces end-of-training discipline
  on a policy that was actually trained under a looser cooldown at that point). Net effect:
  **the "mean_reward"/AUC reported by the evaluation script is not pure `r_env`** — it still
  includes at least one shaping term, at a different rate than training used.
- `HeadingRateLimitWrapper` (a pure action-space transform, not reward shaping) is absent
  entirely at evaluation — raw policy actions reach the simulator unfiltered, even though the
  policy was trained assuming its outputs get rate-limited first.
- The observation-enrichment `ablation_config` used for a given checkpoint is not saved
  anywhere `evaluate_experiment_seeds.py` reads automatically; it guesses "which observation
  mode" a checkpoint used purely from total dimensionality via hardcoded thresholds
  (`obs_dim <= 30` → baseline, `<= 50` → enriched). If you evaluate a `geometry_only`-trained
  model without passing a matching `--ablation-config`, the eval script will silently
  reconstruct **all** enriched features instead — same total dimension, semantically wrong
  content, and nothing will error.

### 5.4 Dead code (large, could waste significant time if you try to trace it as "live")

- `ExpertRolloutBuffer`/`ExpertMixPPO` in `train_bace_clean.py` (~370 lines) — a full alternative
  PPO implementation with an imitation/behavioral-cloning loss and rollout-level expert-action
  mixing. Never instantiated; `main()` builds vanilla `PPO`.
- `hvaa_destruction_penalty.py` — both classes (`HVAADestructionPenalty`,
  `HVAADestructionPenaltyWithDecay`) are imported but never instantiated.
- `RewardBreakdownLogger` (the verbose diagnostic logger, as opposed to `CompactRewardLogger`
  which *is* used) — imported, never instantiated. Its `DEFAULT_KEYS` also reference several
  info-dict keys no currently-active wrapper actually emits, so re-enabling it today would
  silently under-report several reward categories rather than error.
- `Godot_Air_Combat/B_ACE.gd` — orphaned root-scene script, not referenced by any `.tscn`.
- Two of the three `aggressive_hunter_fixed.py` classes (`AggressiveHunterV3`,
  `AggressiveHunterTunable`) — ~95% copy-pasted from the one that's actually used.
- `pursuit_evasion_features.py`'s `compute_features_for_agent`/`get_feature_names`/
  `extract_agent_state_from_obs`, and `enriched_observation_wrapper_ablation.py`'s
  `FeatureOnlyWrapper`/`create_enriched_env`/`get_ablation_summary`/`print_ablation_matrix` —
  all only exercised by each file's own `__main__` self-test block.
- `RangeClosingShaping` (defined inline in `train_bace_clean.py`) — defined, never instantiated.
- `DEBUG_HVAA_OBS` module-level flag in `train_bace_clean.py` — defined, never read.
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

- **No `requirements.txt`/environment file, no automated tests, no CI configuration anywhere**
  in the repository.
- **Bare or overly-broad `except` clauses that silently swallow errors** appear repeatedly:
  `load_b_ace_config()` prints and returns `None` on any failure rather than raising;
  `rollback_eval_callback.py` has bare `except:` (no exception type at all) around VecNormalize
  stat syncing; `evaluate_experiment_seeds.py` has several `except: continue`/`except: return 0`
  patterns that convert malformed data into silent defaults rather than surfacing a problem;
  most `set_global_step` propagation methods across the wrapper files end in
  `except Exception: pass`.
- **Windows console portability risk:** `reward_breakdown_logger.py` prints emoji characters
  (📊, 🎯) directly to stdout. Given the default Windows console codepage is frequently not
  UTF-8, this can raise `UnicodeEncodeError` and crash a training run outright — directly
  relevant if you're running this on Windows, as opposed to the macOS environment the thesis
  itself was developed on.
- **Docstrings that describe different parameter names/values than the actual code**, in
  multiple files: `aggressive_hunter_fixed.py`'s module docstring lists parameters
  (`fire_threshold`, `fire_cooldown`, `g_force`, `turn_rate`) that don't exist in the actual
  constructor at all (the real names are `fire_range_min/max`, `fire_cooldown_steps`,
  `fire_cone`, `pursuit_gain`, `max_turn_rate`); `hvaa_destruction_penalty.py`'s docstring calls
  its second class "HVAADestructionPenaltyAnnealed" when the real name is
  `HVAADestructionPenaltyWithDecay`; `rollback_eval_callback.py`'s module docstring states
  defaults for `entropy_decay_factor`/`freeze_vecnormalize_until_new_best` that don't match the
  actual constructor signature.
- **The scripted expert's "Patrol" mode doesn't do what the thesis describes.** Thesis §3.4.1
  states patrol mode "commands a return to HVAA proximity ... and maintains station-keeping by
  flying toward the HVAA's position." The actual `AggressiveHunterFixed` third branch never
  reads any HVAA-related observation field (distance/angle-off/heading to HVAA are all defined
  in `get_default_obs_indices()` but unused in `get_action()`) — it only drives the aircraft's
  absolute heading toward zero, unrelated to the HVAA's actual position.
- **Substantial code duplication**: the observation-caching helper (`_cache_original_obs`/
  `_get_obs_for_expert`, ~150 lines) is copy-pasted near-verbatim between
  `expert_alignment_wrapper.py` and `expert_action_blending_wrapper.py`; `HVAASurvivalEveryNStepsBonus`
  and `FireCooldownWrapper` are independently redefined (with different parameters, per §5.3)
  in both `train_bace_clean.py` and `evaluate_experiment_seeds.py` instead of living in one
  shared module.
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

1. `rl_files/train_bace_clean.py` — read `main()` and `make_env()` top to bottom; this is
   where every wrapper gets wired together and is the one file you'll edit most.
2. `Godot_Air_Combat/components/Fighter.gd` — `get_obs()`, `set_action()`,
   `process_manouvers_action()`, `launch_missile_at_target()`.
3. `Godot_Air_Combat/components/Missile.gd` — the entire file is short enough to read in one
   sitting and is the crux of any missile-evasion work.
4. `Godot_Air_Combat/SimManager.gd` — `_physics_process()`, `inform_state()`,
   `_get_info_from_agents()`.
5. `b_ace_py/enriched_observation_wrapper_ablation.py` and `b_ace_py/pursuit_evasion_features.py`
   — if you're touching Factors A/B/C.
6. `Godot_Air_Combat/addons/godot_rl_agents/B_ACE_sync.gd` — if you're touching the
   multi-agent protocol for 2v2.
