# Improvement Plan 1: Viewing Results, Running the Code, and Modifying Parameters

**Audience:** this file is written to be read by a human *and* handed to an LLM coding
assistant as an implementation spec. If you're an LLM implementing this: read the
[Ground Rules](#ground-rules) section first and follow it strictly.

**Repo:** `bvr-cca-escort` (AFIT-ENS-MS-26-M-085, Bieger). Background reading, if you need
more context on the codebase than this file provides: `CODEBASE_GUIDE.md` (full architecture +
audit) and `HOW_TO_RUN.md` (current run instructions) at the repo root.

---

## Ground Rules

1. **Do not change any training, reward, observation, or simulation behavior.** Every change
   in this plan must be additive tooling (dashboards, logging integrations, config-file
   runners) or a pure ergonomics change (e.g., a YAML file that maps 1:1 to existing CLI flags).
   If implementing a task here would change what number comes out of the reward function, what
   the observation vector contains, or how a policy is trained, **stop and flag it** — that's
   out of scope for this plan.
2. **Every existing CLI flag on `train_bace_clean.py`, `run_experiment.py`, and
   `evaluate_experiment_seeds.py` must keep working exactly as it does today, with exactly the
   same defaults.** New tooling wraps these scripts; it does not replace or fork them.
3. Prefer the smallest change that gets the job done. Task 1 (below) is genuinely a few lines
   in one file and should be done first regardless of anything else in this plan.
4. This plan is broken into independent tasks. Implement them in order, but each one should be
   a working, committable unit on its own — don't leave the repo in a half-finished state
   between tasks.
5. When a task says "verify," actually run the thing and confirm it works before considering
   the task done — don't just write code that looks plausible.

---

## Current State (read this before changing anything)

### What already exists for viewing results

**TensorBoard is already fully wired up — this is the first thing to try, and it requires zero
new code.** `src/rl_files/train_bace_clean.py` configures an SB3 logger with both `"stdout"` and
`"tensorboard"` output formats (`configure(log_dir.as_posix(), ["stdout", "tensorboard"])`,
near line 2271), and writes event files to `results/runs_sb3/<experiment_name>/<run_name>/`. To view:

```bash
tensorboard --logdir results/runs_sb3/<experiment_name>
```

This gives you, per run, out of the box: `eval/mean_reward`, `eval/std_reward`,
`eval/mean_ep_length` (from `EvaluationBasedCheckpointCallback` / the eval side of
`RollbackEvaluationCallbackTopK`), `rollout/mean_reward_100ep`, `rollout/std_reward_100ep`
(from `MetricsCollectionCallback`), `train/policy_gradient_loss`, `train/value_loss`,
`train/entropy_loss`, `train/clip_fraction`, `train/learning_rate`, `train/entropy_coef`,
`train/clip_range` (standard SB3 PPO diagnostics), and — if `RollbackEvaluationCallbackTopK`'s
rollback mechanism fires — rollback events. Pointing `--logdir` at
`results/runs_sb3/<experiment_name>` (the parent of all seed subfolders) overlays every seed's curves
on the same charts automatically, which is the natural way to see cross-seed variance.

**No Weights & Biases (or any other cloud dashboard) integration exists anywhere in the repo**
today — confirmed by grepping for `wandb` across `src/rl_files/` and `src/b_ace_py/`: zero hits. There
is a stale `wandb/` line in `.gitignore` suggesting someone considered it at some point, but
nothing was ever wired up.

**Other existing result artifacts, not currently visualized anywhere:**
- `results/runs_sb3/<run>/monitor.csv` — raw per-episode reward/length from SB3's `Monitor` wrapper.
- `results/runs_sb3/<run>/training_results_seed<N>.json` — end-of-run dump of hyperparameters, full
  episode reward/length history (downsampled), and the evaluation-callback summary. Written by
  `train_bace_clean.py::main()` near line 2606.
- `results/runs_sb3/<run>/eval_checkpoints/` and `results/runs_sb3/<run>/best_checkpoint/` — saved model +
  `VecNormalize` snapshots at various points, produced by `RollbackEvaluationCallbackTopK`.
- Console output from `src/rl_files/reward_breakdown_logger.py::CompactRewardLogger` — prints a
  running per-episode breakdown of reward components (tempo, range, track-loss, etc.) every 10
  episodes. Text-only, terminal-only, not persisted anywhere.
- Output of `src/rl_files/evaluate_experiment_seeds.py` — writes its own JSON summary (per-seed and
  aggregate statistics) to the experiment directory. No plotting.

**Nothing in the repo currently plots anything.** No `matplotlib`, `pandas`, `seaborn`, or
`plotly` anywhere (confirmed by grep). Any chart the thesis contains (learning curves,
hyperparameter correlation plot, etc.) was produced by tooling that isn't in this repository.

**Visual playback of an actual episode (watching the aircraft fly, not just numbers) already
works too**, via a dedicated config: `src/b_ace_py/Watch_B_ACE_config.json` (renderize on,
real-time speed) — see `HOW_TO_RUN.md` for exact commands. This is a visualization of
*behavior*, complementary to the metrics dashboards this plan is about.

### What already exists for running the code

Everything is CLI-flag driven — there is no config-file-based runner and no GUI. The three
entry points and roughly how many flags each has:
- `src/rl_files/train_bace_clean.py` — ~40 flags (`--total-timesteps`, `--use-enriched-obs`,
  `--ablation-config`, `--expert-alignment`, `--expert-blending`, all PPO hyperparameters, etc.)
- `src/rl_files/run_experiment.py` — a thin multi-seed wrapper around the above; forwards any flag
  it doesn't recognize straight through to `train_bace_clean.py`.
- `src/rl_files/evaluate_experiment_seeds.py` — its own ~20 flags for loading and scoring
  checkpoints.

`HOW_TO_RUN.md` documents exact commands for common cases. There is currently no way to define
"a named experiment configuration" once and reuse it — every run is a fresh, fully-spelled-out
command line, which is error-prone (a forgotten `--total-timesteps` silently falls back to a
value 10x too small, for example — this is already called out in `HOW_TO_RUN.md`).

### Where parameters actually live (reference table)

This is the single most useful thing in this document if you just want to change a number.
"CLI flag" means you can change it without editing any file. "Hardcoded" means you must edit
the named file/location directly — there is no flag for it.

| Parameter | Where | How to change |
|---|---|---|
| Total training timesteps | CLI | `--total-timesteps` |
| Random seed | CLI | `--seed` |
| Learning rate schedule (start/end) | CLI | `--initial-learning-rate` / `--final-learning-rate` |
| Entropy coefficient schedule | CLI | `--initial-entropy-coef` / `--final-entropy-coef` |
| LR/entropy decay shape | CLI | `--decay-schedule` (`linear`/`exponential`) |
| PPO clip range | CLI | `--clip-eps` |
| GAE lambda, gamma | CLI | `--gae-lambda`, `--gamma` |
| Rollout length / batch size / epochs | CLI | `--n-steps` / `--batch-size` / `--epochs` |
| Network architecture (layers/width) | CLI | `--n_layers` / `--layer_size` |
| Which observation-feature factors (A/B/C) are on | CLI | `--use-enriched-obs --ablation-config <name>` — see the table in `HOW_TO_RUN.md` |
| Alignment reward shaping (Factor D) | CLI | `--expert-alignment`, `--alignment-coef`, `--alignment-decay-start/end` |
| Action blending (Factor E) | CLI | `--expert-blending`, `--blend-initial-alpha`, `--blend-final-alpha`, `--blend-decay-start/end/type`, `--blend-fire-mode` |
| Scenario/reward config (spawn zones, Godot-side reward weights) | Hardcoded (JSON) | `src/b_ace_py/SimpleExample_B_ACE_config.json` (the default), or pass `--config <path>` to point at a different JSON file, e.g. `src/b_ace_py/Watch_B_ACE_config.json` |
| Fire cooldown curriculum (start/end/anneal steps) | Hardcoded | `src/rl_files/train_bace_clean.py`, inside `make_env()`, the `FireCooldownWrapper(...)` call (~line 2138) |
| Kill bonus reward | Hardcoded | `src/rl_files/train_bace_clean.py`, `KillRewardWrapper(...)` call (~line 2100) |
| Action smoothness penalty schedule | Hardcoded | `src/rl_files/train_bace_clean.py`, `ActionSmoothnessPenalty(...)` call (~line 2102) |
| Mission tempo shaping (per-step penalty/bonus) | Hardcoded | `src/rl_files/train_bace_clean.py`, `MissionTempoShapingPerStep(...)` call (~line 2111) |
| Sustained-turn penalty | Hardcoded | `src/rl_files/train_bace_clean.py`, `SustainedTurnPenaltySimple(...)` call (~line 2118) |
| Track-loss / range-closure shaping | Hardcoded | `src/rl_files/train_bace_clean.py`, `CombinedTrackAwareShaping(...)` call (~line 2126) |
| HVAA-alive survival bonus | Hardcoded | `src/rl_files/train_bace_clean.py`, `HVAASurvivalEveryNStepsBonus(...)` call (~line 2147) |
| Heading-rate action limiter | Hardcoded | `src/rl_files/train_bace_clean.py`, `HeadingRateLimitWrapper(...)` call (~line 2098) |
| Periodic-evaluation frequency / episode count | Hardcoded | `src/rl_files/train_bace_clean.py`, the `RollbackEvaluationCallbackTopK(...)` constructor call (~line 2404) — `eval_freq`, `n_eval_episodes` |
| Rollback thresholds (when to revert to an earlier checkpoint) | Hardcoded | Same `RollbackEvaluationCallbackTopK(...)` call — `deficit_threshold`, `rollback_patience`, `activation_threshold`, `min_timesteps_before_rollback` |
| Scripted-expert FSM behavior (fire envelope, pursuit memory, etc.) | Hardcoded | `src/rl_files/aggressive_hunter_fixed.py`, `AggressiveHunterFixed.__init__` defaults |
| Final evaluation episode count (post-training smoke check inside `train_bace_clean.py`) | Hardcoded | `src/rl_files/train_bace_clean.py`, the `while episodes_completed < 20:` loop near line 2461 |

If a parameter you want isn't in this table, it's either genuinely a CLI flag not listed here
(check `src/rl_files/train_bace_clean.py`'s `_parse_args()`, ~line 1543 onward) or it's inside one
of the reward-shaping wrapper *class definitions themselves* (their `__init__` defaults) rather
than at the call site — see Improvement Plan 2 for why there are so many of these and how that
could be consolidated.

---

## Task 1: Add Weights & Biases as an optional, additive logging sink (do this first)

**Why this first:** it's the lowest-risk, highest-payoff change in this whole plan. W&B can
mirror an existing TensorBoard log directory with almost no code — this does not touch
training logic at all.

**What to do:**
1. Add `wandb` to the project's dependency list (see Improvement Plan 2, Task on dependency
   pinning, for where that list should live — if it doesn't exist yet, just note in your PR/commit
   that `pip install wandb` is now an optional extra).
2. In `src/rl_files/train_bace_clean.py`, near where the SB3 logger is configured (search for
   `configure(log_dir.as_posix(), ["stdout", "tensorboard"])`), add an **opt-in** W&B hook:
   - New CLI flag: `--use-wandb` (default `False`, `action="store_true"`), plus
     `--wandb-project` (default e.g. `"bvr-cca-escort"`) and `--wandb-entity` (default `None`).
   - When `--use-wandb` is set, call `wandb.init(project=args.wandb_project,
     entity=args.wandb_entity, name=run_name, config=training_config, sync_tensorboard=True)`
     *before* the SB3 logger is created, and call `wandb.finish()` at the very end of `main()`
     (including in whatever exception-safe place makes sense — a `try/finally` around the
     training body is reasonable, but don't restructure the whole function just for this).
   - `sync_tensorboard=True` is doing the heavy lifting here: W&B will automatically mirror
     every scalar already being written to the existing TensorBoard event files. **Do not**
     add separate `wandb.log(...)` calls duplicating what TensorBoard already logs — that
     would risk drifting out of sync with what's actually being trained on, and duplicates
     effort for no benefit.
   - If `--use-wandb` is not passed, none of this should run, and behavior must be pixel-for-
     pixel identical to today (this is important: most users of this repo will not have a W&B
     account, and the script must work exactly as before with zero new required dependencies).
3. Mirror the same three flags (`--use-wandb`, `--wandb-project`, `--wandb-entity`) onto
   `src/rl_files/run_experiment.py` so a multi-seed sweep can opt in too (it just needs to forward
   them through to each `train_bace_clean.py` invocation the same way it forwards everything
   else — check whether they already pass through via `parse_known_args()`'s catch-all before
   adding special-case handling).
4. **Verify:** run a short training job (e.g. `--total-timesteps 2048 --use-wandb`) and confirm
   a run appears in the W&B project dashboard with live-updating charts, and confirm a second
   short run *without* `--use-wandb` behaves exactly as before (no W&B-related output, no new
   files, no crash if `wandb` isn't installed at all — guard the `import wandb` so a missing
   package only breaks things when `--use-wandb` is actually requested, not unconditionally at
   import time).

**Acceptance criteria:** training with `--use-wandb` produces a live W&B dashboard showing the
same curves as TensorBoard already does; training without it is unaffected; the script doesn't
hard-require the `wandb` package to be installed unless the flag is used.

---

## Task 2: A results-browsing dashboard for local runs (no cloud account required)

This is for people who don't want a W&B account, or who want a purpose-built view of *this*
project's metrics (win-adjacent outcomes, episode termination reasons) rather than generic RL
training curves.

**Recommended approach:** a single-file [Streamlit](https://streamlit.io/) app (`pip install
streamlit`), since it needs no build step, no frontend code, and can be launched with one
command (`streamlit run src/tools/dashboard.py`) — this matches "intuitive," is easy for a
follow-on maintainer to extend, and doesn't require standing up a server.

**What it should do**, reading *only already-existing* output files (do not change what
`train_bace_clean.py` or `evaluate_experiment_seeds.py` write out):
1. **Run picker.** Scan `results/runs_sb3/` for experiment directories and seed subdirectories (same
   directory-walking logic `evaluate_experiment_seeds.py::find_seed_models()` already uses —
   reuse or closely mirror it rather than reinventing directory-parsing logic). Let the user
   pick one or more runs to compare.
2. **Training curves.** Read `monitor.csv` (per-episode reward/length) and, if present,
   `training_results_seed<N>.json`'s `metrics.episode_rewards`/`timesteps` for the
   downsampled full-history view. Plot reward-vs-timestep with a rolling mean, one line per
   selected run, using `matplotlib` or `plotly` (either is fine — `plotly` gives free
   interactivity, `matplotlib` has zero extra dependency risk beyond what's already implied by
   the rest of this plan).
3. **Evaluation summary.** If `training_results_seed<N>.json`'s `evaluation_summary` block is
   present, plot `mean_rewards` vs. `timesteps` with a shaded band for `std_rewards` — this is
   the closest thing to a "learning curve with confidence interval" the thesis describes, and
   it's already sitting in the JSON, unused.
4. **Config/hyperparameter summary.** Show the `config` block from
   `training_results_seed<N>.json` as a simple table (which factors were on, what the actual
   hyperparameters were) so it's obvious at a glance what a given run's flags were, without
   needing to remember the original command line.
5. **Link out to TensorBoard.** Don't try to reimplement TensorBoard's own charts inside this
   dashboard — for training-internals diagnostics (policy loss, entropy, clip fraction), just
   show the exact `tensorboard --logdir ...` command to run, or embed TensorBoard via its
   `tensorboard.program` API if that turns out to be low-effort, but don't block the rest of
   the dashboard on getting that embed working.
6. Put this at `src/tools/dashboard.py` (create the `src/tools/` directory if it doesn't exist) and add
   a short usage note to `HOW_TO_RUN.md` once it's implemented.

**Acceptance criteria:** `streamlit run src/tools/dashboard.py` opens a browser tab, lets you pick
one or more completed runs from `results/runs_sb3/`, and shows at minimum a reward-vs-timestep chart
and the run's hyperparameter config, without requiring the training scripts to be modified in
any way that changes what they compute or write out.

---

## Task 3: A declarative, reusable way to define and launch a run

**Problem this solves:** today, running a specific experimental configuration means
remembering/reconstructing a long command line every time (see `HOW_TO_RUN.md`'s factor table).
There's no way to save "this is what design run BD means" once and reuse it.

**What to do:**
1. Create a `config/launch/` directory (repo root) containing one YAML file per named run
   configuration, e.g. `config/launch/baseline.yaml`, `config/launch/factor_B.yaml`, `config/launch/factor_BD.yaml`.
   Each YAML file's keys should be **a 1:1 mapping to existing `train_bace_clean.py` CLI flag
   names** (with dashes or underscores — pick one and be consistent), e.g.:
   ```yaml
   experiment_name: factor_BD
   total_timesteps: 10000000
   use_enriched_obs: true
   ablation_config: engagement_only
   expert_alignment: true
   ```
2. Add a thin launcher script, `src/rl_files/run_from_config.py`, that: loads a YAML file, converts
   its keys into the equivalent `--flag value` / `--flag` (for booleans) arguments, and either
   (a) invokes `train_bace_clean.py`'s existing `main()` with those arguments injected into
   `sys.argv` before calling it, or (b) shells out to `python train_bace_clean.py <args>` via
   `subprocess`. Prefer (b) — it's simpler, and it guarantees the YAML path can never
   accidentally diverge from what a direct CLI invocation does, since it's *building* a real
   CLI invocation rather than reimplementing argument handling.
   - This script **must not** hardcode a duplicate copy of argument defaults or logic — its
     only job is "YAML keys → CLI flags," nothing else. If a key in the YAML doesn't correspond
     to a real flag on `train_bace_clean.py`, it should error loudly, not silently ignore it.
   - Support a `--seeds` list in the YAML (optional) that, if present, loops the same way
     `run_experiment.py` does — or better, just have this script delegate to
     `run_experiment.py` when multiple seeds are requested, rather than reimplementing the
     seed loop a third time.
3. Usage should end up being: `python src/rl_files/run_from_config.py config/launch/factor_BD.yaml`.
4. **Verify:** confirm that a YAML-launched run and the equivalent hand-written CLI command
   produce byte-identical `train_bace_clean.py` invocations (e.g., print the constructed
   command and diff it against the manual one) before trusting this for real experiments.

**Acceptance criteria:** each of the five example configs above (baseline + one factor + one
combo, at minimum) can be launched with a single short command, and produces identical
behavior to the equivalent hand-written CLI invocation documented in `HOW_TO_RUN.md`.

---

## Explicitly out of scope for this plan

- Anything that changes reward values, observation contents, network architecture defaults, or
  training dynamics. If you think something here requires that, stop and ask rather than doing
  it — this plan is tooling-only.
- Rewriting `train_bace_clean.py`'s argument parsing to use YAML/config natively instead of
  argparse. That's a bigger, riskier change than what's being asked for here; Task 3 above
  deliberately layers *on top of* the existing CLI rather than replacing it.
- A full experiment-tracking rewrite (e.g., MLflow). W&B (Task 1) already covers the "hosted
  dashboard" need with far less integration work.
