# `mostly-original` vs `new-code`: Branch Comparison

This repository has two relevant branches beyond `main` (which is identical to
`mostly-original`): **`mostly-original`** (the previous student's code, plus this session's
minimal bug fixes) and **`new-code`** (a deliberate refactor/hardening pass built on top of
the same codebase). This document compares them: what's functionally different, what's the
same, and exactly how to run and analyze an experiment on either one.

Companion docs: [CODEBASE_GUIDE.md](CODEBASE_GUIDE.md) (file-by-file map, `mostly-original`),
[PERFORMANCE_DIAGNOSIS_AND_PILOT_PLAN.md](PERFORMANCE_DIAGNOSIS_AND_PILOT_PLAN.md) (bug fixes
and pilot results on `mostly-original`), [RUNNING_THE_EXPERIMENT.md](RUNNING_THE_EXPERIMENT.md)
(command reference for `mostly-original`).

---

## 1. The core relationship between the branches

`new-code` is **not** a different experiment or a different model — it's a deliberate
refactor/hardening pass over the exact same simulation, reward function, and training logic,
verified with a structural equivalence-check tool (`src/tools/equivalence_check.py`) before/after
each change. The Godot simulation itself is untouched on both branches. The intent, per its
own commit message, was zero behavior change except one explicitly-called-out exception (§3.1).

---

## 2. Bugs found and fixed this session — already fixed independently on `new-code`

All three real bugs found and fixed on `mostly-original` during this session's pilot
experiment already existed and were already fixed on `new-code` — confirming both fixes
independently converged on the same root causes.

| Bug | `mostly-original` fix (this session) | `new-code` fix |
|---|---|---|
| `KeyError: 'action_spaces'` crash on every environment construction | Deleted the erroneous overwrite line | Identical fix |
| `np.cross()` on 2D vectors (NumPy 2.x), breaking all of Factor A (ATDDG) | `AT_unit[0]*AD_vec[1] - AT_unit[1]*AD_vec[0]` | **Byte-for-byte identical formula** |
| `model.save()` pickling crash (`TypeError: cannot pickle '_thread.lock' object`) at the end of every run | Wrapped the one crashing call in `try/except` (symptom-level patch) | **Root-cause fix**: replaced the `model.train` monkey-patch with a `SkippableTrainPPO(PPO)` subclass, so nothing is ever added to `model.__dict__` in the first place. Cleaner, but a larger structural change than made on `mostly-original` given the instruction to keep that branch's changes minimal. |

---

## 3. Real functional differences (not bug fixes — actual behavior/capability changes)

### 3.1 Evaluation wrapper stack — the one deliberate, confirmed behavior change

On `mostly-original`, `evaluate_experiment_seeds.py` builds its own, independently-drifted
copy of the environment wrapper stack — missing roughly 8 wrappers that training uses
(`HeadingRateLimitWrapper`, `ActionSmoothnessPenalty`, `MissionTempoShapingPerStep`,
`SustainedTurnPenaltySimple`, `CombinedTrackAwareShaping`, `KillRewardWrapper`, and others).

`new-code` introduces `src/rl_files/env_factory.py` as the **single shared** `build_env()` used by
both training and evaluation, so evaluation now uses the exact same reward-shaping stack
training does.

**Consequence: reward numbers from `evaluate_experiment_seeds.py` are not directly comparable
between the two branches.** Mission-outcome metrics (HVAA survival %, red kill %, blue loss %)
are unaffected, since those are ground-truth episode facts independent of reward-shaping
composition — only the "Mean Reward" column differs.

### 3.2 YAML-based run launcher (`new-code` only)

`src/rl_files/run_from_config.py` + `config/launch/*.yaml` — doesn't exist on `mostly-original`. Save a
design run's flags once, launch with:
```powershell
python src/rl_files/run_from_config.py config/launch/factor_BD.yaml
```
instead of retyping a long command. Validates every YAML key against `train_bace_clean.py`'s
actual argparse flags at launch time — a typo'd or renamed key is a hard error, not a silent
no-op. Supports `--dry-run` to preview the equivalent CLI command without running it.

Pre-written configs matching the thesis's factors: `baseline.yaml`, `factor_A.yaml`,
`factor_B.yaml`, `factor_BD.yaml`, `factor_BD_multiseed.yaml`, `factor_ABCE.yaml`.

### 3.3 `src/b_ace_py/contracts.py` (`new-code` only)

A safety net asserting, at every environment construction, that Godot's actual reported
observation labels match what the positional-index code
(`enriched_observation_wrapper_ablation.py`, `aggressive_hunter_fixed.py`) assumes. Catches
silent-drift bugs (e.g. if Godot's observation order ever changed during future 2v2/missile-
evasion work) loudly instead of letting code silently read the wrong field into the wrong
computation.

### 3.4 Pinned `requirements.txt` (`new-code` only)

Exact versions confirmed working:
```
gymnasium==1.3.0
pettingzoo==1.27.0
numpy==2.5.2
scipy==1.18.0
torch==2.13.0
stable-baselines3==2.9.0
tensorboard==2.21.0
```
plus optional `wandb` / `streamlit`+`matplotlib`+`pandas` lines, commented out by default.
`mostly-original` has no dependency manifest at all — what's installed in this repo's `.venv`
happens to match, but nothing in-repo declares or enforces it.

### 3.5 Streamlit results dashboard (`new-code` only)

`src/tools/dashboard.py` — reads only existing output files (`monitor.csv`,
`training_results_seed<N>.json`); doesn't change what training/eval compute or write.
```powershell
pip install streamlit matplotlib pandas   # once
streamlit run src/tools/dashboard.py
```
Opens a browser tab to pick and compare completed runs: reward-vs-timestep curves, evaluation
summary with confidence bands, hyperparameter config at a glance.

### 3.6 Weights & Biases integration (`new-code` only)

`--use-wandb --wandb-project bvr-cca-escort` on `train_bace_clean.py` mirrors the same
TensorBoard scalars to a hosted dashboard via `sync_tensorboard=True` — useful for watching a
run remotely (e.g. from a phone) instead of/in addition to local TensorBoard.

### 3.7 Automated tests (`new-code` only)

`tests/test_env_factory_regression.py` + `pytest.ini` — `mostly-original` has zero automated
tests.

### 3.8 Dead code removed (`new-code` only)

~900 lines removed: `ExpertRolloutBuffer`/`ExpertMixPPO` (a full unused alternative PPO
implementation), unused `HVAADestructionPenalty` classes, `RewardBreakdownLogger`,
`RangeClosingShaping`, and others. `train_bace_clean.py` shrinks from ~2,627 lines to ~1,270.
All of this dead code is still present (inert, harmless) on `mostly-original` — real clutter
if reading the file top to bottom, but doesn't affect behavior.

### 3.9 Documentation (`new-code` only)

`HOW_TO_RUN.md` and two `IMPROVEMENT_PLAN_*.md` design docs exist only on `new-code`.
`mostly-original` has no equivalent until this session's `RUNNING_THE_EXPERIMENT.md` and
`PERFORMANCE_DIAGNOSIS_AND_PILOT_PLAN.md`.

### 3.10 What's still missing on *both* branches

`evaluate_experiment_seeds.py` still doesn't compute "Win Rate" directly on either branch —
derive it yourself from HVAA survival + blue loss (see
`PERFORMANCE_DIAGNOSIS_AND_PILOT_PLAN.md` §10 for the exact approach used in this session's
pilot). Neither branch has the thesis's full statistical apparatus (AlgScore, ANOVA, pairwise
significance tables) or an Optuna hyperparameter-search script.

---

## 4. How to run the experiment on each branch

### 4.1 `mostly-original`

```powershell
git checkout mostly-original
.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
python rl_files/train_bace_clean.py --total-timesteps 2000000 --seed 42 --use-enriched-obs --ablation-config engagement_only --experiment-name pilot_B --run-name pilot_B_seed42
```
Full flag-by-flag reference for every factor combination is in `RUNNING_THE_EXPERIMENT.md`
§2-3.

### 4.2 `new-code`

```powershell
git checkout new-code
.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
pip install -r requirements.txt          # once per venv, if not already matching

# Equivalent hand-written command (identical flag names to mostly-original):
python src/rl_files/train_bace_clean.py --total-timesteps 10000000 --use-enriched-obs --ablation-config engagement_only --expert-alignment --experiment-name BD_test

# Or the YAML shortcut for the same run:
python src/rl_files/run_from_config.py config/launch/factor_BD.yaml

# Multi-seed sweep (same flag on either branch):
python src/rl_files/run_experiment.py --experiment-name BD_multiseed --seeds 42 123 456 789 --total-timesteps 10000000 --use-enriched-obs --ablation-config engagement_only --expert-alignment
```

> ⚠️ **The flag *names* are identical on both branches** (same thesis-factor mapping), so a
> copy-pasted command runs without error on the wrong branch and silently produces results
> from the wrong codebase state. Always check `git branch --show-current` before launching a
> new run, especially after switching branches earlier in a session.

---

## 5. Where to find results, and how to analyze them

Both branches write the same directory structure, but under different roots: `runs_sb3/` on
`mostly-original`, `results/runs_sb3/` on `new-code` (this branch's own directory reorganization
moved it; see `docs/SESSION_SUMMARY.md` if you're reading this from `new-code`). Substitute
whichever root matches the branch you're on in every path below:
```
<runs_root>/<name>/<name>_seed<N>/
├── training_results_seed<N>.json     ← config used, per-checkpoint eval history, best/final reward
├── best_checkpoint/best_policy.pt + best_vecnormalize.pkl   ← the usable trained weights
├── monitor.csv, events.out.tfevents.*                        ← tensorboard --logdir <runs_root>/<name>
└── (ppo_bace_blue_lead.zip / best_checkpoint/best_model.zip are 0 bytes on mostly-original
     due to the model.save() bug in §2 — harmless, the .pt files above are authoritative;
     new-code's SkippableTrainPPO fix means these .zip files save correctly there instead)

<runs_root>/<name>/superlative_eval_results.json   ← mission-outcome metrics from evaluate_experiment_seeds.py
```
(`<runs_root>` = `runs_sb3` on `mostly-original`, `results/runs_sb3` on `new-code`.)

**Identical on both branches (besides the `<runs_root>` substitution above):**
- TensorBoard: `tensorboard --logdir <runs_root>/<name>`
- Manual JSON reading:
  ```powershell
  python -c "import json; print(json.dumps(json.load(open('<runs_root>/<name>/superlative_eval_results.json')), indent=2))"
  ```

**`new-code` only:**
- Purpose-built dashboard instead of TensorBoard's generic view or hand-reading JSON:
  ```powershell
  pip install streamlit matplotlib pandas   # once
  streamlit run src/tools/dashboard.py
  ```
- Remote/mobile monitoring via `--use-wandb --wandb-project bvr-cca-escort` on the training
  command.

**Practical recommendation when comparing results across branches**: treat the
**mission-outcome metrics** (HVAA survival, red kill, blue loss) as the comparable numbers —
the raw "Mean Reward" figures will differ because of the evaluation-wrapper-stack change in
§3.1, and that difference isn't a bug or a regression on either side, just a different (more
complete, on `new-code`) accounting of the same reward terms during evaluation.
