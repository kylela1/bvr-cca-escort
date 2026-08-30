# Running the BVR-CCA-Escort Pilot Experiment

Practical, copy-pasteable reference for running training, changing parameters, and
visualizing results on this machine. Commands are given in **PowerShell** (this machine's
shell); a Git Bash equivalent is noted wherever the syntax differs.

For *why* things are set up this way — the bugs that were found and fixed, the reasoning
behind the pilot's scope, and a full analysis of its results — see
[PERFORMANCE_DIAGNOSIS_AND_PILOT_PLAN.md](PERFORMANCE_DIAGNOSIS_AND_PILOT_PLAN.md). For a
file-by-file map of the codebase, see [CODEBASE_GUIDE.md](CODEBASE_GUIDE.md). This file only
covers commands.

All commands assume you're on the **`mostly-original`** branch (`git branch --show-current`)
— the crash-bug fixes this experiment depends on live there, not on `main`'s other siblings.

---

## 1. Setup (every new terminal session)

```powershell
cd C:\Users\kyle2\dev\bvr-cca-escort
.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
```

- **`$env:PYTHONUTF8 = "1"` is required, not optional** — several files print non-ASCII
  characters (emoji, arrows) that crash on Windows' default console codepage otherwise.
- If `Activate.ps1` fails with *"running scripts is disabled on this system"*, run this once
  (per user, not per session) and retry: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
- If you ever see `ModuleNotFoundError: No module named 'stable_baselines3'`, the venv isn't
  activated — either re-run `Activate.ps1`, or skip activation entirely and call
  `.venv\Scripts\python.exe` explicitly instead of `python` in every command below.
- Before starting a new run, check for lingering processes from a previous crashed/interrupted
  one (a real, confirmed cause of slowdowns — see the diagnosis doc §3.4):
  ```powershell
  Get-Process B_ACE_v0.1, python -ErrorAction SilentlyContinue
  ```
  Kill any you find with `Stop-Process -Id <pid> -Force` before proceeding.

**Git Bash equivalent:**
```bash
cd /c/Users/kyle2/dev/bvr-cca-escort
source .venv/Scripts/activate
export PYTHONUTF8=1
```

---

## 2. Running the experiment

Run one config at a time — **not in parallel** (concurrent Godot instances compete for the
same 4 physical CPU cores and slow everything down). Each takes roughly 90-100 minutes at
2,000,000 timesteps on this hardware, including all periodic and final evaluation.

### The 5 pilot configurations

```powershell
# baseline (no heuristic features)
python rl_files/train_bace_clean.py --total-timesteps 2000000 --seed 42 --experiment-name pilot_baseline --run-name pilot_baseline_seed42

# Factor B: WEZ/DMC observation features
python rl_files/train_bace_clean.py --total-timesteps 2000000 --seed 42 --use-enriched-obs --ablation-config engagement_only --experiment-name pilot_B --run-name pilot_B_seed42

# Factor D: alignment-based reward shaping
python rl_files/train_bace_clean.py --total-timesteps 2000000 --seed 42 --expert-alignment --alignment-decay-start 100000 --alignment-decay-end 400000 --experiment-name pilot_D --run-name pilot_D_seed42

# Factor E: action blending
python rl_files/train_bace_clean.py --total-timesteps 2000000 --seed 42 --expert-blending --blend-decay-start 100000 --blend-decay-end 1000000 --experiment-name pilot_E --run-name pilot_E_seed42

# ACD: Factors A+C+D combined (thesis's top-ranked design)
python rl_files/train_bace_clean.py --total-timesteps 2000000 --seed 42 --use-enriched-obs --ablation-config geometry_range --expert-alignment --alignment-decay-start 100000 --alignment-decay-end 400000 --experiment-name pilot_ACD --run-name pilot_ACD_seed42
```

> Each is one long line here on purpose — the safest way to paste a multi-flag command into
> PowerShell without a continuation-character mistake. To split one across lines yourself, end
> each line with a backtick `` ` `` (last character on the line, no trailing space) — not a
> Bash-style `\`, which PowerShell doesn't treat as a continuation at all.

### Evaluating a completed run (mission-outcome metrics)

```powershell
python rl_files/evaluate_experiment_seeds.py --experiment-dir runs_sb3/pilot_baseline --episodes 20
python rl_files/evaluate_experiment_seeds.py --experiment-dir runs_sb3/pilot_B --episodes 20 --use-enriched-obs --ablation-config engagement_only
python rl_files/evaluate_experiment_seeds.py --experiment-dir runs_sb3/pilot_D --episodes 20
python rl_files/evaluate_experiment_seeds.py --experiment-dir runs_sb3/pilot_E --episodes 20
python rl_files/evaluate_experiment_seeds.py --experiment-dir runs_sb3/pilot_ACD --episodes 20 --use-enriched-obs --ablation-config geometry_range
```

**Enriched-observation configs (B, ACD, or any A/C variant) need the matching
`--use-enriched-obs --ablation-config ...` flags at evaluation time too** — the script does
not reliably auto-detect which features a checkpoint was trained with.

`--episodes` defaults to 100 if omitted; 20 keeps each pass to ~15-25 minutes instead of
~1.5-2 hours. Raise it for tighter confidence intervals at the cost of time.

### Quick sanity check before a long run (~1-2 minutes)

```powershell
python rl_files/train_bace_clean.py --total-timesteps 3000 --n-steps 1024 --batch-size 256 --epochs 2 --run-name smoketest
```
Confirms Godot launches and connects without committing to a 90-minute run. Expect a ~5-10
minute silent pause after training finishes — that's the unconditional 20-episode final
evaluation, not a hang. Delete `runs_sb3/smoketest/` afterward.

### Multiple seeds of one configuration

```powershell
python rl_files/run_experiment.py --experiment-name pilot_ACD_multiseed --seeds 42 123 456 789 --total-timesteps 2000000 --use-enriched-obs --ablation-config geometry_range --expert-alignment --alignment-decay-start 100000 --alignment-decay-end 400000
```
**Always pass `--total-timesteps` here explicitly** — `run_experiment.py` has no default of
its own and silently falls back to `train_bace_clean.py`'s 1,000,000 default otherwise.

---

## 3. Changing parameters

### 3.1 Which thesis factor(s) to enable

| Factor | Meaning | Flag(s) |
|---|---|---|
| A | ATDDG observation features | `--use-enriched-obs --ablation-config geometry_only` |
| B | WEZ/DMC observation features | `--use-enriched-obs --ablation-config engagement_only` |
| C | RLPE observation features | `--use-enriched-obs --ablation-config range_limited_only` |
| A+B | | `--ablation-config geometry_engagement` |
| A+C | | `--ablation-config geometry_range` |
| B+C | | `--ablation-config engagement_range` |
| A+B+C | | `--ablation-config all` |
| D | Alignment-based reward shaping | `--expert-alignment` (+ `--alignment-coef`, `--alignment-decay-start`, `--alignment-decay-end`) |
| E | Action blending | `--expert-blending` (+ `--blend-initial-alpha`, `--blend-final-alpha`, `--blend-decay-start`, `--blend-decay-end`, `--blend-decay-type`, `--blend-fire-mode`) |

**"Baseline" = omit `--use-enriched-obs` entirely.** Don't pass `--ablation-config none` as a
substitute — that's a different code path (still runs the enrichment wrapper machinery, just
with zero features added). Don't combine `--expert-alignment` and `--expert-blending` in the
same run — the script warns if you try.

### 3.2 Run length, identity, and seeds

| Flag | Meaning | Thesis value | Pilot's value |
|---|---|---|---|
| `--total-timesteps` | Total PPO training steps | 10,000,000 | 2,000,000 |
| `--seed` | Random seed | 4 seeds (42, 123, 456, 789) | 1 seed (42) |
| `--experiment-name` | Groups runs under `runs_sb3/<name>/` | — | `pilot_<config>` |
| `--run-name` | Names the specific run folder | — | `pilot_<config>_seed<N>` |

**If you change `--total-timesteps` away from 2,000,000 (or 10,000,000), proportionally
rescale any decay-window flags you use** (`--alignment-decay-start/-end`,
`--blend-decay-start/-end`) by the same ratio — e.g. at 2,000,000 (0.2x of the thesis's 10M)
this pilot used `--alignment-decay-end 400000` (0.2 x 2,000,000). Otherwise the curriculum
freezes partway through a schedule calibrated for a different horizon. Note the
checkpoint-rollback safety net's eligibility threshold
(`min_timesteps_before_rollback=4,000,000`) is hardcoded in `rl_files/train_bace_clean.py`
and not CLI-exposed — it simply won't activate below that many total steps.

### 3.3 Algorithm / network hyperparameters

Rarely need changing — the defaults are the thesis's own tuned values.

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
| `--device` | `cpu` | or `cuda`, if a GPU is ever available |

### 3.4 Scenario / environment settings

Spawn randomization, reward weights, battlespace size, missile counts, and episode length
(`max_cycles`) are **not** CLI flags — they live in the JSON file passed via `--config`
(default: `b_ace_py/SimpleExample_B_ACE_config.json`). Edit that file directly, or pass
`--config path\to\your_copy.json` to use an alternate one.

---

## 4. Visualizing training

### 4.1 TensorBoard (learning curves, live or after the fact)

```powershell
tensorboard --logdir runs_sb3/pilot_baseline
```
Then open the printed URL (usually `http://localhost:6006`). Point `--logdir` at `runs_sb3`
itself instead of one run's folder to overlay and compare multiple runs on the same charts —
useful for comparing all 5 pilot configs at once. Works while a run is still in progress.

### 4.2 Watching a trained agent fly (real Godot window, live)

`evaluate_experiment_seeds.py` supports rendering directly — no separate config file needed:

```powershell
python rl_files/evaluate_experiment_seeds.py --experiment-dir runs_sb3/pilot_ACD --episodes 5 --use-enriched-obs --ablation-config geometry_range --renderize 1 --speed-up 1
```

This opens an actual 3D window: blue CCA, red adversary, HVAA, and radar cone, with
`pilot_ACD`'s best checkpoint flying the CCA. Swap `--experiment-dir`/ablation flags for any
other pilot run (`pilot_baseline` and `pilot_D`/`pilot_E` need no ablation flags at all).
`--speed-up 1` plays at real time (a full episode can take several minutes depending on
outcome); raise it (e.g. `--speed-up 5` or `10`) to see more episodes faster while still
watchable, or leave `--renderize 0` (the default) for full 200x-speed evaluation with no
window at all.

**This is a live interactive window, not a saved file.** Nothing in this codebase exports
video automatically. To keep or share a recording, screen-record the window yourself while it
plays — Windows' built-in Xbox Game Bar (`Win+G`) or OBS Studio both work fine for this; it
isn't something that can be generated by a script here.

### 4.3 Reading result files directly (no visualization tooling needed)

```powershell
python -c "import json; print(json.dumps(json.load(open('runs_sb3/pilot_ACD/superlative_eval_results.json')), indent=2))"
```
Mission-outcome metrics (HVAA survival %, red kill %, blue loss %, mean reward, 95% CI) from
the last `evaluate_experiment_seeds.py` run against that config. The equivalent
`training_results_seed<N>.json` inside each run's own folder holds the full per-checkpoint
training-time evaluation history instead.

### 4.4 Where everything lives, for reference

```
runs_sb3/pilot_<name>/pilot_<name>_seed<N>/
├── training_results_seed<N>.json     ← config used, per-checkpoint eval history, best/final reward
├── best_checkpoint/best_policy.pt + best_vecnormalize.pkl   ← the usable trained weights
├── monitor.csv, events.out.tfevents.*                        ← tensorboard --logdir runs_sb3/pilot_<name>
└── (ppo_bace_blue_lead.zip / best_checkpoint/best_model.zip are always 0 bytes — known
     pickling bug, harmless; the .pt files above are the real, working checkpoints)

runs_sb3/pilot_<name>/superlative_eval_results.json   ← mission-outcome metrics from evaluate_experiment_seeds.py
```
