# How to Run This Code

Exact, copy-pasteable instructions for running the BVR-CCA-Escort training/evaluation
pipeline. For an explanation of *what* the code does, see [CODEBASE_GUIDE.md](CODEBASE_GUIDE.md).

> **Status on this machine:** setup is already done — Godot binary exported to `bin/`,
> Python virtual environment created at `.venv/`, all dependencies installed, and every
> blocking bug found along the way already fixed (see CODEBASE_GUIDE.md §4.4). If you're on
> **this same machine**, skip straight to [Quick Start](#quick-start). If you're setting up on
> a **different machine**, do [Full Setup From Scratch](#full-setup-from-scratch) first.

---

## Quick Start

Every time you open a new terminal to work on this project, run these two things first:

**PowerShell:**
```powershell
cd C:\Users\kyle2\dev\bvr-cca-escort
.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
```

**Git Bash:**
```bash
cd /c/Users/kyle2/dev/bvr-cca-escort
source .venv/Scripts/activate
export PYTHONUTF8=1
```

> The `PYTHONUTF8` variable is **required**, not optional — without it, the training script
> crashes with `UnicodeEncodeError` partway through the very first run because several files
> print emoji characters and Windows' default console codepage can't display them.

Then run a training job:

```powershell
python src/rl_files/train_bace_clean.py --total-timesteps 10000000 --experiment-name my_first_run
```

That's it — this launches Godot in the background, connects over a local socket, and trains
for 10 million timesteps (several hours; see [Timing](#how-long-will-it-take)). Everything
below explains the options in that command and what else you can do.

---

## Full Setup From Scratch

Only needed if you're on a machine where this hasn't been done yet.

### 1. Install Godot 4.5

Download **Godot Engine 4.5.x** (the standalone/non-.NET build) from the official Godot
website. This project will not open correctly in 4.4 or earlier, or in a 3.x build. It's fine
to use any 4.5.x patch release (4.5.2 was used here).

### 2. Import the project once

```powershell
$GODOT = "C:\path\to\Godot_v4.5.x-stable_win64.exe"
& $GODOT --headless --path "Godot_Air_Combat" --import --quit
```

This generates the local `.godot/` cache. You'll see scanning/import progress messages; a
clean exit with no errors means it worked.

### 3. Install Godot export templates

Download `Godot_v4.5.x-stable_export_templates.tpz` from Godot's official GitHub releases
page (same version as your editor), then extract it so the contents of its `templates/`
folder end up at:

```
%APPDATA%\Godot\export_templates\4.5.x.stable\
```

(Replace `4.5.x` with your exact patch version, e.g. `4.5.2`.) The easiest way to get this
exactly right is the Godot editor itself: open the project, go to **Editor → Manage Export
Templates → Download and Install** — it downloads and places them automatically.

### 4. Add a Windows Desktop export preset

This repo's `Godot_Air_Combat/export_presets.cfg` only ships macOS and Linux presets. If
you're working from a copy of this repo that already has the `[preset.2]` "Windows Desktop"
block (check the file — this machine's copy already has it), skip to step 5. Otherwise, in
the Godot editor: **Project → Export... → Add... → Windows Desktop**, and set the export path
to `../bin/B_ACE_v0.1.exe`.

### 5. Export the binary

```powershell
& $GODOT --headless --path "Godot_Air_Combat" --export-release "Windows Desktop" "C:\path\to\repo\bin\B_ACE_v0.1.exe"
```

Confirm it worked:
```powershell
Test-Path "bin\B_ACE_v0.1.exe"   # should print True
```
You should now have `bin\B_ACE_v0.1.exe` (~96 MB) and `bin\B_ACE_v0.1.pck`.

### 6. Set up Python

Requires **Python 3.10+** (3.13 confirmed working).

```powershell
cd C:\path\to\repo
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Do **not** additionally `pip install godot-rl-agents` — this repo has its own socket client
and doesn't use that package. `requirements.txt` pins exact versions confirmed working on this
project; the optional `wandb` / `streamlit`+`matplotlib`+`pandas` lines in it are commented out
by default — uncomment them only if you're using the Weights & Biases logging or results
dashboard features.

You're now at the same state as [Quick Start](#quick-start).

---

## Watching It in the Godot Engine (Visual Playback)

Training normally runs fully headless (no window, sped up 200x) for throughput. But you can
watch the aircraft actually fly in a real Godot window — confirmed working — using a separate
config file: `src/b_ace_py/Watch_B_ACE_config.json`. It's identical to the default scenario config
except two settings: `"renderize": 1` (opens a real window instead of headless) and
`"speed_up": 1` (real-time, so it's actually watchable instead of a blur), plus a different
port (12600) so it won't collide with a training run already using the default port.

```powershell
python src/rl_files/train_bace_clean.py `
  --config src/b_ace_py/Watch_B_ACE_config.json `
  --total-timesteps 2000 --n-steps 512 --batch-size 128 --epochs 1 `
  --run-name watch_session
```

What happens: a real Godot window opens showing the 3D battlespace (blue CCA, red adversary,
HVAA, radar cone). It's driven by whatever policy is currently training (random weights at
first, since this starts a fresh model) — you're watching the actual simulation the RL agent is
learning in, not a replay. Runs at roughly real-time speed (~10 physics steps/sec observed).

While the window is open, you can use Godot's own in-game controls:
- **Camera:** WASD/arrow keys to pan, `F`/`C` for camera presets, `N` to cycle between aircraft
  cockpit views.
- **Pause:** the pause button in the HUD.
- **Speed:** the speed-up button cycles through 1x/10x/50x/200x/1000x if you want to fast-forward
  once you're done watching closely.

**To stop:** close the window, or let it run to `--total-timesteps`. If you close the window
partway through, the underlying Python process and Godot subprocess can be left running in the
background — check for lingering processes afterward:
```powershell
Get-Process B_ACE_v0.1, python -ErrorAction SilentlyContinue
```
and `Stop-Process -Id <pid> -Force` any that are still there from a session you closed early.

**Watching a specific trained checkpoint instead of a fresh model:** use
`evaluate_experiment_seeds.py`, which has dedicated flags for this —
`--renderize 1 --speed-up 1` — pointed at a completed run's directory:
```powershell
python src/rl_files/evaluate_experiment_seeds.py `
  --experiment-dir results/runs_sb3/<your_experiment> --episodes 5 `
  --renderize 1 --speed-up 1
```
This is the more useful mode once you have an actual trained model — it shows you what the
*learned* policy does, not a randomly-initialized one.

Don't use `--renderize 1` for a real multi-hour training run — headless + `speed_up=200` is
what makes training fast; rendering in real time would make a 10M-timestep run take on the
order of weeks instead of hours.

---

## Running Training

All commands below assume you've done the [Quick Start](#quick-start) two-liner (activated
venv + set `PYTHONUTF8`) and are in the repo root.

### A single training run

```powershell
python src/rl_files/train_bace_clean.py `
  --total-timesteps 10000000 `
  --experiment-name my_run
```

(In Git Bash, replace the backtick line-continuations with `\`.)

This uses the default scenario config (`src/b_ace_py/SimpleExample_B_ACE_config.json`) and the
"baseline" observation space (no heuristic features), with a single fixed seed (default 42).
Output goes to `results/runs_sb3/my_run/`.

### Turning on the experimental factors

The thesis studies five on/off factors. Here's exactly which flags map to which:

| Factor | What it is | Flag(s) |
|---|---|---|
| A | ATDDG observation features | `--use-enriched-obs --ablation-config geometry_only` |
| B | WEZ/DMC observation features | `--use-enriched-obs --ablation-config engagement_only` |
| C | RLPE observation features | `--use-enriched-obs --ablation-config range_limited_only` |
| A+B+C combos | any mix of the three above | `geometry_engagement`, `geometry_range`, `engagement_range`, or `all` |
| D | Alignment-based reward shaping | `--expert-alignment` |
| E | Action blending | `--expert-blending` |

**Important:** "Baseline" (all factors off) means leaving off `--use-enriched-obs` entirely —
do not pass `--ablation-config none`, which is a different, non-equivalent code path (see
CODEBASE_GUIDE.md §4.3 for why).

Example — Factors B and D together (this is design run "BD" in the thesis's naming):
```powershell
python src/rl_files/train_bace_clean.py `
  --total-timesteps 10000000 `
  --use-enriched-obs --ablation-config engagement_only `
  --expert-alignment `
  --experiment-name BD_test
```

Example — all three observation factors plus action blending ("ABCE"):
```powershell
python src/rl_files/train_bace_clean.py `
  --total-timesteps 10000000 `
  --use-enriched-obs --ablation-config all `
  --expert-blending `
  --experiment-name ABCE_test
```

Don't combine `--expert-alignment` and `--expert-blending` in the same run — they're
different techniques (Factors D and E) and the script will print a warning if you try.

### Multiple seeds of the same configuration

```powershell
python src/rl_files/run_experiment.py `
  --experiment-name BD_multiseed `
  --seeds 42 123 456 789 `
  --total-timesteps 10000000 `
  --use-enriched-obs --ablation-config engagement_only --expert-alignment
```

This runs `train_bace_clean.py` once per seed, one after another, saving each under
`results/runs_sb3/BD_multiseed/BD_multiseed_seed<N>_<timestamp>/`. **You must pass
`--total-timesteps` here too** — `run_experiment.py` has no default of its own, and silently
falls back to `train_bace_clean.py`'s default of 1,000,000, which is far too short.

### Launching a saved run configuration from a YAML file

Instead of retyping the flag combinations above every time, save them once as a YAML file under
`config/launch/` and launch with:

```powershell
python src/rl_files/run_from_config.py config/launch/factor_BD.yaml
```

Each YAML key must equal the `dest` name of a real `train_bace_clean.py` flag (e.g.
`--total-timesteps` -> `total_timesteps`, `--use-enriched-obs` -> `use_enriched_obs`) — an
unrecognized key errors out immediately rather than silently doing nothing. Add a `seeds: [42,
123, ...]` list (instead of a single `seed:`) to delegate to `run_experiment.py` for a
multi-seed sweep. See `config/launch/baseline.yaml`, `config/launch/factor_A.yaml`, `config/launch/factor_B.yaml`,
`config/launch/factor_BD.yaml`, `config/launch/factor_ABCE.yaml`, and `config/launch/factor_BD_multiseed.yaml` for
worked examples matching the factor table above. Pass `--dry-run` to print the constructed
`train_bace_clean.py`/`run_experiment.py` command without running it.

### A quick smoke test (confirm everything still works, ~2 minutes)

Before committing to a multi-hour run, it's worth confirming the pipeline still connects:

```powershell
python src/rl_files/train_bace_clean.py `
  --total-timesteps 512 --n-steps 256 --batch-size 64 --epochs 1 `
  --run-name smoketest
```

You should see Godot launch, a printed observation-label table, several full episodes run,
and a "TRAINING COMPLETE!" summary with no traceback. Delete `results/runs_sb3/smoketest/` afterward
— it's just a connectivity check, not a real training result.

---

## Running Evaluation

Once a run (or a multi-seed sweep) has finished:

```powershell
python src/rl_files/evaluate_experiment_seeds.py `
  --experiment-dir results/runs_sb3/BD_multiseed `
  --episodes 200
```

If you trained with a non-default `--ablation-config`, pass the **same** value here too — the
evaluation script does not reliably auto-detect which observation features a checkpoint was
trained with (see CODEBASE_GUIDE.md §5.3). E.g., if you trained with
`--ablation-config engagement_only`, evaluate with `--ablation-config engagement_only` as well.

This computes per-episode outcomes (including win rate) and cross-seed 95% confidence intervals
on all of them. It also automatically generates a cross-seed learning-curve figure
(`<experiment-dir>/learning_curve_analysis.png`) with a Peak-Return/AUC/AlgScore summary — pass
`--skip-figures` to skip this (e.g. if matplotlib isn't installed). See
`docs/THESIS_REPRODUCTION_GUIDE.md` §6 for exactly what that figure measures and its provenance
caveat. Still **not** produced: the thesis's main-effects/ANOVA tables or pairwise-vs-baseline
significance tests — see CODEBASE_GUIDE.md §5.6.

---

## Watching Training Progress Live

While a run is in progress, in a **second** terminal (with the same venv activated):

```powershell
tensorboard --logdir results/runs_sb3/my_run
```

Then open the URL it prints (usually `http://localhost:6006`) in a browser.

### Optional: mirror to Weights & Biases

If you'd rather watch a run from a hosted dashboard (e.g. on your phone, or to compare runs
across machines) instead of/in addition to TensorBoard, pass `--use-wandb` to
`train_bace_clean.py` (or `run_experiment.py`, which forwards it through):

```powershell
pip install wandb   # only needed once, if not already installed
wandb login          # only needed once
python src/rl_files/train_bace_clean.py --use-wandb --wandb-project bvr-cca-escort ...
```

This mirrors the exact same TensorBoard scalars to a W&B run (via `sync_tensorboard=True`) — it
doesn't log anything extra or change training. Omitting `--use-wandb` (the default) behaves
exactly as before and never imports `wandb` at all.

### Optional: a local results-browsing dashboard (no cloud account needed)

For a purpose-built view of this project's runs (reward-vs-timestep curves, evaluation summary
with confidence band, hyperparameter config at a glance) without needing TensorBoard's more
generic training-internals view or a W&B account:

```powershell
pip install streamlit matplotlib pandas   # only needed once, if not already installed
streamlit run src/tools/dashboard.py
```

This opens a browser tab where you can pick one or more completed runs from `results/runs_sb3/` to
compare. It only reads existing output files (`monitor.csv`, `training_results_seed<N>.json`)
and never modifies or affects training in any way.

---

## How Long Will It Take?

- The thesis trained each configuration for ~10,000,000 timesteps. On the reference hardware
  (Apple M3 MacBook, CPU-only, `speed_up=200`), a full run took multiple hours. Expect similar
  or somewhat different timing on this machine (CPU-only PyTorch here too).
- A periodic evaluation runs every 200,000 timesteps (15 episodes each) — you'll see progress
  in the console and in TensorBoard well before the run finishes.
- The 512-timestep smoke test above takes about 1-2 minutes and is the fastest way to confirm
  nothing is broken before starting a long run.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError: ... Tried .../bin/B_ACE_v0.1.exe ... but neither exists` | The Godot binary hasn't been exported. Do [Full Setup From Scratch](#full-setup-from-scratch) steps 3-5. |
| `UnicodeEncodeError` mid-run | You forgot `$env:PYTHONUTF8 = "1"` (PowerShell) / `export PYTHONUTF8=1` (Bash) this session. Set it and re-run. |
| `waiting for remote GODOT connection on port ...` hangs forever | Another Godot process (from a previous crashed run) may still be holding the port. Check Task Manager for lingering `B_ACE_v0.1.exe` processes and end them, then retry. |
| Training runs but nothing seems to improve | Expected for short runs — meaningful learning in this environment takes millions of timesteps, not thousands. Use TensorBoard to check `eval/mean_reward` over a long run, not a smoke test. |
| `ModuleNotFoundError` for any package | Confirm the venv is activated (`(.venv)` should be showing in your prompt) — `pip install` the missing package if it's genuinely not in the list above. |
| You want to change what's rewarded/penalized | The reward function is spread across `Godot_Air_Combat/assets/Sim_assets.gd` (Godot side) and ~10 wrapper classes chained together in `src/rl_files/env_factory.py`'s `build_env()` function (Python side) — there is no single file to edit. See CODEBASE_GUIDE.md §1 and §5.1. |
| You want to launch a saved config, mirror to W&B, or use the results dashboard | See [Launching a saved run configuration from a YAML file](#launching-a-saved-run-configuration-from-a-yaml-file) / [Optional: mirror to Weights & Biases](#optional-mirror-to-weights--biases) / [Optional: a local results-browsing dashboard](#optional-a-local-results-browsing-dashboard-no-cloud-account-needed) above. |

For anything not covered here — what the code does, why a specific number looks off, how to
extend it for missile evasion or 2v2 — see [CODEBASE_GUIDE.md](CODEBASE_GUIDE.md).
