# Reproducing the Thesis Pipeline, Beginning to End

A concise guide to running the experiment the way Capt. Bieger's thesis describes, and to what
is/isn't actually reproducible from this repository. Read this before assuming a described
capability exists in code — two important pieces (marked below) were **never checked into
this repo** and would need to be rebuilt or done manually.

---

## 1. What's actually here vs. what isn't

| Capability | In this repo? |
|---|---|
| Training / evaluation pipeline, reward shaping, all 5 ablation factors | **Yes** — fully working (§2-3) |
| Watching the agent live in a real Godot window, at any checkpoint | **Yes** (§4) |
| Saving that as an actual video file | **No** — see §4 caveat |
| Optuna hyperparameter search | **No** — only the winning values remain, hardcoded (§5) |
| A multi-seed "AlgScore / 95% CI" learning-curve figure (like the one that motivated this doc) | **Yes, now** — added this session, see §6. Not a verified reproduction of Bieger's original tool (that one's still not in this repo) — a fresh implementation of the same idea. |

The Optuna and video gaps aren't bugs — they're confirmed by grepping the entire repo history on
every branch: zero hits for `optuna` anywhere. Whatever tooling Bieger used for hyperparameter
search lived outside this checkout, and no automatic video-file output was ever implemented (see
§4).

---

## 2. Running the full factorial experiment as originally intended

The thesis's 5 factors map onto CLI flags on `train_bace_clean.py` like this:

| Factor | What it is | Flag(s) |
|---|---|---|
| A | ATDDG geometry features | `--use-enriched-obs --ablation-config geometry_only` |
| B | WEZ/DMC engagement features | `--use-enriched-obs --ablation-config engagement_only` |
| C | Range-limited (RLPE) features | `--use-enriched-obs --ablation-config range_limited_only` |
| A+B+C combined | all theoretical obs. features | `--use-enriched-obs --ablation-config all` |
| D | Alignment-based reward shaping | `--expert-alignment` |
| E | Action blending with expert policy | `--expert-blending` |
| Baseline | no heuristics at all | omit `--use-enriched-obs` entirely (**not** `--ablation-config none` — that still runs the enrichment wrapper with zero features, a different code path) |

Factors combine by combining flags (e.g. "ACD" = `--use-enriched-obs --ablation-config all
--expert-alignment`). The thesis's full design has 16-18 named configurations; none of their
exact flag combinations are saved anywhere in this repo, so reconstruct each one from this table.

**Single seed:**
```powershell
python src\rl_files\train_bace_clean.py `
  --total-timesteps 10000000 --use-enriched-obs --ablation-config engagement_only `
  --expert-alignment --experiment-name BD_test
```

**Multi-seed (the thesis uses several seeds per design point):**
```powershell
python src\rl_files\run_experiment.py `
  --experiment-name BD_multiseed --seeds 42 123 456 789 1011 `
  --total-timesteps 10000000 --use-enriched-obs --ablation-config engagement_only --expert-alignment
```
`run_experiment.py` has **no default `--total-timesteps` of its own** — it silently inherits
`train_bace_clean.py`'s default of 1,000,000 if you omit it. Always pass it explicitly.

Output lands under `results\runs_sb3\<experiment-name>\<experiment-name>_seed<N>_<timestamp>\`.

---

## 3. Evaluating an experiment that has already finished

```powershell
python src\rl_files\evaluate_experiment_seeds.py `
  --experiment-dir results\runs_sb3\BD_multiseed --episodes 100
```
- `--experiment-dir` points at the folder containing the seed subfolders (or a single run folder).
- `--episodes` — the thesis-style headline numbers in this project used 100-200.
- If the run used a non-default reward mode, obs mode, or ablation config, **pass the matching
  flags** (`--reward-mode`, `--use-enriched-obs --ablation-config ...`) — none of that is saved
  in a way the eval script reads back automatically, so it will silently reconstruct the wrong
  features/reward for that checkpoint otherwise.
- By default this evaluates the **best (superlative) checkpoint** and writes
  `results\runs_sb3\BD_multiseed\superlative_eval_results.json` (mission-outcome metrics: win
  rate, HVAA survival, red kill rate, blue loss rate, mean reward, 95% CI across seeds).

**Evaluate a specific checkpoint instead of the best one** (see §4 for why you'd want this):
```powershell
# First, see what checkpoints were actually saved for each seed:
python src\rl_files\evaluate_experiment_seeds.py `
  --experiment-dir results\runs_sb3\BD_multiseed --list-timesteps

# Then evaluate one or more specific timesteps:
python src\rl_files\evaluate_experiment_seeds.py `
  --experiment-dir results\runs_sb3\BD_multiseed --episodes 20 `
  --timesteps 800000,4800000,9400000 `
  --output-file results\runs_sb3\BD_multiseed\eval_by_stage.json
```
(`--output-file` keeps this from overwriting the main `superlative_eval_results.json`.)
Checkpoints are only saved at whatever timesteps `eval_checkpoints/` actually contains for that
run — `--list-timesteps` tells you the real numbers instead of guessing.

---

## 4. Watching agent behavior at different training stages ("crawl / walk / run")

There's no code anywhere in this repo that writes a video file — `evaluate_experiment_seeds.py`
can only open a **real, live Godot window** at real-time speed so a human can watch it:

```powershell
python src\rl_files\evaluate_experiment_seeds.py `
  --experiment-dir results\runs_sb3\BD_multiseed --episodes 3 `
  --timesteps 800000 `
  --renderize 1 --speed-up 1
```
Run this once per checkpoint (an early one, a middle one, the final/best one) to see the
early/mid/late behavior progression. **To actually get a video out of this**, you record your
own screen while it plays (OBS Studio, Xbox Game Bar `Win+Alt+R`, or any screen recorder) — that
external step is almost certainly what Bieger did; it isn't automatable from anything in this
codebase today.

`--speed-up 200` (the default) is what training/normal evaluation uses — always set it to `1`
for anything you intend to watch or record, or it'll be an unwatchable blur.

---

## 5. Hyperparameter tuning (Optuna) — not reproducible as-is

No Optuna script exists anywhere in this repo, on any branch. The thesis describes a two-stage
search (a broad "Search One" of 150 trials, then a narrower "Search Two" of 70 trials), but only
its **winning result** survived into this codebase, hardcoded as `train_bace_clean.py`'s argparse
defaults:

| Hyperparameter | Value | Flag |
|---|---|---|
| Initial / final learning rate | 0.000258 → 0.00001 (linear decay) | `--initial-learning-rate` / `--final-learning-rate` |
| Initial / final entropy coefficient | 0.051 → 0.005 (linear decay) | `--initial-entropy-coef` / `--final-entropy-coef` |
| Discount factor (γ) | 0.99 | `--gamma` |
| GAE λ | 0.920664477208517 | `--gae-lambda` |
| PPO clip range | 0.197991384633053 | `--clip-eps` |
| Value-function coefficient | 0.7728139 | `--vf-coef` |
| Rollout length | 6144 | `--n-steps` |
| Minibatch size | 1024 | `--batch-size` |
| PPO epochs per update | 8 | `--epochs` |
| Network shape | 2 layers × 512 units | `--n_layers` / `--layer_size` |

(The high-precision decimals on GAE λ and clip range are themselves evidence these came straight
out of an Optuna trial rather than being hand-picked.) If you need to actually re-run the search,
you're writing an Optuna study yourself around `train_bace_clean.py`'s existing CLI flags —
`optuna.create_study` + `study.optimize(objective, n_trials=150)`, where `objective` launches a
short training run per trial and returns its best evaluation reward. Nothing else to reuse exists
in-repo.

---

## 6. The AlgScore / per-seed confidence-interval figure

**Implemented this session** (`src/rl_files/experiment_statistics.py`), and wired into
`evaluate_experiment_seeds.py`'s default (superlative) mode — it runs automatically every time
you evaluate an experiment, no extra flag needed:

```powershell
python src\rl_files\evaluate_experiment_seeds.py `
  --experiment-dir results\runs_sb3\BD_multiseed --episodes 100
```
This now also writes `results\runs_sb3\BD_multiseed\learning_curve_analysis.png` — every seed's
`eval/mean_reward` curve, a black cross-seed mean line, a shaded 95% CI band, and a
Peak-Return/AUC/AlgScore summary in the title — plus the same numbers under a
`"learning_curve_analysis"` key in `superlative_eval_results.json`. Pass `--skip-figures` to
disable it (e.g. running without matplotlib installed).

**Provenance caveat, unchanged from before**: no code implementing this ever existed anywhere in
this repo's history, on any branch — the formula is only known from a reference chart's title
text, not from recovered source. This implementation is a same-name reconstruction, not a
verified reproduction of Bieger's original tool:
```
AlgScore = 0.6 * Peak_LCB + 0.4 * AUC_LCB
```
`Peak_LCB`/`AUC_LCB` are read as the lower bound of a 95% cross-seed CI (t-distribution, the same
`stats_utils.t_distribution_ci` `evaluate_experiment_seeds.py` uses elsewhere) on, respectively,
each seed's peak `eval/mean_reward` and a **time-averaged** AUC (area under the curve divided by
step range, so it's on the same reward scale as peak return — a raw unnormalized area would mix
incompatible units with a 0.6/0.4 weighted sum). Seeds are aligned onto a common step grid via
linear interpolation before averaging, since they don't necessarily log at identical steps. If
you have the actual original formula, tell me and I'll match it exactly instead.

---

## See also

- `docs/HOW_TO_RUN.md` — full command reference, troubleshooting, TensorBoard/W&B/dashboard setup.
- `docs/CODEBASE_GUIDE.md` §5.6 — the full list of thesis statistical machinery absent from this
  repo (ANOVA tables, main-effects tables, pairwise significance tests, in addition to AlgScore).
- `docs/SESSION_SUMMARY.md` — the env build/step/reset lifecycle and this project's own
  reward-redesign commands.
