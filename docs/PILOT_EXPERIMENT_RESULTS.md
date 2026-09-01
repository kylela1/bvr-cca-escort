# Pilot Experiment Results (Reference, Run on `mostly-original`)

This document carries forward the results and operational findings from a 5-configuration
pilot experiment run on the `mostly-original` branch, before switching to `new-code` to begin
the reward-redesign project. It's included here as a **reference baseline and set of
operational lessons**, not as a claim that these exact numbers apply unchanged to `new-code`
— see the caveat in §3 before treating them as your project's official baseline.

Full original write-up (bug root-causes, exact diagnosis process, complete command reference):
`PERFORMANCE_DIAGNOSIS_AND_PILOT_PLAN.md` and `RUNNING_THE_EXPERIMENT.md` on the
`mostly-original` branch (`git show mostly-original:PERFORMANCE_DIAGNOSIS_AND_PILOT_PLAN.md`
to read without switching branches).

---

## 1. Operational lessons that still apply here

- **A 4-core/8-thread CPU-only laptop is not actually the bottleneck.** Raw environment
  stepping measured ~1,365 it/s; full PPO training measured ~125-210 fps. Both are healthy.
- **Evaluation episodes are expensive and easy to underestimate.** Episodes in this
  environment routinely run 6,000-13,500+ steps before terminating. The periodic evaluation
  (every 200,000 steps, 15 episodes) and the final evaluation (20 episodes) both use this same
  uncapped episode length, so their wall-clock cost is substantial and does not scale down
  automatically for short/smoke-test runs. If you shorten `--total-timesteps` for faster
  iteration while designing your new reward, expect evaluation phases to still take several
  minutes each — budget for that when planning a session.
- **Check for lingering `B_ACE_v0.1.exe`/`python.exe` processes after any crashed or
  interrupted run** (`Get-Process B_ACE_v0.1, python -ErrorAction SilentlyContinue` in
  PowerShell). A stale process from a previous session was found still running 10 days later
  during this pilot, silently competing for CPU with new runs.
- **`$env:PYTHONUTF8 = "1"` is required every session** — several files print non-ASCII
  characters that crash on Windows' default console codepage otherwise.

## 2. Pilot results (measured on `mostly-original`, after its crash-bug fixes)

5 configurations x 1 seed (42) x 2,000,000 timesteps each (0.2x the thesis's 10,000,000):

| Config | Final PPO reward | Best ckpt reward | HVAA Survival | Red Kill | Blue Loss | Approx. win rate\* |
|---|---|---|---|---|---|---|
| baseline (no heuristics) | -38.85 | -21.37 | 0% | 0% | 0% | ~0% |
| B (WEZ/DMC obs.) | -35.10 | -15.62 | 0% | 0% | 15% | ~0% |
| D (alignment reward) | -34.40 | -14.75 | **50%** | 25% | 45% | **~50%** |
| E (action blending) | -39.93 | -31.71 | 5% | 0% | 20% | ~5% |
| ACD (A+C+D combined) | **-7.52** | **+3.56** | 25% | **45%** | **0%** | ~25% |

\*Approximated as HVAA survival rate (20 evaluation episodes each) — episodes where the CCA is
destroyed appear to terminate before mission completion in every case observed, making blue
loss and HVAA survival effectively disjoint outcomes.

**Findings, all consistent with the thesis at this reduced scale:**
- Action blending (E) underperforms even the no-heuristics baseline on every metric — the
  cleanest, least ambiguous result.
- Individual factors (B, D) each modestly improve over baseline, mainly by inducing more
  decisive combat (real kills/losses instead of every episode timing out uniformly).
- The combined configuration (ACD) compounds those gains dramatically on raw reward and
  lethality/survivability, but does *not* have the highest win-rate proxy — D does. This
  independently reproduces the thesis's own Section 4.7.1 finding that training reward
  correlates imperfectly with actual mission-level win rate.

## 3. Important caveat before using these as your project's baseline

These numbers were computed using `mostly-original`'s evaluation pipeline, which (before this
branch's `env_factory.py` consolidation existed) built an **independently-drifted, narrower**
wrapper stack for evaluation than for training — missing several reward-shaping terms that
`new-code`'s unified `build_env()` includes in both training and evaluation. The **mission-
outcome metrics** (HVAA survival, red kill, blue loss) are ground-truth episode facts
unaffected by this and remain directly usable as reference numbers. The **raw PPO reward**
values are not directly comparable to anything produced under `new-code`'s evaluation stack.

**Recommendation**: for your actual thesis comparison (existing/baseline reward vs. your
redesigned mission-aligned reward), establish a fresh baseline run under `new-code` at
whatever timestep budget you settle on, rather than citing the numbers above directly as that
baseline. Use this table for qualitative sanity-checking ("does my baseline behave the way
Bieger's thesis and this pilot both suggest it should?") rather than as a precise numeric
reference point.
