# CSCE 723 Final Project — Writing Outline

A section-by-section bullet outline for the final paper, mapped onto the actual work in this
repo (reward redesign for BVR-CCA escort RL, following on Capt. Bieger's AFIT thesis). Each
section lists what to write and which existing figure(s)/numbers to pull in. Nothing here is
prose to copy-paste — it's a scaffold pointing at the real artifacts you already have.

**Figures referenced below live in `results/figures/` and `results/runs_sb3/<experiment>/`.**
Regenerate any of them with `python src/analysis/generate_figures.py` /
`generate_comparison_figures.py`, or a fresh `evaluate_experiment_seeds.py` run, if you change
the underlying data before submitting.

---

## Abstract (~1-2 paragraphs)

- One sentence: BVR-CCA escort RL (PPO, B-ACE Godot sim) trains agents against a reward that
  correlates imperfectly with the actual mission outcome (thesis's own Sec 4.7.1 finding).
- One sentence stating your objective: redesign the reward to be mission-aligned and test
  whether win rate (HVAA survives **and** CCA survives) improves without sacrificing training
  stability.
- One–two sentences on methodology: quantified the specific misalignment mechanism (a per-step
  "still alive" bonus that pays more for loitering than finishing the mission), removed it, added
  terminal HVAA-loss/mission-success bonuses, trained both variants full-scale (10M timesteps),
  compared via win rate + a cross-seed learning-curve/AlgScore analysis.
- One–two sentences of key results: comparable mission-level performance (state the win-rate
  numbers), intentionally lower raw reward (explain why that's expected, not a regression), and
  a genuine finding — the mission-aligned run's training curve destabilizes after its peak
  (~6.6M steps) while baseline keeps improving to 10M.
- One sentence of conclusion: reward reshaping succeeded at removing the identified
  misalignment without harming mission performance on this seed, but revealed a
  stability/sparsity tradeoff worth investigating further (→ motivates your future thesis work).

---

## 1. Introduction

- **Motivation & Context**
  - DoD relevance: Beyond-visual-range escort missions for Collaborative Combat Aircraft (CCA)
    are a live Air Force autonomy priority; this is a direct extension of Capt. Bieger's AFIT
    thesis (AFIT-ENS-MS-26-M-085) and your own stated thesis trajectory — say so explicitly,
    this is exactly the kind of framing the guidance asks for ("is it related to your
    thesis/dissertation work?").
  - Background a reader needs: what B-ACE is (Godot-based 3DOF BVR air-combat sim), what a
    blue-lead/HVAA/CCA escort scenario is, that the prior thesis trained PPO agents on a
    hand-designed reward and found (Sec 4.7.1) that raw training reward ranks configurations
    differently than actual win rate does.
- **Problem Statement**: does the existing reward function reward the right thing? If not, can a
  redesigned, more mission-aligned reward change agent behavior without degrading performance?
- **Contribution & Scope**
  - Contribution: (1) a quantified, mechanistic explanation of *why* the existing reward
    misaligns with the mission (not just "it's imperfect" — the exact wrapper and constants
    responsible), (2) an implemented, switchable alternative reward mode
    (`--reward-mode mission_aligned`), (3) a same-seed, same-hyperparameter, full-scale (10M
    timestep) comparison.
  - Scope/limitations to state up front: single seed per condition (say this plainly — it's a
    real limitation, not something to hide), one specific redesign (not an exhaustive search
    over reward designs).
- **Goalposts / baseline**: the existing (unmodified) reward, trained under identical
  hyperparameters, is your baseline. "Satisfactory" = mission-outcome metrics (win rate, HVAA
  survival) hold steady or improve under the redesigned reward, since raw reward is expected to
  drop by construction (explain why in Methodology/Results, don't let a reviewer misread it as
  a regression).
- **Roadmap**: one sentence per section.

**Figure for this section**: `results/figures/fig4_reward_vs_winrate_rank.png` — the clearest
single image of "why this problem is real": ranks the 5-config pilot by best reward vs. by
win-rate proxy and shows they disagree. This is your motivating evidence in one picture.

---

## 2. Background & Related Work

Need **≥2 real peer-reviewed sources** — don't fabricate citations. Candidates found this
session (verify access/exact bibliographic details yourself before citing):

- **Reward hacking / reward misspecification** — directly explains the mechanism you found and
  fixed (a dense per-step term dominating the sparse terminal signal is a textbook case):
  - Skalse et al., "Defining and Characterizing Reward Hacking," NeurIPS 2022 —
    https://arxiv.org/pdf/2209.13085
  - Pan et al., "The Effects of Reward Misspecification: Mapping and Mitigating Misaligned
    Models," ICLR 2022 — https://arxiv.org/pdf/2201.03544
- **Prior autonomous air-combat AI** (establishes the lineage this work sits in — hand-crafted
  controllers → learned policies):
  - Ernest et al., "Genetic Fuzzy based Artificial Intelligence for Unmanned Combat Aerial
    Vehicle Control in Simulated Air Combat Missions," Journal of Defense Management, 2016 —
    https://www.researchgate.net/publication/301944635 (the ALPHA/GFT system — good "gap" framing:
    rule-based/fuzzy control predates the learned-policy approach this project and Bieger's
    thesis both take)
- **BVR + PPO / multi-agent RL specifically** (closest technical neighbors to this project):
  - AFIT thesis, "Proximal Policy Optimization Applied to the Beyond Visual Range Air Combat
    Maneuvering Problem" — https://scholar.afit.edu/etd/8281/ (same lab/program as Bieger's
    thesis — check whether this is a direct predecessor Bieger himself cites; if so, it's a very
    natural reference)
  - IEEE, "Application of Deep Reinforcement Learning in Maneuver Planning of Beyond-Visual-Range
    Air Combat" — https://ieeexplore.ieee.org/document/9358136/
- **Bieger's own thesis** is your primary reference throughout (not "peer-reviewed" but it's the
  work you're extending — cite it explicitly and often).

Structure this section as: (1) prior rule-based air-combat AI, (2) the shift to learned
policies (PPO) for the same problem class, (3) reward hacking/misspecification as a known
general failure mode in RL, (4) the gap: Bieger's thesis identified reward/win-rate
misalignment but didn't redesign the reward — that's your contribution.

---

## 3. Methodology

Guiding principle: reproducibility. You have exact code/line numbers for all of this in
`docs/SESSION_SUMMARY.md` — pull directly from there rather than re-deriving.

- **Environment** (RL-specific guidance requirement):
  - Observation space: 27-dim baseline vector (or note enriched-obs dimensionality if you used
    it), built by `SingleAgentBACEEnv` wrapping `B_ACE_GodotPettingZooWrapper`.
  - Action space: 4-dim continuous, `[-1, 1]`.
  - Dynamics: 3DOF kinematics, Godot-side simulation, TCP socket to the Python training process,
    episodes terminate on HVAA/CCA destruction, mission success, or a 13,500-step cap.
  - Cite `docs/SESSION_SUMMARY.md` §2 for the exact `build()`/`step()`/`reset()` code if your
    paper wants a code listing.
- **Reward structure** (RL-specific guidance requirement — this is your paper's centerpiece):
  - Explain the two-layer architecture: Godot-side one-time terminal rewards (kill, mission
    success) + Python-side per-step shaping wrappers (`REWARD_SHAPING_STACK` in
    `env_factory.py`).
  - State the exact mechanism you found: `MissionTempoShapingPerStep` pays **+0.006/step** for
    every step the HVAA survives after a kill — up to ~+70 over a ~12,000-step episode — which
    exceeds the one-time mission-completion bonus (+8), so the policy is rewarded more for
    loitering than finishing. Use the actual numbers from `fig5_reward_misalignment_illustration.png`.
  - State your fix: zero that per-step bonus, add `MissionOutcomeBonus` (-15 one-time for HVAA
    loss, +25 one-time for mission success). This is a **direct, evidence-based design
    justification** — exactly what the guidance asks for ("justify your design choices").
  - Discuss the tradeoff explicitly: a sparser reward is more mission-aligned but (per your own
    Results) harder for PPO to keep improving on — this is real scientific content, not a flaw
    to hide.
- **Agent & training**:
  - PPO via Stable-Baselines3. List the hyperparameters from `docs/THESIS_REPRODUCTION_GUIDE.md`
    §5 (learning rate/entropy decay schedules, γ=0.99, GAE λ=0.9207, clip=0.198, n_steps=6144,
    batch=1024, epochs=8, net=2×512) and **say plainly these are inherited from Bieger's prior
    Optuna search, not retuned by you** — that's an honest, correct scoping statement, not a
    weakness.
  - Checkpoint-rollback mechanism (`RollbackEvaluationCallbackTopK`) — briefly explain it exists
    to protect against exactly the kind of late-training degradation you found in the
    mission-aligned run.
  - Training budget: 10,000,000 timesteps, seed 42, one seed per condition — **state this
    limitation here too**, not just in Conclusion.

**Figure for this section**: `results/figures/fig5_reward_misalignment_illustration.png`
(quantified misalignment mechanism) is the one figure this section really needs. Optionally
`results/figures/fig1_full_scale_learning_curve.png` if you want to show the rollback mechanism
visually (the dotted vertical lines).

---

## 4. Results & Analysis

Lead with the comparison, not a data dump. You already have the full written analysis in
`docs/REWARD_REDESIGN_RESULTS.md` — this section should read like a tightened version of that.

- **Headline comparison table** (from `docs/REWARD_REDESIGN_RESULTS.md` §2 / your latest
  `superlative_eval_results.json` re-run): mean reward, win rate, HVAA survival, red kill, blue
  loss, for baseline vs. mission-aligned. State plainly: win rate is statistically
  indistinguishable at n=1 seed / 20 episodes (a 1-episode swing either way) — **say this out
  loud**, don't let a reader think you're claiming a significant win-rate improvement you can't
  support.
- **Why raw reward dropped ~40%**: explain via `fig_c3` — this is expected given the
  Methodology change, not a regression. This paragraph is what stops a reviewer from
  misreading your own headline table.
- **The real finding**: the mission-aligned run's evaluation-return curve rises with baseline
  through ~6.6M steps, then *declines* for the remaining 3.4M while baseline keeps climbing.
  This is your most interesting result — spend real analysis on it: your hypothesis (the dense
  per-step signal was doing double duty as a stabilizing shaping term for credit assignment, not
  just injecting misalignment; removing it bought alignment at the cost of a harder credit-
  assignment problem). This is exactly the "negative results are still results" / "analyze,
  don't just state" guidance — this is where you earn Results-section points.
- **AlgScore / Peak-Return / AUC framing** (new this session, `experiment_statistics.py`):
  report Peak Return, time-averaged AUC, and AlgScore for both runs — the AUC number in
  particular (strongly negative for mission-aligned, ~+9 for baseline) is a clean quantitative
  summary of the instability finding above. **State plainly in a footnote/caveat that this
  AlgScore formula is your own reconstruction of a reference metric's name, not a validated
  standard metric** — see `docs/THESIS_REPRODUCTION_GUIDE.md` §6 for exact wording to reuse.

**Figures for this section** (the core Results figures):
1. `results/runs_sb3/reshaped_rewards1/learning_curve_analysis.png` (or `fig_c1_learning_curves_comparison.png`, which overlays both runs directly — probably the better single figure for a paper) — the training-curve divergence.
2. `results/figures/fig_c2_mission_outcomes_comparison.png` — win rate / HVAA / red kill / blue loss, baseline vs. mission-aligned.
3. `results/figures/fig_c3_reward_magnitude_comparison.png` — supports the "why reward dropped" paragraph.

Guidance reminders: figure text should match body-text size (the dataviz-styled figures here
were built at 150 DPI for print — check they're still legible at your paper's column width);
write full descriptive captions (e.g. "Figure 3: Cross-seed mission-outcome rates for the
baseline and mission-aligned reward, 20 evaluation episodes, seed 42. Win rate and HVAA survival
are within one episode of each other; red kill rate is higher under the mission-aligned reward."),
not "results of experiment 2."

---

## 5. Conclusion & Future Work

- **Conclusion**: restate the two real findings — (1) the reward redesign removed the identified
  misalignment mechanism without harming mission-level performance on this seed, (2) it
  introduced a training-stability cost visible only in the learning curve, not the final
  checkpoint (which the existing rollback mechanism already protects against in practice).
- **Limitations** (be direct, this is expected and rewarded by the rubric, not penalized):
  - Single seed per condition — the win-rate comparison is not statistically powered to detect
    small differences; see `docs/REWARD_REDESIGN_RESULTS.md` §4 for exactly what would be needed
    (2-3 more seeds per condition).
  - The AlgScore metric is a same-name reconstruction, not a verified standard — flag this again
    here so it's not mistaken for a validated benchmark result.
  - Hyperparameters were inherited from prior work, not retuned for the new reward — it's
    possible the mission-aligned reward would benefit from different entropy/LR schedules that
    weren't explored.
- **Future work** (this is your bridge to the actual thesis — say so explicitly):
  - Add seeds to get a real confidence interval on the win-rate delta (highest-value, cheapest
    next step).
  - Investigate the post-peak decline directly: does a smaller post-peak learning rate, a longer
    `ActionSmoothnessPenalty` anneal, or an intermediate reward density (partial credit signals
    short of the full per-step bonus) recover stability without reintroducing the misalignment?
  - Extend the same mission-aligned framework to the full factorial design (Factors A-E) rather
    than just baseline vs. one redesign.

---

## Presentation Outline (10-15 min)

Mirror the paper; weight toward Methodology + Results per the rubric.

1. **Title/motivation** (1 slide, ~1 min): DoD/thesis framing, one sentence problem statement.
2. **Background** (1 slide, ~1-2 min): one sentence on prior rule-based AI → learned PPO
   policies, one sentence on reward hacking as a known RL failure mode, cite your 2+ sources by
   name on the slide (not full citations — save that for a backup/references slide).
3. **The problem, quantified** (1 slide, ~2 min): `fig5_reward_misalignment_illustration.png`
   or `fig4_reward_vs_winrate_rank.png` — whichever makes the misalignment click faster.
4. **Methodology** (2 slides, ~3-4 min): the reward-architecture diagram/description, then the
   specific fix (tempo bonus → 0, add terminal bonuses). This is where the rubric wants the most
   depth — don't rush it.
5. **Results** (2-3 slides, ~4-5 min): headline comparison table, `fig_c1`/`fig_c2`, then the
   training-instability finding as its own slide — it's your most interesting result, give it
   room.
6. **Conclusion + future work** (1 slide, ~1-2 min): the two findings, the seeds-needed
   limitation, the bridge to your thesis.

Keep each slide's text short — the figures above are already built to stand on their own; let
them do the talking and narrate the analysis verbally rather than repeating figure text.
