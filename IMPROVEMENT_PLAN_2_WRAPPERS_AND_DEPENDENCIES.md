# Improvement Plan 2: Wrapper Design and Dependency Coupling

**Audience:** this file is written to be read by a human *and* handed to an LLM coding
assistant as an implementation spec. If you're an LLM implementing this: read the
[Ground Rules](#ground-rules) section first and follow it strictly. This plan is more
sensitive than Improvement Plan 1 — several tasks here touch files that shape training
behavior, and getting the "no functional change" constraint wrong here is much easier to do
by accident.

**Repo:** `bvr-cca-escort` (AFIT-ENS-MS-26-M-085, Bieger). Background reading:
`CODEBASE_GUIDE.md` (full architecture + audit — cited extensively below) and `HOW_TO_RUN.md`.

---

## Ground Rules

1. **No task in this plan may change the sequence of reward computations, their coefficients,
   the observation vector's contents, or the action pipeline for any existing CLI flag
   combination.** This plan is entirely about *how the code is organized*, not what it
   computes. Several tasks below explicitly require a before/after equivalence check for
   exactly this reason — do not skip those checks.
2. **Do not delete any class or function without first re-confirming, via a fresh repo-wide
   grep at the time you do the work, that it is genuinely unreferenced.** This document lists
   specific things as "confirmed dead code" based on an audit performed earlier — code moves
   fast, re-verify before acting on a claim made in this document.
3. Do these tasks incrementally, one at a time, and prove each one preserves behavior before
   moving to the next (see the "equivalence check" method described in Task 1). Don't batch
   multiple structural changes into one commit — if something breaks, you want to know which
   change did it.
4. If a task description conflicts with something you observe directly in the code, trust the
   code and flag the discrepancy rather than proceeding on stale information.

---

## Part A: The Wrapper-of-Wrappers Design

### Is the pattern itself bad?

**No.** Composing a `gymnasium.Env` out of a chain of `gym.Wrapper` subclasses — each adding
one independent concern (an observation transform, a reward-shaping term, an action
constraint) — is the standard, idiomatic way to build RL environments in Gymnasium/Stable-
Baselines3. It's genuinely the *right* tool for "I want N independent, toggleable pieces of
behavior layered on a base simulator." The design intent here (each shaping term is its own
small wrapper class) is sound and shouldn't be undone wholesale.

**What's actually wrong is three specific implementation problems**, all fixable without
touching the pattern itself:

### Problem 1: The stack is built twice, by hand, and the two copies have drifted apart

`rl_files/train_bace_clean.py` and `rl_files/evaluate_experiment_seeds.py` each define their
**own separate copy** of the environment-construction logic (`make_env()` in the former, an
equivalent inline builder in the latter). Some wrapper classes are even redefined from scratch
in both files rather than imported once. Confirmed concretely:

- `HVAASurvivalEveryNStepsBonus` is instantiated with `every_n_steps=25, bonus=0.01` in
  `train_bace_clean.py` and `every_n_steps=50, bonus=0.05` in `evaluate_experiment_seeds.py` —
  different reward-shaping rates during training vs. evaluation, presumably by accident (see
  `CODEBASE_GUIDE.md` §5.3).
- `FireCooldownWrapper` is annealed (70→120 steps over 5,000,000 steps) in `train_bace_clean.py`
  but fixed at a flat 120-step cooldown, no annealing, in `evaluate_experiment_seeds.py`.

This is a direct consequence of *not* having one source of truth — nothing structurally
prevents the two copies from disagreeing, and nothing would tell you if they did except reading
both files side by side, which is what the earlier audit had to do to find this.

**This is very likely the actual cause of at least one real, confirmed bug already fixed**: the
`model.save()` pickling crash (see Problem 3 below) came from a monkey-patch introduced in the
callback layer that nobody had a single, obvious place to reason about because environment/
training/evaluation construction logic is scattered across multiple independently-maintained
copies.

### Problem 2: Stack construction is 150+ lines of sequential imperative code, not data

Inside `train_bace_clean.py::make_env()`, the wrapper stack is built as ~15 sequential
`e = SomeWrapper(e, arg1=..., arg2=...)` lines, interleaved with `print()` calls, conditionals
for optional wrappers, and a locally-defined debug wrapper class (`_TermTrace`). There is no
single place you can look to see "here is the current wrapper order" without reading the whole
function top to bottom and mentally filtering out debug/logging code. This makes the order
(which matters — rewards accumulate through the chain, and one wrapper reading an observation
index another wrapper hasn't touched yet is order-dependent) hard to audit and easy to get
wrong when adding a new wrapper.

### Problem 3: One piece of code reaches *outside* the wrapper stack into the PPO model itself

`rl_files/rollback_eval_callback.py`'s `RollbackEvaluationCallbackTopK._init_callback()` does
`self.model.train = wrapped_train` — monkey-patching the `train()` method of the live PPO model
object from within a callback, to implement "skip N training updates after a rollback." This
breaks the wrapper/callback abstraction boundary: callbacks are supposed to observe and
influence training through SB3's public hook methods (`_on_step`, `_on_rollout_end`, etc.), not
reach in and replace methods on the model itself.

This was not a hypothetical risk — it caused a real, confirmed bug: leaving a `'train'` key in
`model.__dict__` (even after attempting to restore it) made `model.save()` crash with
`TypeError: cannot pickle '_thread.lock' object`, because a bound method's `__self__` bypasses
Stable-Baselines3's save-exclusion filtering and drags the live environment (with its open
socket connection) into the pickle attempt. **This specific instance has already been fixed**
(see `CODEBASE_GUIDE.md` §5.2b) by having the restore step `del` the instance attribute instead
of reassigning it — but the monkey-patching *pattern* itself is still there and is worth
replacing properly, not just patched around, per Task 3 below.

### Problem 4: Confirmed-dead code sits alongside live code with no signal of which is which

A reader can't tell, just from opening these files, that the following are **never
instantiated anywhere in the codebase** (confirmed by repo-wide grep during the earlier audit —
re-verify before touching):
- `ExpertMixPPO` / `ExpertRolloutBuffer` in `train_bace_clean.py` (~370 lines) — a full
  alternative PPO implementation with its own rollout collection and behavioral-cloning loss.
  `main()` builds a plain `stable_baselines3.PPO` instead.
- `RewardBreakdownLogger` in `reward_breakdown_logger.py` (only `CompactRewardLogger` is used).
- `HVAADestructionPenalty` / `HVAADestructionPenaltyWithDecay` in `hvaa_destruction_penalty.py`.
- `RangeClosingShaping` (defined inline in `train_bace_clean.py`).
- `AggressiveHunterV3` / `AggressiveHunterTunable` in `aggressive_hunter_fixed.py` (only
  `AggressiveHunterFixed` is used).
- `FeatureOnlyWrapper`, `create_enriched_env`, `get_ablation_summary`, `print_ablation_matrix`
  in `enriched_observation_wrapper_ablation.py` — only exercised by that file's own
  `if __name__ == "__main__"` block.
- `compute_features_for_agent`, `get_feature_names`, `extract_agent_state_from_obs` in
  `pursuit_evasion_features.py` — same, self-test-only.
- The module-level `DEBUG_HVAA_OBS` flag in `train_bace_clean.py`.

None of this is *wrong*, but a maintainer unfamiliar with the codebase (as you may be) has no
way to distinguish "this is the real implementation" from "this was an earlier design that got
superseded and left in place" without doing exactly the kind of full-repo grep the earlier
audit did. That's wasted effort every time someone new picks this codebase up.

---

### Task 1: Establish an equivalence-check method (do this before any other Part A task)

Before restructuring anything, set up a way to prove a refactor didn't change behavior:

1. Pick a fixed seed and a short, fixed sequence of actions (e.g., 50-100 steps of a simple
   deterministic policy, or just `env.action_space.seed(N); env.action_space.sample()` per
   step with a fixed seed).
2. Write a small script that constructs the environment via the *current* (pre-refactor)
   `train_bace_clean.py::make_env()` path, resets with a fixed seed, steps through the fixed
   action sequence, and dumps every observation vector, reward, and `info` dict to a file
   (e.g., `tools/equivalence_baseline.json`) — do this once, before refactoring, and commit
   the output.
3. After each subsequent task in this plan, re-run the same script against the *refactored*
   construction path and diff the output against the committed baseline. Any difference means
   the refactor changed behavior and must be fixed before proceeding — do not proceed to the
   next task with a known behavior diff.
4. Do this for at least two configurations: the default baseline (no enriched obs) and one
   configuration with all five factors on (`--use-enriched-obs --ablation-config all
   --expert-alignment` — note per the ground rules of Improvement Plan 1, don't combine
   `--expert-alignment` and `--expert-blending`, so test them in two separate runs if you want
   both covered).

This is the single most important task in this file — every other task's "don't change
behavior" requirement is only actually verifiable if this exists.

### Task 2: Extract a single, shared, declarative wrapper-stack definition

1. Create `rl_files/env_factory.py` (or similar name) containing one function,
   e.g. `build_env(bace_config, args, is_eval: bool) -> gym.Env`, that is the **only** place
   the wrapper stack is constructed. Both `train_bace_clean.py` and
   `evaluate_experiment_seeds.py` should import and call this function instead of each having
   their own copy.
2. Represent the stack as an explicit ordered data structure before applying it — e.g. a list
   of `(WrapperClass, kwargs_dict)` pairs (kwargs computed from `args`/`is_eval` as needed),
   applied in a simple loop. This makes "what's the current order" a single readable list
   instead of 150 lines of interleaved logic, and makes it possible to unit-test "the stack
   has exactly these N layers in this order" directly.
3. Where training and eval currently use *different* wrapper parameters intentionally (e.g.
   `ExpertAlignmentWrapper`/`ExpertActionBlendingWrapper` are correctly skipped at eval time —
   this one is deliberate and documented, not a bug), keep that distinction, but make it
   explicit and centralized (an `if not is_eval:` branch in the one shared function) rather
   than implicit (two separately-maintained files that happen to differ). Where training and
   eval currently differ *unintentionally* (the two bugs listed in Problem 1 above), **this is
   the point where you must decide and document which behavior is correct** — do not silently
   pick one. Flag both cases explicitly in your commit message / PR description and let a human
   confirm the choice before merging, since either choice is a behavior change relative to
   *something* that currently exists.
4. Run the Task 1 equivalence check against the *training* path after this change (the eval
   path will differ from before by design, per point 3 — that's expected and should be called
   out, not treated as a failure).

**Acceptance criteria:** `train_bace_clean.py` and `evaluate_experiment_seeds.py` both call
`env_factory.build_env(...)`; no wrapper class is defined in more than one file; the ordered
list of `(WrapperClass, kwargs)` is visible as data, not derived by reading imperative code;
Task 1's baseline diff is clean for the training path.

### Task 3: Replace the `model.train` monkey-patch with a supported mechanism

`RollbackEvaluationCallbackTopK`'s "skip N training updates" mechanism needs a way to make
`PPO.train()` do nothing for the next N calls, without patching the method object itself.
Options, in order of preference:

1. **Preferred:** implement this as a tiny `PPO` subclass (yes, a subclass — but a *small*,
   clearly-scoped one, not a reimplementation of rollout collection like the dead `ExpertMixPPO`
   was) that overrides `train()` to check a skip-counter attribute set by the callback via a
   public method call (not attribute-poking), then calls `super().train()` when not skipping.
   This keeps the override as a normal part of the class hierarchy — `model.__dict__` never
   gains a stray `'train'` instance key, so the pickling failure mode this plan is trying to
   avoid can't recur.
2. **Alternative if a subclass is undesirable:** have the callback track "should the *next*
   rollout's training step be skipped" as its own state, and use SB3's existing callback hook
   surface (`_on_rollout_end`, or whatever hook fires immediately before `train()` is called in
   the SB3 version this repo targets) to prevent the training step from happening — check
   Stable-Baselines3's actual `OnPolicyAlgorithm.learn()` loop in the installed version to see
   if there's a supported way to skip a `train()` call from a callback without subclassing.
   If there isn't, option 1 is simpler and more robust.
3. Whichever option is chosen: **`model.__dict__` must never contain an attribute after
   training that wasn't there before `model.learn()` was first called.** This is the concrete,
   testable invariant that failed before. Add a one-line assertion or test confirming this
   (e.g., snapshot `set(model.__dict__.keys())` before `learn()` and after, assert equal minus
   any keys SB3 itself is known to add during training).
4. Run the Task 1 equivalence check, and additionally confirm `model.save()` still succeeds and
   the saved model reloads correctly and produces the same actions as the unsaved in-memory
   model for a fixed observation.

**Acceptance criteria:** the rollback "skip training" behavior works identically to before;
`model.save()` succeeds after training with rollback active; no monkey-patching of bound
methods onto model instances remains anywhere in the codebase.

### Task 4: Remove confirmed-dead code

For each item listed under Problem 4 above: **re-confirm via a fresh grep of the entire repo**
(not just `rl_files/`) that it has zero references outside its own definition and its own
`__main__`/self-test block, then delete it. If a grep turns up even one unexpected reference,
stop and investigate rather than deleting — don't trust this document's claim over what you
actually find.

Do this as its own commit, separate from Tasks 2/3, so it's trivially revertible if any of
these turns out to be less dead than believed.

**Acceptance criteria:** every item in the Problem 4 list is either deleted (grep-confirmed
dead) or explicitly kept with a one-line comment explaining why it's retained despite appearing
unused; the training and evaluation scripts still run and produce identical output to the
Task 1 baseline (deleting genuinely-dead code should be a no-op for behavior by definition, but
verify anyway — that's the whole point of the equivalence check).

---

## Part B: Dependency Coupling

### What's actually coupled, and why it's fragile

The repo has several places where two pieces of code must agree on an implicit contract, with
nothing enforcing that agreement:

1. **Observation index coupling.** `b_ace_py/enriched_observation_wrapper_ablation.py`'s
   `_extract_state_from_obs()` hardcodes numeric indices (`IDX_OWN_HDG = 5`, `IDX_OWN_SPEED = 6`,
   etc.) assuming a specific, fixed layout of the raw observation vector Godot emits. If the
   Godot-side observation order ever changes (a plausible thing to happen during 2v2 or missile-
   evasion follow-on work), this silently starts reading the wrong field into the wrong feature
   computation — no error, just quietly wrong numbers feeding a differential-game formula. There
   is currently no check anywhere that the *actual* observation labels Godot reports at startup
   (which are already printed to the console — see `B_ACE_GodotPettingZooWrapper.py`) match
   what this file assumes.
2. **`info`-dict string-key coupling.** Many files (reward wrappers, `evaluate_experiment_seeds.py`,
   `SimManager.gd` on the Godot side) communicate purely through dictionary string keys —
   `"red_killed_event"`, `"hvaa_alive"`, `"blue_shot_first"`, `"blue_pk"`, and many more (see
   `CODEBASE_GUIDE.md` for the fuller enumeration from the Godot-side audit). Nothing declares
   this contract in one place; it's implicit in whichever Python file happens to read a given
   key, and a rename on the Godot side would produce silent `None`/default-value fallbacks
   rather than an error in most of the places that read these keys.
3. **The wrapper-duplication issue from Part A** is *also* a dependency-coupling problem, not
   just a design-cleanliness one: it's two pieces of code that must be kept in sync by a human
   remembering to update both, with nothing that would catch drift.
4. **Reaching into Stable-Baselines3's private internals.** `rollback_eval_callback.py` reads/
   writes `model._total_timesteps`, `model.lr_schedule`, and directly pokes optimizer
   `param_group['lr']` values. These are implementation details of the installed SB3 version,
   not its public API — an SB3 upgrade could rename or restructure any of these without it
   counting as a breaking change from SB3's own perspective, silently breaking this repo.
5. **No pinned dependency versions anywhere** (no `requirements.txt`/lockfile existed before
   this project's setup was done manually — see `HOW_TO_RUN.md` for the versions confirmed
   working: `gymnasium 1.3.0`, `pettingzoo 1.27.0`, `numpy 2.5.2`, `scipy 1.18.0`,
   `torch 2.13.0`, `stable-baselines3 2.9.0`). Two different setups of this repo could
   currently silently resolve to different library versions with different behavior.
6. **No automated tests anywhere** to catch any of the above breaking. This is the root reason
   all of the above are risks rather than just theoretical possibilities — there's no safety
   net.

### Task 5: Pin dependency versions

Add a `requirements.txt` (simplest option, matches how this project is otherwise structured) at
the repo root, pinning at minimum:
```
gymnasium==1.3.0
pettingzoo==1.27.0
numpy==2.5.2
scipy==1.18.0
torch==2.13.0
stable-baselines3==2.9.0
tensorboard==2.21.0
tqdm
rich
```
(These are the exact versions already confirmed working during setup — see `HOW_TO_RUN.md`.
Use `>=` instead of `==` for any of these if you have a specific reason to want flexibility,
but pin at least `stable-baselines3` tightly given Task 6 below depends on its exact internal
behavior.) Reference this file from `HOW_TO_RUN.md`'s setup instructions (replacing the
individual `pip install` list currently there with `pip install -r requirements.txt`).

**Acceptance criteria:** a fresh `python -m venv` + `pip install -r requirements.txt` on a
clean machine reproduces the exact environment already confirmed working.

### Task 6: Make the observation/info-dict contracts explicit and self-checking

1. Add a small module (e.g. `b_ace_py/contracts.py`) that declares, as plain Python data (a
   list of strings, or a small dataclass), the **expected** base observation labels (the 27
   labels already printed at startup by `B_ACE_GodotPettingZooWrapper` — copy them verbatim
   from a real run's console output) and the **expected** `info`-dict keys the Python side
   actually reads (enumerate these by grepping `rl_files/` and `b_ace_py/` for `info.get(`/
   `info[` calls — `CODEBASE_GUIDE.md`'s earlier audit already has a fuller list of Godot-
   emitted fields to cross-check against).
2. At environment-construction time (inside the new `env_factory.py` from Task 2, or at the
   point `B_ACE_GodotPettingZooWrapper` receives its observation labels from Godot), add an
   assertion comparing Godot's actual reported labels against the expected list from #1, with
   a clear error message naming exactly what differs if they don't match. **This must not
   fire against the current, correct Godot build** — if it does, the expected list in #1 was
   transcribed wrong, not the check itself.
3. This converts "silently compute wrong features forever" into "fail loudly and immediately
   the first time anything changes," which is a pure safety improvement with no effect on
   current behavior (as long as #2's assertion is correct).

**Acceptance criteria:** the contract module exists and is used; running the existing
training/eval scripts against the current Godot binary produces no assertion failures; a
scratch test that deliberately renames one label (e.g., in a throwaway copy of the config or by
mocking Godot's response) causes the assertion to fire with a clear message.

### Task 7: Reduce and clearly flag remaining SB3-private-API reliance

1. For each private-attribute access in `rollback_eval_callback.py`
   (`model._total_timesteps`, `model.lr_schedule`, optimizer `param_group` mutation), check
   whether Stable-Baselines3's public API (as of the pinned version from Task 5) offers a
   supported equivalent. If it does, switch to it. If it doesn't, **do not invent a workaround
   that's equally fragile** — instead, add a clear comment at each access point explaining
   *why* the private attribute is used and what SB3 version this was verified against, so a
   future upgrade attempt knows exactly what to re-check.
2. Do not attempt to eliminate 100% of private-API reliance if a genuinely supported
   alternative doesn't exist — the goal here is reducing *undocumented* risk, not achieving an
   arbitrary purity target at the cost of reimplementing SB3 internals.

**Acceptance criteria:** every remaining private-attribute access in the codebase has an
adjacent comment explaining why, and the Task 1 equivalence check still passes.

### Task 8: A minimal regression test that would have caught the bugs already found

Add one pytest test (mark it `slow` or put it behind a flag if it needs the real Godot binary
and is too slow for a fast test loop) that: launches the real environment via
`env_factory.build_env(...)`, resets, steps a fixed short sequence of actions, and asserts (a)
observation shape and labels match the Task 6 contract, (b) reward is always a finite float,
(c) after a `model.learn()` call over a small number of timesteps with
`RollbackEvaluationCallbackTopK` attached, `model.save()` succeeds and the reloaded model
predicts the same action for a fixed observation as the pre-save model. This test would have
caught two of the three runtime bugs already found and fixed during initial setup (see
`CODEBASE_GUIDE.md` §5.2b) — it's cheap insurance against the same class of bug recurring.

**Acceptance criteria:** the test exists, passes against the current (fixed) codebase, and
would fail if either of the two relevant fixes from `CODEBASE_GUIDE.md` §5.2b were reverted
(confirm this by temporarily reverting one and checking the test catches it, then re-applying
the fix).

---

## On "manual PPO" specifically — what's actually true today

**The live training path already uses plain, unmodified `stable_baselines3.PPO` directly** —
`train_bace_clean.py::main()` constructs `PPO(policy="MlpPolicy", env=vec_env, ...)` and calls
`model.learn(...)` on it. There is no custom training loop running in practice.

What *does* exist, and reads like a custom/manual PPO implementation, is `ExpertMixPPO` /
`ExpertRolloutBuffer` — but per Problem 4 above, this is confirmed dead code, never
instantiated. **Task 4 (deleting confirmed-dead code) already resolves the "decouple from
manual PPO" request** — once it's deleted, there's no ambiguity left about whether a custom PPO
implementation is or isn't in the training path, because there won't be one in the repo to
wonder about.

### Should this move to RLlib instead of Stable-Baselines3?

**Not as part of this plan.** SB3 is already the framework in active use, it already works
(confirmed by a real end-to-end training run), and swapping frameworks entirely would touch
checkpoint formats, hyperparameter surfaces, environment-wrapper expectations, and almost
certainly subtle training dynamics — exactly the kind of functional change the ground rules of
this plan rule out. There is no problem described in this document that requires a framework
swap to solve; every issue above is solvable within SB3.

**Where RLlib genuinely becomes worth evaluating:** if/when this project moves to real
multi-agent (2v2) training (see `CODEBASE_GUIDE.md`'s 2v2 section), since SB3 is fundamentally
built around single-agent `Env`/`VecEnv` and this repo already has to work around that (the
`SingleAgentBACEEnv` collapsing layer), whereas RLlib has native multi-agent training support
built on top of the same kind of `ParallelEnv`/PettingZoo interface `B_ACE_GodotPettingZooWrapper`
already implements. That's a decision for whoever picks up the 2v2 work specifically, informed
by how much of `SingleAgentBACEEnv`'s workarounds they'd rather not reinvent under a different
framework — it is explicitly **not** part of this plan.

---

## Explicitly out of scope for this plan

- Any change to reward coefficients, observation contents, or training hyperparameter defaults.
- Removing the wrapper-composition pattern itself in favor of one monolithic environment class
  — the pattern is correct, only its current implementation has the four problems above.
- Deleting any wrapper or class not confirmed dead by a fresh, repo-wide grep at
  implementation time.
- Migrating from Stable-Baselines3 to RLlib or any other RL framework.
- Rewriting the Godot/GDScript side to eliminate the info-dict string-key pattern (e.g.,
  switching to a schema/protobuf-based protocol) — Task 6 makes the existing contract explicit
  and self-checking without requiring a protocol change on the Godot side.
