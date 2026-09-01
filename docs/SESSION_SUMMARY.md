# Session Summary: Reward Redesign Implementation & First Comparison

Branch: `new-code`. Companion docs: `REWARD_SHAPING_STEPS_FORWARD.md` (full rationale),
`REWARD_REDESIGN_RESULTS.md` (results/analysis for this comparison), `PILOT_EXPERIMENT_RESULTS.md`
(earlier 5-config pilot), `BRANCH_COMPARISON.md` (`mostly-original` vs `new-code`).

## 1. What was done this session

1. **Fixed a latent checkpoint-corruption bug.** `RollbackEvaluationCallbackTopK._create_scaled_lr_schedule()`
   in `src/rl_files/rollback_eval_callback.py` returned a closure that captured `self` via
   `self.min_lr`. Once a rollback fired (any run past 4,000,000 timesteps), that closure got
   assigned onto the model and made the whole model object unpicklable by `cloudpickle` at
   `model.save()` time — silently producing 0-byte `.zip` checkpoints with no error. Fixed by
   capturing `min_lr` and `base_schedule` as local values before defining the closure.
2. **Implemented a mission-aligned reward mode** in `src/rl_files/env_factory.py` (`build_env()`'s
   new `reward_mode` parameter) and threaded it through `train_bace_clean.py` and
   `evaluate_experiment_seeds.py` as a `--reward-mode {baseline,mission_aligned}` CLI flag
   (details in §3 below).
3. **Added a `win_rate` metric** to `evaluate_experiment_seeds.py` (HVAA survives *and* CCA
   survives — the thesis's own definition), since the existing metrics only tracked HVAA
   survival, red kill, and blue loss separately.
4. **Trained and evaluated two full-scale (10,000,000-timestep, seed 42) runs**:
   `my_first_run` (baseline reward) and `reshaped_rewards1` (mission-aligned reward), and
   compared them — see `REWARD_REDESIGN_RESULTS.md` for the full write-up. Headline result:
   mission-outcome performance is comparable (95% vs 90% win rate on 20 eval episodes), raw
   reward is intentionally lower (no more per-step loiter bonus), and the mission-aligned run's
   evaluation reward peaks around 6,600,000 timesteps then degrades — the rollback mechanism
   already protects the deployed checkpoint from this.
5. **Generated comparison figures** (`src/analysis/generate_comparison_figures.py` →
   `fig_c1/c2/c3_*.png`) extending the existing pilot-figure pipeline
   (`src/analysis/generate_figures.py`).

## 2. The environment lifecycle — build, call, step, reset (exact code)

Everything below is `src/rl_files/env_factory.py` unless noted.

### Build — `build_env()` assembles one fully-wrapped environment

```python
# src/rl_files/env_factory.py, build_env() signature (line 668)
def build_env(
    bace_config: Dict[str, Any],
    *,
    is_eval: bool,
    log_dir_str: str,
    seed: int,
    use_enriched_obs: bool = False,
    ablation_config: str = "all",
    expert_alignment: bool = False,
    expert_blending: bool = False,
    max_episode_steps: int = 13_500,
    controlled_agent_id: str = "agent_0",
    reward_mode: str = "baseline",   # <-- new this session: "baseline" | "mission_aligned"
    ...
) -> gym.Env:
```

Internally, `build_env()`:
1. Constructs the raw environment: `e = SingleAgentBACEEnv(bace_config=bace_config, controlled_agent_id=controlled_agent_id, max_episode_steps=max_episode_steps)` (line 737).
2. Optionally wraps it with `EnrichedObservationWrapper` + `SplitNormalizationWrapper` if `use_enriched_obs=True` (Factors A/B/C).
3. Picks a reward-wrapper stack based on `reward_mode` (line 868):
   ```python
   if reward_mode == "baseline":
       active_stack = REWARD_SHAPING_STACK          # unchanged, existing reward
   elif reward_mode == "mission_aligned":
       active_stack = MISSION_ALIGNED_REWARD_STACK   # tempo bonus -> 0.0, + MissionOutcomeBonus
   for wrapper_cls, wrapper_kwargs in active_stack:
       e = wrapper_cls(e, **wrapper_kwargs)
   ```
4. Returns the fully-wrapped `gym.Env`, already reset once.

`train_bace_clean.py` and `evaluate_experiment_seeds.py` both call this **same** function —
that's the whole point of `env_factory.py`: one place assembles the wrapper stack, so training
and evaluation can never silently drift apart again (they used to, on the old code).

### Call — constructing `SingleAgentBACEEnv` starts the actual Godot process

```python
# src/rl_files/env_factory.py, SingleAgentBACEEnv.__init__ (line 87)
self._ma_env = B_ACE_GodotPettingZooWrapper(device="cpu", **bace_config)
```

`B_ACE_GodotPettingZooWrapper` (in `src/b_ace_py/B_ACE_GodotPettingZooWrapper.py`) launches the
`B_ACE_v0.1.exe` Godot subprocess, opens the TCP socket to it, and sends the scenario config
(`bace_config`, loaded from a JSON file like `src/b_ace_py/SimpleExample_B_ACE_config.json`).

### Reset — `SingleAgentBACEEnv.reset()` (line 190)

```python
def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
    super().reset(seed=seed)
    self._elapsed_steps = 0
    obs_dict, info_dict = self._ma_env.reset()      # tells Godot to reset the scenario
    obs = self._extract_single_obs(obs_dict)         # pulls out agent_0's observation vector
    info = info_dict.get(self.controlled_agent_id, {})
    return obs, info
```
Every `gym.Wrapper` in the stack (e.g. `KillRewardWrapper`, `MissionOutcomeBonus`) also defines
its own `reset(self, **kwargs)` that just clears its own per-episode state (e.g.
`self._hvaa_loss_applied = False`) and calls `self.env.reset(**kwargs)` — resets cascade down
the wrapper chain automatically via `gym.Wrapper`'s default behavior.

### Step — `SingleAgentBACEEnv.step()` (line 203)

```python
def step(self, action: np.ndarray):
    self._elapsed_steps += 1
    action_dict = self._build_action_payload(action)          # clips to [-1, 1], broadcasts to agents
    obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self._ma_env.step(action_dict)
    obs = self._extract_single_obs(obs_dict)
    reward = float(rew_dict.get(self.controlled_agent_id, 0.0))
    terminated = bool(term_dict.get(self.controlled_agent_id, False)) or bool(term_dict.get("__all__", False))
    truncated = bool(trunc_dict.get(self.controlled_agent_id, False)) or bool(trunc_dict.get("__all__", False))
    if self._elapsed_steps >= self._max_episode_steps and not terminated:
        truncated = True
    info = info_dict.get(self.controlled_agent_id, {})
    return obs, reward, terminated, truncated, info
```
Action is a 4-vector in `[-1, 1]` (heading, and 3 others per the scenario's action space).
Each wrapper in the stack then adds to `reward` and can flip `terminated`/read `info` on its
way back up — e.g. `MissionOutcomeBonus.step()`:
```python
# src/rl_files/env_factory.py, MissionOutcomeBonus.step() (line 358)
obs, reward, terminated, truncated, info = self.env.step(action)
if not self._hvaa_loss_applied and not info.get("hvaa_alive", True):
    reward += self.hvaa_loss_penalty            # -15.0, one-time
    self._hvaa_loss_applied = True
if (terminated or truncated) and info.get("mission_success", False):
    reward += self.mission_success_bonus        # +25.0, one-time
```

## 3. How to run the experiment (PowerShell)

Activate the venv once per terminal session:
```powershell
.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
```

**Train** (baseline reward, the existing default):
```powershell
python src\rl_files\train_bace_clean.py `
  --experiment-name my_run --seed 42 --total-timesteps 10000000 `
  --reward-mode baseline --torch-threads 1
```

**Train with the mission-aligned reward redesign** (just change one flag):
```powershell
python src\rl_files\train_bace_clean.py `
  --experiment-name my_reshaped_run --seed 42 --total-timesteps 10000000 `
  --reward-mode mission_aligned --torch-threads 1
```
Output goes to `results\runs_sb3\<experiment-name>\<experiment-name>_seed42_<timestamp>\`.

**Evaluate** a finished run's best checkpoint (20 episodes, must match the `--reward-mode`
it was trained with):
```powershell
python src\rl_files\evaluate_experiment_seeds.py `
  --experiment-dir results\runs_sb3\my_reshaped_run --episodes 20 --reward-mode mission_aligned
```
Writes `results\runs_sb3\my_reshaped_run\superlative_eval_results.json` (win rate, HVAA survival,
red kill, blue loss, mean reward — the numbers used in `REWARD_REDESIGN_RESULTS.md`).

**Visualize training** (TensorBoard, live while training or after):
```powershell
tensorboard --logdir results\runs_sb3\my_reshaped_run
```

**Regenerate the comparison figures** after a new pair of runs:
```powershell
python src\analysis\generate_comparison_figures.py
```
Edit the `RUNS` dict at the top of `src\analysis\generate_comparison_figures.py` to point at your
new run directories and eval-results files first.

### Key parameters

| Flag | Default | What it does |
|---|---|---|
| `--reward-mode` | `baseline` | `baseline` = existing reward unchanged. `mission_aligned` = removes the per-step post-kill tempo bonus, adds -15 HVAA-loss / +25 mission-success terminal bonuses. |
| `--total-timesteps` | 1,000,000 | Thesis scale is 10,000,000; pilots in this project used 2,000,000. |
| `--seed` | 42 | Also seeds the eval env when the same value is passed to `evaluate_experiment_seeds.py`. |
| `--torch-threads` | 1 | Keep at 1 on this machine — the default PyTorch thread pool competes with the Godot subprocess for CPU and slows training. |
| `--experiment-name` | none (required in practice) | Groups the run under `results\runs_sb3\<name>\`. |
| `--use-enriched-obs` / `--ablation-config` | off / `all` | Factors A/B/C (geometry/engagement/range-limited observation features). |
| `--expert-alignment` | off | Factor D (alignment-based reward shaping during training only). |
| `--expert-blending` | off | Factor E (action blending with an expert policy during training only). |
| `--episodes` (eval script) | — | Number of evaluation episodes; this project used 20 for headline numbers. |

## 4. Caveats to carry forward

- `reshaped_rewards1`'s `reward_mode` was inferred from its reward magnitude, not read from a
  saved config file — its `training_results_seed42.json` never got written (see
  `REWARD_REDESIGN_RESULTS.md` §5). If you rerun this, double check that file exists when the
  run finishes.
- All results above are single-seed. Before citing a win-rate difference between reward modes
  as a finding, add 2+ more seeds per condition.
