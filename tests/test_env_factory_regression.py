"""
Minimal regression tests for env_factory.build_env() and RollbackEvaluationCallbackTopK, added
per IMPROVEMENT_PLAN_2_WRAPPERS_AND_DEPENDENCIES.md Task 8. The two `slow` tests below would
each have caught one of the two runtime bugs found and fixed during this repo's initial setup
(see CODEBASE_GUIDE.md Sec 5.2b):

- test_build_env_observation_and_reward_contract would have caught the
  `KeyError: 'action_spaces'` crash that made every single environment construction fail.
- test_rollback_callback_save_reload_predicts_consistently would have caught the
  `TypeError: cannot pickle '_thread.lock' object` crash in model.save() caused by the old
  model.train monkey-patch (see IMPROVEMENT_PLAN_2_WRAPPERS_AND_DEPENDENCIES.md Task 3).

The `slow` tests launch the real Godot binary and take on the order of a minute each. Run
`pytest -m "not slow"` for a fast loop, or `pytest -m slow` (or plain `pytest`) to include them
-- they require the environment to be set up per HOW_TO_RUN.md (venv + built bin/B_ACE_v0.1.exe).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT, REPO_ROOT / "rl_files"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from b_ace_py.utils import load_b_ace_config
from b_ace_py.contracts import EXPECTED_OBSERVATION_LABELS


def _load_bace_config():
    cfg = load_b_ace_config(str(REPO_ROOT / "b_ace_py" / "SimpleExample_B_ACE_config.json"))
    cfg.setdefault("EnvConfig", {})["env_path"] = str(REPO_ROOT / "bin" / "B_ACE_v0.1.exe")
    return cfg


def test_rollback_callback_requires_skippable_train_ppo(tmp_path):
    """
    RollbackEvaluationCallbackTopK must reject a model that isn't a SkippableTrainPPO instance
    at _init_callback time (fail fast) rather than silently doing nothing, the way a monkeypatch
    of model.train would have (see IMPROVEMENT_PLAN_2_WRAPPERS_AND_DEPENDENCIES.md Task 3). No
    Godot instance needed -- this only exercises the callback's own guard clause.
    """
    from rollback_eval_callback import RollbackEvaluationCallbackTopK

    class NotASkippableTrainPPO:
        pass

    callback = RollbackEvaluationCallbackTopK(
        eval_env=None,
        log_dir=tmp_path,
        skip_initial_eval=True,
        verbose=0,
    )
    callback.model = NotASkippableTrainPPO()

    with pytest.raises(TypeError):
        callback._init_callback()


@pytest.mark.slow
def test_build_env_observation_and_reward_contract(tmp_path):
    """
    Launches the real environment via env_factory.build_env(), resets, steps a fixed short
    sequence of actions, and asserts (a) observation shape matches the Task 6
    EXPECTED_OBSERVATION_LABELS contract and every value is finite, (b) reward is always a
    finite float.
    """
    from env_factory import build_env

    cfg = _load_bace_config()
    env = build_env(
        cfg,
        is_eval=True,
        log_dir_str=str(tmp_path),
        seed=123,
        use_enriched_obs=False,
        max_episode_steps=200,
    )
    try:
        obs, info = env.reset(seed=123)
        assert obs.shape == (len(EXPECTED_OBSERVATION_LABELS),)
        assert np.all(np.isfinite(obs))

        rng = np.random.default_rng(0)
        for _ in range(20):
            action = rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.all(np.isfinite(obs))
            assert np.isfinite(float(reward))
            if terminated or truncated:
                obs, info = env.reset(seed=123)
    finally:
        env.close()


@pytest.mark.slow
def test_rollback_callback_save_reload_predicts_consistently(tmp_path):
    """
    Trains for a small number of timesteps with RollbackEvaluationCallbackTopK attached (the
    same callback used in real training runs), confirms model.save() succeeds, and that the
    reloaded model predicts the same action for a fixed observation as the pre-save model.
    Also asserts 'train' never ends up in model.__dict__ (the concrete invariant from
    IMPROVEMENT_PLAN_2_WRAPPERS_AND_DEPENDENCIES.md Task 3 that failed before the fix).
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from env_factory import build_env
    from rollback_eval_callback import RollbackEvaluationCallbackTopK, SkippableTrainPPO

    cfg = _load_bace_config()
    n_steps = 64

    def make_train_env():
        return build_env(
            cfg, is_eval=False, log_dir_str=str(tmp_path), seed=7,
            use_enriched_obs=False, max_episode_steps=200,
        )

    def make_eval_env():
        return build_env(
            cfg, is_eval=True, log_dir_str=str(tmp_path / "eval"), seed=99,
            use_enriched_obs=False, max_episode_steps=200,
        )

    train_venv = VecNormalize(DummyVecEnv([make_train_env]))
    eval_venv = VecNormalize(DummyVecEnv([make_eval_env]), training=False)

    try:
        model = SkippableTrainPPO(
            policy="MlpPolicy",
            env=train_venv,
            n_steps=n_steps,
            batch_size=32,
            n_epochs=1,
            verbose=0,
            seed=42,
        )

        callback = RollbackEvaluationCallbackTopK(
            eval_env=eval_venv,
            eval_freq=10_000_000,  # never fires in this short test
            n_eval_episodes=1,
            log_dir=tmp_path,
            skip_initial_eval=True,
            verbose=0,
        )

        model.learn(total_timesteps=n_steps, callback=callback)

        assert "train" not in model.__dict__

        save_path = tmp_path / "model"
        model.save(save_path.as_posix())

        fixed_obs = np.zeros((1, train_venv.observation_space.shape[0]), dtype=np.float32)
        action_before, _ = model.predict(fixed_obs, deterministic=True)

        reloaded = PPO.load(save_path.as_posix(), device="cpu")
        action_after, _ = reloaded.predict(fixed_obs, deterministic=True)

        assert np.allclose(action_before, action_after)
    finally:
        train_venv.close()
        eval_venv.close()
