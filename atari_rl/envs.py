"""Environment construction and Atari wrappers for the Atlantis experiments.

The observation pipeline uses the standard Gymnasium wrappers
(`AtariPreprocessing` + frame stacking). One optional domain-specific wrapper is
added:

    FireOnlyActionWrapper -- drops NOOP and exposes only the firing actions
    (FIRE / RIGHTFIRE / LEFTFIRE), resolved by action name at runtime.

Rewards are returned RAW by the environment. The learning-signal transform
(sign-clipping or scaling) is applied in the training loop, so reported episode
returns are always the true game score, comparable to the random baseline.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

# Importing ale_py registers the ALE/* environment ids.
import ale_py  # noqa: F401

try:  # Explicit registration for Gymnasium >= 1.0; harmless on 0.29.
    gym.register_envs(ale_py)
except AttributeError:
    pass


def _frame_stack(env: gym.Env, k: int = 4) -> gym.Env:
    """Stack the last ``k`` observations (Gymnasium 0.29 and 1.0 wrapper names)."""
    if hasattr(gym.wrappers, "FrameStackObservation"):
        return gym.wrappers.FrameStackObservation(env, stack_size=k)
    return gym.wrappers.FrameStack(env, num_stack=k)


class FireOnlyActionWrapper(gym.ActionWrapper):
    """Restrict the action space to the firing actions (drop NOOP).

    For Atlantis: FIRE, RIGHTFIRE, LEFTFIRE (k = 3).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        meanings = env.unwrapped.get_action_meanings()
        self._fire_actions = [i for i, m in enumerate(meanings) if "FIRE" in m]
        if not self._fire_actions:
            raise ValueError(f"No FIRE actions found in {meanings!r}")
        self.action_space = gym.spaces.Discrete(len(self._fire_actions))
        self.fire_action_meanings = [meanings[i] for i in self._fire_actions]

    def action(self, action: int) -> int:
        return self._fire_actions[int(action)]


def make_env(
    env_id: str = "ALE/Atlantis-v5",
    seed: int = 0,
    idx: int = 0,
    fire_only: bool = True,
    record_stats: bool = True,
    render_mode: str | None = None,
):
    """Return a thunk that builds a single wrapped environment.

    ``RecordEpisodeStatistics`` reports raw episode returns via
    ``info["episode"]["r"]`` even though the learner sees transformed rewards.
    """

    def thunk() -> gym.Env:
        # frameskip=1: AtariPreprocessing does the frame skipping itself.
        env = gym.make(env_id, frameskip=1, render_mode=render_mode)
        env = gym.wrappers.AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=4,
            screen_size=84,
            terminal_on_life_loss=False,
            grayscale_obs=True,
            scale_obs=False,
        )
        env = _frame_stack(env, 4)
        if fire_only:
            env = FireOnlyActionWrapper(env)
        if record_stats:
            env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env

    return thunk


def make_vector_env(
    env_id: str = "ALE/Atlantis-v5",
    num_envs: int = 8,
    seed: int = 0,
    fire_only: bool = True,
):
    """Build a ``SyncVectorEnv`` of ``num_envs`` Atlantis environments.

    Sync (not Async) for reliability on Windows; slightly slower.
    """
    envs = gym.vector.SyncVectorEnv(
        # record_stats=False on sub-envs: RecordEpisodeStatistics is attached to
        # the vector env below (infos["episode"]["r"][i], mask infos["_episode"][i]).
        # Sub-envs return RAW rewards; the reward transform happens in the training loop.
        [make_env(env_id, seed, idx, fire_only, record_stats=False) for idx in range(num_envs)]
    )
    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    return envs


def iter_episode_stats(infos: dict, num_envs: int):
    """Yield (raw_return, length) for every environment that finished this step.

    Handles both info layouts: top-level ``infos["episode"]`` arrays with an
    ``infos["_episode"]`` mask, or a per-env ``infos["final_info"]`` list.
    """
    ep = infos.get("episode")
    mask = infos.get("_episode")
    if isinstance(ep, dict) and mask is not None:
        for i in range(num_envs):
            if mask[i]:
                yield float(ep["r"][i]), int(ep["l"][i])
        return
    for info in infos.get("final_info", []) or []:
        if info and "episode" in info:
            yield float(info["episode"]["r"]), int(info["episode"]["l"])


def transform_reward(reward: np.ndarray, mode: str, scale: float = 0.01) -> np.ndarray:
    """Transform the raw environment reward into the learning signal.

    - ``"clip"``: DeepMind-style sign clipping to {-1, 0, +1} (default).
    - ``"scale"``: multiply by ``scale``; keeps relative reward magnitudes.
    - ``"none"``: raw rewards (e.g. with Pop-Art; otherwise the value loss dominates).
    """
    if mode == "clip":
        return np.sign(reward)
    if mode == "scale":
        return reward * scale
    if mode == "none":
        return reward
    raise ValueError(f"Unknown reward mode: {mode!r}")
