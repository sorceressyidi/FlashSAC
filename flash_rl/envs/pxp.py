"""env_type `pxp`: a physical-exploration environment (pxp.envs) as a Gymnasium vector env.

Built from the run's pxp_config.json (env, body, critic.inputs) written by `python -m pxp.run --write`. Same
autoreset semantics as this repo's own vector envs (SameStep): a finished world is reset inside step(), its last
observation is returned in info["final_obs"] and its final info in info["final_info"]. info["success"] and
info["heads"] (pxp.critic head signals) are arrays over worlds. Asymmetric observations follow this repo's prefix
convention: the actor reads the first info["actor_observation_size"] entries, so a critic-only block must be the
trailing block of the state vector (pxp's fatigue block is). No rendering: record envs are not supported.
"""
import json
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space


class PxpVectorEnv(VectorEnv):
    metadata: dict[str, Any] = {}

    def __init__(self, pxp_config: str, num_envs: int, seed: int, eval_mode: bool = False):
        from pxp.envs import make_env

        with open(pxp_config) as f:
            cfg = json.load(f)
        body = dict(cfg.get("body") or {})
        if eval_mode and body.get("fatigue"):   # evaluation worlds: same observation layout, fatigue off (as OGPO's eval env)
            body["fatigue"] = dict(body["fatigue"], enabled=False, log_dir=None)
        self._env = make_env(cfg["env"], num_envs, seed=seed, body=body)
        self.num_envs = int(num_envs)
        obs_shape = self._env.spec.obs["state"][0]
        self.single_observation_space = gym.spaces.Box(-np.inf, np.inf, obs_shape, np.float32)
        self.single_action_space = gym.spaces.Box(-1.0, 1.0, (self._env.spec.action_dim,), np.float32)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        actor_dim = obs_shape[0]
        inputs = (cfg.get("critic") or {}).get("inputs") or {}
        if inputs.get("actor") == ["state_without_fatigue"]:
            actor_dim -= self._env.fatigue_width
        self.env_info = {"actor_observation_size": (actor_dim,)}
        self.max_episode_steps = self._env.spec.max_steps
        self.head_spec = getattr(self._env, "head_spec", None)
        if self.head_spec:
            self.env_info["head_spec"] = [list(h) for h in self.head_spec]

    def reset(self, *, seed=None, options=None):
        obs = self._env.reset()["state"]
        return np.array(obs, dtype=np.float32), dict(self.env_info)

    def step(self, actions):
        obs, reward, terminated, truncated, info = self._env.step(np.asarray(actions, dtype=np.float32))
        obs = np.array(obs["state"], dtype=np.float32)
        terminated = np.asarray(terminated, dtype=bool)
        truncated = np.asarray(truncated, dtype=bool)
        out: dict[str, Any] = {"success": np.asarray(info["success"], dtype=np.float32)}
        if "heads" in info:
            out["heads"] = np.asarray(info["heads"], dtype=np.float32).reshape(self.num_envs, -1)
        done = terminated | truncated
        if done.any():
            out["final_obs"] = obs.copy()
            out["final_info"] = {"success": out["success"].copy()}
            fresh = np.asarray(self._env.reset(np.flatnonzero(done))["state"], dtype=np.float32)
            obs[done] = fresh[done]
        return obs, np.asarray(reward, dtype=np.float32), terminated, truncated, out

    def get_state(self):
        """Cross-episode env state (body fatigue, sessions, reset streams) for exact resume."""
        return self._env.get_state()

    def set_state(self, state):
        self._env.set_state(state)

    def render(self):
        raise NotImplementedError("pxp envs do not render; run with num_record_episodes=0")

    def close(self):
        self._env.close()


def make_pxp_envs(pxp_config: str, seed: int, num_train_envs: int, num_eval_envs: int, **_: Any):
    train_env = PxpVectorEnv(pxp_config, num_train_envs, seed)
    eval_env = PxpVectorEnv(pxp_config, num_eval_envs, seed + 1000, eval_mode=True)
    return train_env, eval_env, eval_env
