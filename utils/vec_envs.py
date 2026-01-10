import numpy as np
from abc import ABC, abstractmethod
from typing import Callable


class BaseVecEnv(ABC):
    @property
    def single_observation_space(self):
        return self.envs[0].single_observation_space

    @property
    def single_action_space(self):
        return self.envs[0].single_action_space

    @property
    def is_vector_env(self):
        return True

    @abstractmethod
    def reset(self, seed: int | None = None):
        raise NotImplementedError

    @abstractmethod
    def step(self, action: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError


class SequentialVecEnv(BaseVecEnv):
    # Basic VecEnv that runs the environment actions in sequence
    def __init__(self, envs):
        self.envs = envs
        self.agent_ordering = sorted(self.envs[0].possible_agents)

    def reset(self, seed: int | None = None):
        observations = []
        infos = []

        for env in self.envs:
            obs, info = env.reset(seed=seed)
            # Additional one/zero at the end for masking
            obs_np = np.vstack([(np.hstack((obs[agentId], np.ones(1))) if agentId in obs else np.zeros(self.single_observation_space.shape)) for agentId in self.agent_ordering])
            observations.append(obs_np)
            infos.append(info)

        return np.stack(observations), infos

    def step(self, action: np.ndarray):
        # Expects shape of (num_env, AIRCRAFT_COUNT, action_dimension + 1)

        if action.shape[0] != len(self.envs):
            raise ValueError(f"action.shape[0] ({action.shape[0]}) != len(self.envs) ({len(self.envs)})")

        for idx, env in enumerate(self.envs):
            env.step(action[idx])

    def close(self):
        for env in self.envs:
            env.close()

    @classmethod
    def make_vec_env(cls, env_count: int, make_env_fn: Callable, **kwargs):
        envs = [make_env_fn(env_id=env_id, **kwargs) for env_id in range(env_count)]

        return SequentialVecEnv(envs)