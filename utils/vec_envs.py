import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple, Dict, Any


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
    def early_reset(self, env_idx: int, seed: int | None = None):
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
        self.early_reset_data: List[Optional[Tuple[Dict[str, np.ndarray], Dict[str, Any]]]] = [None for _ in envs]

    def _dict_obs_to_np(self, dict_obs: dict) -> np.ndarray:
        # Additional one/zero at the end for masking
        return np.vstack([(np.hstack((dict_obs[agentId], np.ones(1))) if agentId in dict_obs else np.zeros(self.single_observation_space.shape)) for agentId in self.agent_ordering])

    def _dict_reward_to_np(self, dict_reward: dict) -> np.ndarray:
        return np.array([(dict_reward[agentId] if agentId in dict_reward else 0) for agentId in self.agent_ordering])

    def _dict_termination_to_np(self, dict_termination: dict) -> np.ndarray:
        return np.array([(dict_termination[agentId] if agentId in dict_termination else 0) for agentId in self.agent_ordering])

    def _dict_truncation_to_np(self, dict_truncation: dict) -> np.ndarray:
        return np.array([(dict_truncation[agentId] if agentId in dict_truncation else 0) for agentId in self.agent_ordering])

    def reset(self, seed: int | None = None):
        observations = []
        infos = []

        for idx, env in enumerate(self.envs):
            if self.early_reset_data[idx] is not None:
                obs, info = self.early_reset_data[idx]
            else:
                obs, info = env.reset(seed=seed)
            obs_np = self._dict_obs_to_np(obs)
            observations.append(obs_np)
            infos.append(info)

        self.early_reset_data = [None for _ in self.envs]

        return np.stack(observations), infos

    def early_reset(self, env_idx: int, seed: int | None = None):
        if self.early_reset_data[env_idx] is not None:
            return

        obs, info = self.envs[env_idx].reset(seed=seed)
        self.early_reset_data[env_idx] = (obs, info)

    def step(self, action: np.ndarray):
        # Expects shape of (num_env, AIRCRAFT_COUNT, action_dimension + 1)

        if action.shape[0] != len(self.envs):
            raise ValueError(f"action.shape[0] ({action.shape[0]}) != len(self.envs) ({len(self.envs)})")

        observations = []
        rewards = []
        terminations = []
        truncations = []
        infos = []

        for idx, env in enumerate(self.envs):
            if self.early_reset_data[idx] is not None:
                # Already terminated early, return all 0s
                obs_np = np.zeros((len(self.agent_ordering), ) + self.single_observation_space.shape)
                reward_np = np.zeros(len(self.agent_ordering))
                termination_np = np.zeros(len(self.agent_ordering))
                truncation_np = np.zeros(len(self.agent_ordering))
                info = {}
            else:
                obs, reward, termination, truncation, info = env.step(action[idx])
                obs_np = self._dict_obs_to_np(obs)
                reward_np = self._dict_reward_to_np(reward)
                termination_np = self._dict_termination_to_np(termination)
                truncation_np = self._dict_truncation_to_np(truncation)
            observations.append(obs_np)
            rewards.append(reward_np)
            terminations.append(termination_np)
            truncations.append(truncation_np)
            infos.append(info)

        return np.stack(observations), np.vstack(rewards), np.vstack(terminations), np.array(truncations), infos

    def close(self):
        for env in self.envs:
            env.close()

    @classmethod
    def make_vec_env(cls, env_count: int, make_env_fn: Callable, **kwargs):
        envs = [make_env_fn(env_id=env_id, **kwargs) for env_id in range(env_count)]

        return SequentialVecEnv(envs)