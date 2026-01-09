import numpy as np
from common.constants import AIRCRAFT_COUNT
from functools import lru_cache
from pettingzoo import ParallelEnv


AGENT_NAME_PREFIX = "agent_"


class TC2GymPettingZooEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "tc2_pettingzoo"}

    # Our environment will handle up to AIRCRAFT_COUNT aircraft simultaneously
    def __init__(self, gym_env, num_agents=AIRCRAFT_COUNT):
        super().__init__()
        self.gym_env = gym_env

        self.possible_agents = [f"{AGENT_NAME_PREFIX}{i}" for i in range(num_agents)]
        self.agents = []

        # Define Action and Observation spaces per agent
        # All agents share the same space structure as the underlying gym env
        self.observation_spaces = {
            agent: self.gym_env.observation_space for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: self.gym_env.action_space for agent in self.possible_agents
        }

    @lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    @staticmethod
    def _gym_observation_to_dict_observation(obs: np.ndarray) -> dict[str, np.ndarray]:
        if obs.shape[0] != AIRCRAFT_COUNT:
            raise ValueError(f"Expected {AIRCRAFT_COUNT} in obs, got {obs.shape[0]}")

        dict_obs = {}

        for idx, row in enumerate(obs):
            # Agent ID is located in last column, verify it is in ascending order
            if idx != row[-1]:
                raise ValueError(f"Expected ascending agent ID, got {obs[:,-1]}")

            if row[-2] == 0:
                # No aircraft for this index, ignore
                continue

            dict_obs[f"{AGENT_NAME_PREFIX}{idx}"] = row[:-1]

        return dict_obs

    @staticmethod
    def _gym_reward_to_dict_reward(reward: np.ndarray) -> dict[str, np.ndarray]:
        if reward.shape[0] != AIRCRAFT_COUNT:
            raise ValueError(f"Expected {AIRCRAFT_COUNT} in reward, got {reward.shape[0]}")

        dict_reward = {}

        for idx, row in enumerate(reward):
            if idx != row[-1]:
                raise ValueError(f"Expected ascending agent ID, got {reward[:,-1]}")

            if row[-2] == 0:
                continue

            dict_reward[f"{AGENT_NAME_PREFIX}{idx}"] = row[0]

        return dict_reward

    def reset(self, seed=None, options=None):
        # Contract with our gym_env: obs is sorted in ascending order by agent ID
        # And it is possible for an agent to not exist, which will be indicated by the 2nd last column in esch row
        obs, info = self.gym_env.reset(seed=seed, options=options)

        observations = self._gym_observation_to_dict_observation(obs)
        self.agents = list(observations.keys())
        infos = {agent: info for agent in self.agents}

        return observations, infos

    def step(self, actions: np.ndarray):
        # Observations and hence actions are in increasing agent ID
        # Actions shape is (AIRCRAFT_COUNT, 3), flatten
        combined_action = actions.flatten()

        obs, reward, terminated, truncated, info = self.gym_env.step(combined_action)

        # We broadcast the scalar values from the single-agent env to all agents
        observations = self._gym_observation_to_dict_observation(obs)
        rewards = self._gym_reward_to_dict_reward(reward)
        terminations = {agent: terminated for agent in self.possible_agents}
        truncations = {agent: truncated for agent in self.agents}
        infos = {agent: info for agent in self.possible_agents}

        # 5. Update the active agents list
        if terminated or truncated:
            self.agents = []
        else:
            self.agents = self.possible_agents[:]

        return observations, rewards, terminations, truncations, infos

    def render(self):
        return self.gym_env.render()

    def close(self):
        self.gym_env.close()