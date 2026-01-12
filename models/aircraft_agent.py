import numpy as np
import torch
from torch import nn as nn
from torch.distributions import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MLPAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # Simple MLP
        # TODO Maybe try local/neighbouring aircraft info too (with GNN)
        self.network = nn.Sequential(
            layer_init(nn.Linear(18, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, 64)),
            nn.ReLU(),
        )
        self.action_space_dims = envs.single_action_space.nvec
        self.actor = layer_init(nn.Linear(64, sum(self.action_space_dims)), std=0.01)
        self.critic = layer_init(nn.Linear(64, 1), std=1)

    def get_value(self, x):
        # TODO Modify with GNN (centralized critic MAPPO)
        # Current implementation is IPPO (no centralized critic)
        x = x.clone()
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        # Input shape: (..., feature_size)
        # Modified to support MultiDiscrete
        x = x.clone()
        hidden = self.network(x)
        logits = self.actor(hidden)
        hdg_logits, alt_logits, spd_logits = logits.split(tuple(self.action_space_dims), dim=-1)
        hdg_probs = Categorical(logits=hdg_logits)
        alt_probs = Categorical(logits=alt_logits)
        spd_probs = Categorical(logits=spd_logits)
        if action is None:
            action = torch.stack((hdg_probs.sample(), alt_probs.sample(), spd_probs.sample()))
        log_prob = hdg_probs.log_prob(action[0]) + alt_probs.log_prob(action[1]) + spd_probs.log_prob(action[2])
        entropy = hdg_probs.entropy() + alt_probs.entropy() + spd_probs.entropy()
        return action, log_prob, entropy, self.critic(hidden)
