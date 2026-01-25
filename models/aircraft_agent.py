import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Sequential, ReLU, GELU, Linear, BatchNorm1d, Module
from torch.distributions import Categorical
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MLPAgent(Module):
    def __init__(self, envs):
        super().__init__()
        # Simple MLP
        self.actor_latent = Sequential(
            layer_init(Linear(18, 32)),
            ReLU(),
            layer_init(Linear(32, 64)),
            ReLU(),
        )
        self.action_space_dims = envs.single_action_space.nvec
        self.actor = layer_init(Linear(64, sum(self.action_space_dims)), std=0.01)

        # Current implementation is IPPO (no centralized critic)
        self.value_latent = Sequential(
            layer_init(Linear(18, 32)),
            ReLU(),
            layer_init(Linear(32, 64)),
            ReLU(),
        )
        self.critic = layer_init(Linear(64, 1), std=1)

    def get_value(self, x):
        x = x.clone()
        return self.critic(self.value_latent(x))

    def get_action_and_value(self, x, action=None, use_mode: bool = False):
        # Input shape: (..., feature_size)
        # Modified to support MultiDiscrete
        x = x.clone()
        hidden = self.actor_latent(x)
        logits = self.actor(hidden)
        hdg_logits, alt_logits, spd_logits = logits.split(tuple(self.action_space_dims), dim=-1)
        hdg_probs = Categorical(logits=hdg_logits)
        alt_probs = Categorical(logits=alt_logits)
        spd_probs = Categorical(logits=spd_logits)
        if action is None:
            if use_mode:
                action = torch.stack((hdg_probs.mode, alt_probs.mode, spd_probs.mode))
            else:
                action = torch.stack((hdg_probs.sample(), alt_probs.sample(), spd_probs.sample()))
        log_prob = hdg_probs.log_prob(action[0]) + alt_probs.log_prob(action[1]) + spd_probs.log_prob(action[2])
        entropy = hdg_probs.entropy() + alt_probs.entropy() + spd_probs.entropy()
        value_hidden = self.value_latent(x)
        print(value_hidden.shape)
        # TODO Global average pooling within each graph
        value_hidden = value_hidden.mean(dim=0)
        return action, log_prob, entropy, self.critic(value_hidden)


class GNNLatentNet(Module):
    def __init__(self, node_feature_count, edge_feature_count):
        super().__init__()
        self.node_feature_count = node_feature_count
        self.edge_feature_count = edge_feature_count

        nn1 = Sequential(
            layer_init(Linear(node_feature_count, 32)),
            BatchNorm1d(32),
            GELU(),
            layer_init(Linear(32, 32)),
        )
        nn2 = Sequential(
            layer_init(Linear(32, 64)),
            BatchNorm1d(64),
            GELU(),
            layer_init(Linear(64, 64)),
        )

        self.gine1 = GINEConv(nn1, edge_dim=edge_feature_count, train_eps=True)
        self.bn1 = BatchNorm1d(32)
        self.gine2 = GINEConv(nn2, edge_dim=edge_feature_count, train_eps=True)
        self.bn2 = BatchNorm1d(64)
        self.linear = layer_init(Linear(64, 64))

    def forward(self, x, edge_index, edge_attr):
        h = self.gine1(x, edge_index, edge_attr)
        h = self.bn1(h)
        h = F.gelu(h)
        h = self.gine2(h, edge_index, edge_attr)
        h = self.bn2(h)
        h = F.gelu(h)
        h = self.linear(h)

        return h


class GNNAgent(Module):
    def __init__(self, envs, node_feature_count: int, edge_feature_count: int):
        super().__init__()
        # TODO Maybe reduce to only local/neighbouring aircraft info instead of global
        self.actor_latent = GNNLatentNet(node_feature_count, edge_feature_count)
        self.action_space_dims = envs.single_action_space.nvec
        self.actor = layer_init(Linear(64, sum(self.action_space_dims)), std=0.01)

        # Centralized GNN critic (MAPPO)
        self.value_latent = GNNLatentNet(node_feature_count, edge_feature_count)
        self.critic = layer_init(Linear(64, 1), std=1)

    def _groupwise_mean(self, x, classes) -> torch.Tensor:
        classes = classes.view(classes.size(0), 1).expand(-1, x.size(1))

        unique_labels, labels_count = classes.unique(dim=0, return_counts=True)

        res = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, classes, x)
        return res / labels_count.float().unsqueeze(1)

    def get_value(self, x: Data):
        x = x.clone()
        value_hidden = self.value_latent(x.x, x.edge_index, x.edge_attr)
        values = self.critic(value_hidden)
        # TODO Pad values for non-existent aircraft
        return values

    def get_action_and_value(self, x: Data, action=None, use_mode: bool = False):
        # Input shape: (..., feature_size)
        # Modified to support MultiDiscrete
        x = x.clone()
        hidden = self.actor_latent(x.x, x.edge_index, x.edge_attr)
        logits = self.actor(hidden)
        hdg_logits, alt_logits, spd_logits = logits.split(tuple(self.action_space_dims), dim=-1)
        hdg_probs = Categorical(logits=hdg_logits)
        alt_probs = Categorical(logits=alt_logits)
        spd_probs = Categorical(logits=spd_logits)
        if action is None:
            if use_mode:
                action = torch.stack((hdg_probs.mode, alt_probs.mode, spd_probs.mode))
            else:
                action = torch.stack((hdg_probs.sample(), alt_probs.sample(), spd_probs.sample()))
        log_prob = hdg_probs.log_prob(action[0]) + alt_probs.log_prob(action[1]) + spd_probs.log_prob(action[2])
        entropy = hdg_probs.entropy() + alt_probs.entropy() + spd_probs.entropy()
        value_hidden = self.value_latent(x.x, x.edge_index, x.edge_attr)
        values = self.critic(value_hidden)
        # TODO Pad values for non-existent aircraft
        return action, log_prob, entropy, values
