import torch
from enum import Enum
from models.gnn_net import GNNNet, GNNGineNet, GNNGatV2Net
from models.model_utils import layer_init
from torch.nn import Linear, Module
from torch.distributions import Categorical
from torch_geometric.data import Data


class ModelRegistry(Enum):
    GNNGatV2Net = GNNGatV2Net
    GNNGineNet = GNNGineNet

    @classmethod
    def get_model(cls, class_name: str) -> type[Module]:
        return cls[class_name].value

    @classmethod
    def model_is_gnn(cls, class_name: str) -> bool:
        return issubclass(cls[class_name].value, GNNNet)


class Agent(Module):
    def __init__(self, envs, latent_net_class: type[Module], freeze_action=False, freeze_value=False):
        super().__init__()
        self.actor_latent = latent_net_class()
        self.action_space_dims = envs.single_action_space.nvec
        self.actor = layer_init(Linear(64, sum(self.action_space_dims)), std=0.01)
        self.alt_dim: int = self.action_space_dims[1]

        if freeze_action:
            for param in self.actor_latent.parameters():
                param.requires_grad = False
            for param in self.actor.parameters():
                param.requires_grad = False

        self.value_latent = latent_net_class()
        self.critic = layer_init(Linear(64, 1), std=1)

        if freeze_value:
            for param in self.value_latent.parameters():
                param.requires_grad = False
            for param in self.critic.parameters():
                param.requires_grad = False

    def get_value(self, x):
        x = x.clone()
        return self.critic(self.value_latent(x))

    def get_action_and_value(self, x, alt_action_mask_int: torch.Tensor=None, action=None, use_mode: bool = False):
        # Input shape: (..., feature_size)
        # Modified to support MultiDiscrete
        x = x.clone()
        hidden = self.actor_latent(x)
        logits = self.actor(hidden)
        hdg_logits, alt_logits, spd_logits = logits.split(tuple(self.action_space_dims), dim=-1)
        if alt_action_mask_int is not None:
            mask_base = 2 ** torch.arange(self.alt_dim - 1, -1, -1)
            offset = (mask_base.bitwise_and(alt_action_mask_int) == 0) * torch.finfo(torch.float32).min
            if alt_logits.shape != offset.shape:
                raise ValueError(f"Expected alt_logits shape {alt_logits.shape} to match offset {offset.shape}")
            alt_logits = alt_logits + offset
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
        return action, log_prob, entropy, self.critic(value_hidden)

    def rep_string(self):
        return f"-----------------\n{self.actor_latent}\n{self.actor}\n-----------------\n{self.value_latent}\n{self.critic}\n-----------------"


class GNNAgent(Module):
    def __init__(
            self, envs, node_feature_count: int, edge_feature_count: int, latent_net_class: type[GNNNet],
            freeze_action=False, freeze_value=False
    ):
        super().__init__()
        self.actor_latent = latent_net_class(node_feature_count, edge_feature_count)
        self.action_space_dims = envs.single_action_space.nvec
        self.actor = layer_init(Linear(128, sum(self.action_space_dims)), std=0.01)
        self.alt_dim: int = self.action_space_dims[1]

        if freeze_action:
            for param in self.actor_latent.parameters():
                param.requires_grad = False
            for param in self.actor.parameters():
                param.requires_grad = False

        # Centralized GNN critic (MAPPO)
        self.value_latent = latent_net_class(node_feature_count, edge_feature_count)
        self.critic = layer_init(Linear(128, 1), std=1)

        if freeze_value:
            for param in self.value_latent.parameters():
                param.requires_grad = False
            for param in self.critic.parameters():
                param.requires_grad = False

    def _pad_actions(self, actions: torch.Tensor, batch: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        padded_actions = []
        for env_idx in range(mask.shape[0]):
            env_mask = mask[env_idx]
            actions_no_pad = actions[:,batch == env_idx]
            padded_output = torch.zeros((env_mask.shape[0], self.action_space_dims.shape[0]), dtype=torch.long, device=env_mask.device)
            padded_output[env_mask.bool()] = actions_no_pad.transpose(0, 1)
            padded_actions.append(padded_output)
        # Returned shape: (num_envs, num_aircraft, 3)
        return torch.stack(padded_actions, dim=0)

    def _pad_scalar(self, values: torch.Tensor, batch: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        padded_values = []
        for env_idx in range(mask.shape[0]):
            env_mask = mask[env_idx]
            values_no_pad = values[batch == env_idx]
            padded_output = torch.zeros(env_mask.shape[0], device=env_mask.device)
            padded_output[env_mask.bool()] = values_no_pad
            padded_values.append(padded_output)
        return torch.stack(padded_values, dim=0)

    def get_value(self, x: Data, mask: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        value_hidden = self.value_latent(x.x, x.edge_index, x.edge_attr)
        values = self.critic(value_hidden).squeeze(-1)
        return self._pad_scalar(values, x.batch, mask)

    def get_action_and_value(
            self, x: Data, mask: torch.Tensor, alt_action_mask_int: torch.Tensor=None,
            action=None, use_mode: bool=False
    ):
        # Input shape: (..., feature_size)
        # Modified to support MultiDiscrete
        x = x.clone()
        hidden = self.actor_latent(x.x, x.edge_index, x.edge_attr)
        logits = self.actor(hidden)
        hdg_logits, alt_logits, spd_logits = logits.split(tuple(self.action_space_dims), dim=-1)
        if alt_action_mask_int is not None:
            mask_base = 2 ** torch.arange(self.alt_dim - 1, -1, -1)
            offset = (mask_base.bitwise_and(alt_action_mask_int) == 0) * torch.finfo(torch.float32).min
            if alt_logits.shape != offset.shape:
                raise ValueError(f"Expected alt_logits shape {alt_logits.shape} to match offset {offset.shape}")
            alt_logits = alt_logits + offset
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
        values = self.critic(value_hidden).squeeze(-1)
        return (
            self._pad_actions(action, x.batch, mask),
            self._pad_scalar(log_prob, x.batch, mask),
            entropy, self._pad_scalar(values, x.batch, mask)
        )

    def rep_string(self):
        return f"-----------------\n{self.actor_latent}\n{self.actor}\n-----------------\n{self.value_latent}\n{self.critic}\n-----------------"
