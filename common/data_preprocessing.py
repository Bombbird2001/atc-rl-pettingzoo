import numpy as np
import torch
from abc import ABC, abstractmethod
from common.constants import AIRCRAFT_COUNT, X_Y_SCALE_DOWN, ALT_SCALE_DOWN, HDG_BINS, ALT_BINS, SPD_BINS
from torch import Tensor
from torch_geometric.data import Data
from typing import Tuple


AC_FAMILY_MAPPING = {
    "B737": "B737NG",
    "B738": "B737NG",
    "B739": "B737NG",
    "A359": "A350",
    "A35K": "A350",
    "B752": "B757",
    "B772": "B777",
    "B773": "B777",
    "B77W": "B777",
    "B77L": "B777",
    "B788": "B787",
    "B789": "B787",
    "B78X": "B787",
    "A319": "A320",
    "A320": "A320",
    "A321": "A320",
    "A21N": "A320neo",
    "A20N": "A320neo",
    "B744": "B747",
    "B748": "B748",
    "A388": "A380",
    "A333": "A330",
    "A332": "A330",
    "A339": "A330neo",
    "B733": "B737Classic",
    "B734": "B737Classic",
    "B763": "B767",
    "B38M": "B737MAX",
    "E290": "E2",
    "E295": "E2",
    "GLF4": "G450",
    "GLF6": "G650",
    "GLEX": "GLEX",
    "FA8X": "FA8X",
    "CL60": "CL60",
    None: "Unknown",
}


RECAT_MAPPING = {
    "B737": "D",
    "B738": "D",
    "B739": "D",
    "A359": "B",
    "A35K": "B",
    "B752": "C",
    "B772": "B",
    "B773": "B",
    "B77W": "B",
    "B77L": "B",
    "B788": "B",
    "B789": "B",
    "B78X": "B",
    "A319": "D",
    "A320": "D",
    "A321": "D",
    "A21N": "D",
    "A20N": "D",
    "B744": "B",
    "B748": "B",
    "A388": "A",
    "A333": "B",
    "A332": "B",
    "A339": "B",
    "B733": "E",
    "B734": "E",
    "B763": "C",
    "B38M": "D",
    "E290": "D",
    "E295": "D",
    "GLF4": "E",
    "GLF6": "E",
    "GLEX": "E",
    "FA8X": "E",
    "CL60": "E",
    None: "Unknown",
}


class DataProcessor(ABC):
    @abstractmethod
    def preprocess_data(self, obs: torch.Tensor):
        raise NotImplementedError()

    @abstractmethod
    def postprocess_data(self, **kwargs):
        raise NotImplementedError()


class TransformerProcessor(DataProcessor):
    def preprocess_data(self, obs: torch.Tensor) -> Tuple[Tensor, Tensor]:
        obs = obs.reshape((1, AIRCRAFT_COUNT, -1))

        return obs[:,:,:-1], obs[:,:,-1]

    def postprocess_data(self, action: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
        action = action.squeeze(0)[attention_mask.squeeze(0).to(torch.int32) == 1]
        action = torch.hstack((action[:,:72].argmax(dim=1).unsqueeze(-1), action[:,72:])).numpy()
        action[:,0] = action[:,0]
        action[:,1] = np.round(action[:,1] * 16)
        action[:,2] = np.round(action[:,2] * 10 + 22)
        action = np.hstack((action, action[:,3:6].any(axis=1, keepdims=True)))
        action = np.vstack((action, np.zeros((AIRCRAFT_COUNT - action.shape[0], action.shape[1]))))

        return action.reshape(1, -1).astype(np.int32)


class GNNProcessor(DataProcessor):
    def preprocess_data(self, obs: torch.Tensor) -> Data:
        obs = obs[obs[:,-1] == 1,:-1]
        # print(obs.shape)

        node_count = obs.shape[0]

        if node_count == 0:
            return Data(x=torch.Tensor(obs).to(torch.float32), edge_index=torch.empty(2, 0, dtype=torch.int32), edge_attr=torch.empty(0, 2))

        # edge_index = torch.asarray([(i, i) for i in range(node_count)])
        edge_index = torch.asarray([(i, j) for j in range(node_count) for i in range(node_count)])

        # ["ias", "track_rate", "x", "y", "combined_alt", "combined_alt_rate", "track_x", "track_y", "prev_cleared_hdg_x", "prev_cleared_hdg_y",
        # "prev_cleared_alt", "prev_cleared_ias"] + [f"aircraft_type_{j}" for j in range(aircraft_category_count)] + mask
        edge_pos_0 = obs[edge_index[:,0]][:,[2, 3]]
        edge_pos_1 = obs[edge_index[:,1]][:,[2, 3]]
        edge_v_0 = obs[edge_index[:,0]][:,[6, 7]]
        edge_v_1 = obs[edge_index[:,1]][:,[6, 7]]
        additional_range = np.array([1500, -1500]) / ALT_SCALE_DOWN
        edge_alt_0 = torch.hstack((
            obs[edge_index[:,0]][:,[4, 8]].max(dim=1, keepdim=True).values,
            obs[edge_index[:,0]][:,[4, 8]].min(dim=1, keepdim=True).values
        )) + additional_range
        edge_alt_1 = torch.hstack((
            obs[edge_index[:,1]][:,[4, 8]].max(dim=1, keepdim=True).values,
            obs[edge_index[:,1]][:,[4, 8]].min(dim=1, keepdim=True).values)
        ) + additional_range
        delta_pos = edge_pos_0 - edge_pos_1
        v_sum = edge_v_1 - edge_v_0
        alt_overlap = ((edge_alt_0[:,1] <= edge_alt_1[:,0]) & (edge_alt_0[:,0] >= edge_alt_1[:,1]))

        # Put distance, closure rate in edge_attr
        # Closure rate is defined as (pos2 - pos1) dot (v1 - v2) / norm(pos2 - pos1)
        pos_dist = torch.linalg.norm(delta_pos, axis=1)
        within_15nm = torch.Tensor(pos_dist <= 15 / X_Y_SCALE_DOWN).to(torch.bool)
        selected_edges = alt_overlap & within_15nm
        # print(selected_edges)
        edge_attr = np.vstack((
            pos_dist / np.sqrt(8),
            # Divide function call to handle when elements of pos_dist == 0
            np.divide(np.vecdot(delta_pos, v_sum), pos_dist, out=np.zeros_like(pos_dist), where=pos_dist != 0) / 2
        )).transpose()
        edge_attr = torch.Tensor(edge_attr)[selected_edges].to(torch.float32)
        edge_index = edge_index[selected_edges]
        edge_index = edge_index.transpose(0, 1)
        # print(x)
        # print(y)
        # print(edge_index)
        # print(edge_attr)
        return Data(x=torch.Tensor(obs).to(torch.float32), edge_index=edge_index, edge_attr=edge_attr)

    def postprocess_data_multi_aircraft(self, action: torch.Tensor) -> np.ndarray:
        # print(action)
        action = torch.hstack((action[:,:72].argmax(dim=1).unsqueeze(-1), action[:,72:74], action[:,74:77].sigmoid() >= 0.5)).numpy()
        action[:,0] = action[:,0]
        action[:,1] = np.round(action[:,1] * 16).clip(2, 15)
        action[:,2] = np.round(action[:,2] * 10 + 22).clip(16, 25)
        action = np.hstack((action[:,:3], action[:,3:6].any(axis=1, keepdims=True)))
        action = np.vstack((action, np.zeros((AIRCRAFT_COUNT - action.shape[0], action.shape[1]))))

        return action.reshape(1, -1).astype(np.int32)

    def postprocess_data(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # We want to return raw logits for each row, and the max logit for hdg/alt/spd
        # clearance changed for each of the (up to) 15 aircraft
        class_logits = torch.hstack((torch.zeros(1), action[:,HDG_BINS + ALT_BINS + SPD_BINS:].max(dim=1).values))

        return class_logits, action[:,:HDG_BINS + ALT_BINS + SPD_BINS]
