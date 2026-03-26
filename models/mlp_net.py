import torch.nn.functional as F
from models.agent_nets import AgentNet
from models.model_utils import layer_init
from torch.nn import Linear


class MLPNet(AgentNet):
    def __init__(self, node_feature_count: int):
        super().__init__()
        self.linear1 = layer_init(Linear(node_feature_count, 32))
        self.linear2 = layer_init(Linear(32, 64))

    def forward(self, x):
        h = self.linear1(x)
        h = F.relu(h)
        h = self.linear2(h)
        h = F.relu(h)

        return h
