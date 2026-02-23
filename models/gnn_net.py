import torch.nn.functional as F
from models.agent_nets import GNNNet
from models.model_utils import layer_init
from torch.nn import Sequential, Linear, LayerNorm, GELU
from torch_geometric.nn import GINEConv, GATv2Conv


class GNNGineNet(GNNNet):
    def __init__(self, node_feature_count: int, edge_feature_count: int):
        super().__init__()
        self.node_feature_count = node_feature_count
        self.edge_feature_count = edge_feature_count

        nn1 = Sequential(
            layer_init(Linear(node_feature_count, 32)),
            LayerNorm(32),
            GELU(),
            layer_init(Linear(32, 64)),
        )

        self.gine1 = GINEConv(nn1, edge_dim=edge_feature_count, train_eps=True)
        self.ln1 = LayerNorm(64)
        self.linear = layer_init(Linear(64, 64))
        self.ln2 = LayerNorm(64)

    def forward(self, x, edge_index, edge_attr):
        h = self.gine1(x, edge_index, edge_attr)
        h = F.gelu(h)
        h = self.ln1(h)
        h = self.linear(h)
        h = F.gelu(h)
        h = self.ln2(h)

        return h


class GNNGatV2Net(GNNNet):
    def __init__(self, node_feature_count, edge_feature_count):
        super().__init__()
        self.node_feature_count = node_feature_count
        self.edge_feature_count = edge_feature_count

        self.gat1 = GATv2Conv(node_feature_count, 32, heads=2, edge_dim=edge_feature_count)
        self.ln1 = LayerNorm(64)
        self.linear = layer_init(Linear(64, 64))
        self.ln2 = LayerNorm(64)

    def forward(self, x, edge_index, edge_attr):
        h = self.gat1(x, edge_index, edge_attr)
        h = F.gelu(h)
        h = self.ln1(h)
        h = self.linear(h)
        h = F.gelu(h)
        h = self.ln2(h)

        return h