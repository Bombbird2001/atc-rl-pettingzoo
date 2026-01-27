import torch


class RolloutBuffer:
    def __init__(self, max_size, device: torch.device):
        self.max_size = max_size
        self.device = device
        self.obs = torch.Tensor().to(device)
        self.logprobs = torch.Tensor().to(device)
        self.actions = torch.Tensor().to(device)
        self.advantages = torch.Tensor().to(device)
        self.returns = torch.Tensor().to(device)
        self.values = torch.Tensor().to(device)

    @property
    def size(self):
        return self.obs.shape[0]

    @property
    def full(self):
        return self.size == self.max_size

    def reset(self):
        self.obs = torch.Tensor().to(self.device)
        self.logprobs = torch.Tensor().to(self.device)
        self.actions = torch.Tensor().to(self.device)
        self.advantages = torch.Tensor().to(self.device)
        self.returns = torch.Tensor().to(self.device)
        self.values = torch.Tensor().to(self.device)

    def add_data(self, obs, logprobs, actions, advantages, returns, values):
        # Adds data to the buffer; discards extra data that would have exceeded max_size
        data_length = obs.shape[0]
        assert data_length <= self.max_size
        assert data_length == logprobs.shape[0]
        assert data_length == actions.shape[0]
        assert data_length == advantages.shape[0]
        assert data_length == returns.shape[0]
        assert data_length == values.shape[0]

        capacity_remaining = self.max_size - data_length - self.obs.shape[0]
        if capacity_remaining >= 0:
            self.obs = torch.cat((self.obs, obs))
            self.logprobs = torch.cat((self.logprobs, logprobs))
            self.actions = torch.cat((self.actions, actions))
            self.advantages = torch.cat((self.advantages, advantages))
            self.returns = torch.cat((self.returns, returns))
            self.values = torch.cat((self.values, values))
        else:
            self.obs = torch.cat((self.obs, obs[:capacity_remaining]))
            self.logprobs = torch.cat((self.logprobs, logprobs[:capacity_remaining]))
            self.actions = torch.cat((self.actions, actions[:capacity_remaining]))
            self.advantages = torch.cat((self.advantages, advantages[:capacity_remaining]))
            self.returns = torch.cat((self.returns, returns[:capacity_remaining]))
            self.values = torch.cat((self.values, values[:capacity_remaining]))

    def get_data(self):
        return self.obs, self.logprobs, self.actions, self.advantages, self.returns, self.values


class GraphRolloutBuffer:
    def __init__(self, max_size, device: torch.device):
        self.max_size = max_size
        self.device = device
        self.obs = []
        self.logprobs = torch.Tensor().to(device)
        self.actions = torch.Tensor().to(device)
        self.advantages = torch.Tensor().to(device)
        self.returns = torch.Tensor().to(device)
        self.values = torch.Tensor().to(device)
        self.node_count = 0

    @property
    def size(self):
        return self.node_count

    @property
    def full(self):
        return self.size >= self.max_size

    def reset(self):
        self.obs = []
        self.logprobs = torch.Tensor().to(self.device)
        self.actions = torch.Tensor().to(self.device)
        self.advantages = torch.Tensor().to(self.device)
        self.returns = torch.Tensor().to(self.device)
        self.values = torch.Tensor().to(self.device)
        self.node_count = 0

    def add_data(self, obs, logprobs, actions, advantages, returns, values):
        if self.full:
            raise Exception("GraphRolloutBuffer is already full")

        # Adds data to the buffer; we allow the sum of node in all graphs to exceed the max size up to the last graph, if removing the last graph
        # would cause the total node count to fall below the max size
        graph_sizes = list(map(lambda x: x.x.shape[0], obs))
        data_length = sum(graph_sizes)
        assert data_length == logprobs.shape[0]
        assert data_length == actions.shape[0]
        assert data_length == advantages.shape[0]
        assert data_length == returns.shape[0]
        assert data_length == values.shape[0]

        capacity_remaining = self.max_size - data_length - self.node_count
        if capacity_remaining >= 0:
            self.obs.extend(obs)
            self.logprobs = torch.cat((self.logprobs, logprobs))
            self.actions = torch.cat((self.actions, actions))
            self.advantages = torch.cat((self.advantages, advantages))
            self.returns = torch.cat((self.returns, returns))
            self.values = torch.cat((self.values, values))
            self.node_count += data_length
        else:
            # Determine up to which index to add
            newly_added_nodes = 0
            for idx, graph in enumerate(obs):
                newly_added_nodes += graph.x.shape[0]
                if self.size + newly_added_nodes >= self.max_size:
                    break

            self.obs.extend(obs[:idx + 1])
            self.logprobs = torch.cat((self.logprobs, logprobs[:newly_added_nodes]))
            self.actions = torch.cat((self.actions, actions[:newly_added_nodes]))
            self.advantages = torch.cat((self.advantages, advantages[:newly_added_nodes]))
            self.returns = torch.cat((self.returns, returns[:newly_added_nodes]))
            self.values = torch.cat((self.values, values[:newly_added_nodes]))
            self.node_count += newly_added_nodes

    def get_data(self):
        return self.obs, self.logprobs, self.actions, self.advantages, self.returns, self.values