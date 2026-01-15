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
