import torch
import torch.nn as nn


def init_layer(layer: nn.Linear) -> nn.Linear:
    nn.init.kaiming_normal_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class Policy(nn.Module):
    def __init__(self, input_dim: int, action_space: int, hidden_dim: int = 128):
        super(Policy, self).__init__()
        self.input_dim = input_dim
        self.action_space = action_space
        self.hidden_dim = hidden_dim
        self.model = nn.Sequential(
            init_layer(nn.Linear(input_dim, hidden_dim)),
            nn.ReLU(),
            init_layer(nn.Linear(hidden_dim, action_space)),
            nn.Softmax(dim=-1),
        )

    def forward(self, state):
        return self.model(state)


class Critic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super(Critic, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.model = nn.Sequential(
            init_layer(nn.Linear(input_dim, hidden_dim)),
            nn.ReLU(),
            init_layer(nn.Linear(hidden_dim, 1)),
        )

    def forward(self, state):
        return self.model(state)


if __name__ == "__main__":
    policy = Policy(input_dim=10, action_space=2)
    input = torch.randn(10)
    with torch.no_grad():
        output = policy(input)
    print(output)
