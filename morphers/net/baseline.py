import torch
from torch import nn

from morphers.net.interface import Net


class BaselineNet(Net):
    def __init__(self, n_input: int, n_features: int, hidden_module: nn.Module, head: nn.Module = None) -> None:
        super().__init__()
        self.l_in = nn.Linear(n_input, n_features)
        self.hidden_module = hidden_module
        self.l_out = head if head is not None else nn.Linear(n_features, n_input)

    def get_input_shape(self) -> tuple[int]:
        return (self.l_in.in_features, )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.leaky_relu(self.l_in(x))
        x = nn.functional.leaky_relu(self.hidden_module(x))
        x = self.l_out(x)
        return x
