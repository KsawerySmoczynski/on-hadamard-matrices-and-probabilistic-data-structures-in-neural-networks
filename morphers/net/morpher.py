import torch
from morphers.hadamard import get_closest_hadamard_size, hadamard_transform_torch
from morphers.net.interface import Net
from morphers.net.layers import InverterLayer
from torch import nn


class MorpherNet(Net):
    def __init__(self, n_input: int, n_features: int, hidden_module: nn.Module, head: nn.Module | None = None):
        super().__init__()
        self.n_input = n_input
        self.n_features = n_features
        hadamard_size = get_closest_hadamard_size(n_input, floor=False)

        self.padding_size = hadamard_size - n_input
        self.n_patches = hadamard_size // n_features
        self.hidden_module = hidden_module
        self.inv = InverterLayer(n_features, self.n_patches)
        self.patch_weights = torch.ones(n_features, self.n_patches)
        self.head = head

    def get_input_shape(self) -> tuple[int]:
        return (self.n_input,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (0, self.padding_size), value=0.0)
        x = hadamard_transform_torch(x, True)

        aggregate = torch.zeros((x.shape[0], self.n_features), device=x.device, dtype=x.dtype)
        with torch.no_grad():
            normalizing_term = self.patch_weights.sum(-1)
        for p_idx in range(self.n_patches):
            selection = slice(p_idx * self.n_features, (p_idx + 1) * self.n_features)
            patch = x[:, selection]
            aggregate += self.inv(patch, p_idx) * (self.patch_weights[:, p_idx] / normalizing_term)[None, :]
        aggregate = self.hidden_module(aggregate)
        if self.head:
            return self.head(aggregate)

        # Decoder branch
        patches = []
        for p_idx in range(self.n_patches):
            patch = aggregate * self.patch_weights[None, :, p_idx] # verify this weighting scheme
            patch = self.inv(patch, p_idx, inverse=True)
            patches.append(patch)
        x = torch.cat(patches, dim=1)
        x = hadamard_transform_torch(x, True)
        if self.padding_size != 0:
            x = x[:, : -self.padding_size]
        return x


class WeightedAutomorpherNet(MorpherNet):
    def __init__(self, n_input: int, n_features: int, hidden_module: nn.Module = None, head: nn.Module = None):
        super().__init__(n_input, n_features, hidden_module)
        # Trainable patch weights
        self.patch_weights = nn.ParameterList([nn.Parameter(torch.ones(n_features)) for _ in range(self.n_patches)])
