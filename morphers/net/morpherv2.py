import torch
from morphers.hadamard import get_closest_hadamard_size, get_coalesce_permutation, hadamard_transform_torch
from morphers.net.interface import Net
from morphers.net.layers import InverterLayer
from torch import nn


class MorpherNetv2(Net):
    def __init__(
        self,
        input_shape: tuple[int, int],
        hidden_module: nn.Module,
        head: nn.Module | None = None,
        coalesce_factor: int = 1,
    ):
        super().__init__()
        self.original_input_shape = input_shape

        self.coalesce_factor = coalesce_factor
        permutation_vector, inverse_permutation_vector = get_coalesce_permutation(input_shape[0], coalesce_factor)
        self.register_buffer("permutation", torch.tensor(permutation_vector), persistent=True)
        self.register_buffer("inverse_permutation", torch.tensor(inverse_permutation_vector), persistent=True)

        self.input_shape = (input_shape[0] // coalesce_factor, input_shape[1] * coalesce_factor)
        hadamard_size = get_closest_hadamard_size(self.input_shape[1], floor=False)
        self.padding_size = hadamard_size - self.input_shape[1]
        self.n_features = hadamard_size
        self.n_patches = self.input_shape[0]
        self.hidden_module = hidden_module
        self.inv = InverterLayer(self.n_features, self.n_patches)
        self.patch_weights = torch.ones(self.n_features, self.n_patches)
        self.head = head

    def get_input_shape(self) -> tuple[int]:
        return self.original_input_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orginal_shape = x.shape
        x = x[:, self.permutation, :].view(-1, *self.input_shape)
        x = nn.functional.pad(x, (0, self.padding_size), value=0.0)
        padded_shape = x.shape
        x = hadamard_transform_torch(x.view(-1, self.input_shape[-1] + self.padding_size), True).view(padded_shape[0], -1)
        with torch.no_grad():
            normalizing_term = self.patch_weights.sum(-1)
        aggregate = torch.zeros((x.shape[0], self.n_features), device=x.device, dtype=x.dtype)
        for p_idx in range(self.n_patches):
            selection = slice(p_idx * self.n_features, (p_idx + 1) * self.n_features)
            patch = x[:, selection]
            aggregate += self.inv(patch, p_idx) * (self.patch_weights[:, p_idx] / normalizing_term)[None, :]
        aggregate = self.hidden_module(aggregate / self.n_patches)
        if self.head:
            return self.head(aggregate)

        # Decoder branch
        patches = []
        for p_idx in range(self.n_patches):
            patch = self.inv(aggregate, p_idx, inverse=True) * self.patch_weights[None, :, p_idx]  # verify this weighting scheme
            patches.append(patch)
        x = torch.cat(patches, dim=1)
        x = hadamard_transform_torch(x.view(-1, self.input_shape[-1] + self.padding_size), True).view(padded_shape)
        if self.padding_size != 0:
            x = x[:, :, : -self.padding_size]
        return x.reshape(orginal_shape)[:, self.inverse_permutation, :]


class WeightedMorpherNetv2(MorpherNetv2):
    def __init__(self, input_shape: tuple[int, int], hidden_module: nn.Module, head: nn.Module | None = None, coalesce_factor: int = 1):
        super().__init__(input_shape, hidden_module, head, coalesce_factor)
        # Trainable patch weights
        self.patch_weights = nn.ParameterList([nn.Parameter(torch.ones(self.n_features)) for _ in range(self.n_patches)])
