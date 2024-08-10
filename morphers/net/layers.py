import math

import torch
from torch import nn
from torch.nn import functional as F


class InverterLayer(nn.Module):
    __constants__ = ["n_features"]
    n_features: int
    n_hidden: int
    m_patches: int
    weight: torch.Tensor

    def __init__(self, n_features: int, m_patches: int, reduction_ratio: int = 4, bias: bool = False, device=None, dtype=None):
        super().__init__()
        self._factory_kwargs = {"device": device, "dtype": dtype}
        self.n_features = n_features
        self.n_hidden = n_features // reduction_ratio
        self.m_patches = m_patches

        self.weight_in = nn.Parameter(torch.empty(n_features, self.n_hidden, **self._factory_kwargs))
        self.patches_weights = nn.ParameterList([nn.Parameter(torch.empty(self.n_hidden, self.n_hidden, **self._factory_kwargs)) for _ in range(self.m_patches)])  # replace with identity?
        self.weight_out = nn.Parameter(torch.empty(self.n_hidden, self.n_features, **self._factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(n_features, **self._factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.selector = torch.arange(n_features)
        # self.ones = nn.Parameter(torch.ones(n_features, **self._factory_kwargs    ))  # This is apparently more stable
        self.ones = nn.ParameterList([nn.Parameter(torch.ones(n_features, **self._factory_kwargs)) for _ in range(self.m_patches)])
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight_in, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_out, a=math.sqrt(5))
        for patch_weight in self.patches_weights:
            nn.init.kaiming_uniform_(patch_weight, a=math.sqrt(5))
            with torch.no_grad():
                patch_weight[torch.arange(self.n_hidden), torch.arange(self.n_hidden)] = torch.ones(self.n_hidden, **self._factory_kwargs)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_in)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def __call__(self, x: torch.Tensor, patch_idx: int, inverse: bool = False) -> torch.Tensor:
        return self.forward(x, patch_idx, inverse)

    def forward(self, x: torch.Tensor, patch_idx: int, inverse: bool) -> torch.Tensor:
        weight = self.weight_in @ self.patches_weights[patch_idx] @ self.weight_out

        weight[self.selector, self.selector] = weight[self.selector, self.selector] + self.ones[patch_idx]

        if inverse:
            if self.bias is not None:
                x = x - self.bias[None, :]
            result = torch.linalg.solve(weight, x, left=False)
            return result
        result = F.linear(x, weight, self.bias)
        return result
