import torch
from torch import nn
from torch.optim import AdamW

epsilon = 1e-10

MATRIX_ORDER = 128
PRIVATE_DIM = MATRIX_ORDER // 4

ABC = nn.Sequential(nn.Linear(MATRIX_ORDER, PRIVATE_DIM, bias=False), nn.Linear(PRIVATE_DIM, PRIVATE_DIM, bias=False), nn.Linear(PRIVATE_DIM, MATRIX_ORDER, bias=False))


class IdentityDecomposition(nn.Module):
    def __init__(self, order=MATRIX_ORDER, private_dim=PRIVATE_DIM):
        super().__init__()
        self.A = nn.Parameter(torch.randn(order, private_dim) * (2 / torch.sqrt(torch.tensor(order))))
        self.B = nn.Parameter(torch.randn(private_dim, private_dim) * (2 / torch.sqrt(torch.tensor(private_dim))))
        self.C = nn.Parameter(torch.randn(private_dim, order) * (2 / torch.sqrt(torch.tensor(private_dim))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.A @ self.B @ self.C)[None, ...] - x


net = IdentityDecomposition(MATRIX_ORDER, PRIVATE_DIM)
# optim = AdamW(ABC.parameters(), lr=0.0001, weight_decay=0.01)
optim = AdamW(net.parameters(), lr=0.0001, weight_decay=0.001)

BATCH_SIZE = 512
identity = torch.stack([torch.eye(MATRIX_ORDER) for _ in range(BATCH_SIZE)], dim=0)
x = identity

loss_fn = nn.MSELoss()

loss = float("inf")
i = 0
while loss > epsilon:
    # means = torch.randn(BATCH_SIZE * MATRIX_ORDER) * 100
    # variances = torch.abs(torch.randn(BATCH_SIZE * MATRIX_ORDER) * 10)
    # x = (torch.randn(BATCH_SIZE * MATRIX_ORDER) * variances + means).view(BATCH_SIZE, MATRIX_ORDER)
    # x = (torch.randn(BATCH_SIZE * MATRIX_ORDER)).view(BATCH_SIZE, MATRIX_ORDER)
    # y_hat = ABC(x)
    y_hat = net(x)
    loss = torch.sqrt(loss_fn(y_hat[None, ...], x[None, ...]))
    loss.backward()
    optim.step()
    i += 1

# ABC[0]
