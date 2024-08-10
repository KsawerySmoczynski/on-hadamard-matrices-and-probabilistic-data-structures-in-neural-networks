import numpy as np
import torch
from scipy.linalg import hadamard


def get_closest_hadamard_size(input_size: int, floor=True):
    round = np.ceil if not floor else np.floor
    return int(2 ** (round(np.log2(input_size))))


def get_coalesce_permutation(n: int, coalesce_factor: int) -> tuple[np.ndarray, np.ndarray]:
    if n // coalesce_factor != n / coalesce_factor:
        raise ValueError(f"Coalesce factor has to divide n, got n:{n} and coalesce_factor:{coalesce_factor}")
    vector = np.arange(n)
    permutation_vector = vector.reshape(-1, coalesce_factor).T.reshape(-1)
    temp = np.c_[permutation_vector, vector]
    inverse_permutation_vector = temp[temp[:, 0].argsort()][:, 1]
    return permutation_vector, inverse_permutation_vector


def get_bitflip_vector(size: int) -> torch.Tensor:
    if size % 2 != 0:
        raise ValueError("Size of bitflip vector isnt even")
    bitflip_vector = torch.ones(size, dtype=torch.int8)
    selector = torch.randperm(size)[: size // 2]
    bitflip_vector[selector] *= -1
    return bitflip_vector


def get_walsh_permutation(size: int) -> np.ndarray:
    a = hadamard(size)
    sequency_values = (np.roll(a, 1) - a) != 0
    sequency_values[:, 0] = 0
    sequency_values = sequency_values.sum(1)
    permutation = np.c_[sequency_values, np.arange(len(sequency_values))]
    permutation = permutation[permutation[:, 0].argsort()]

    return permutation[:, 1]


def hadamard_transform_torch(u, normalize=False):
    """
    Taken from: https://github.com/HazyResearch/structured-nets

    Multiply H_n @ u where H_n is the Hadamard matrix of dimension n x n.
    n must be a power of 2.
    Parameters:
        u: Tensor of shape (..., n)
        normalize: if True, divide the result by 2^{m/2} where m = log_2(n).
    Returns:
        product: Tensor of shape (..., n)
    """
    _, n = u.shape
    m = int(np.log2(n))
    assert n == 1 << m, "n must be a power of 2"
    x = u[..., np.newaxis]
    for d in range(m)[::-1]:
        x = torch.cat((x[..., ::2, :] + x[..., 1::2, :], x[..., ::2, :] - x[..., 1::2, :]), dim=-1)
    return x.squeeze(-2) / 2 ** (m / 2) if normalize else x.squeeze(-2)
