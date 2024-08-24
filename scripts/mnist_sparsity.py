from pathlib import Path

from morphers.dataset.MNIST_mapped_entities import get_full_mnist_dataset

datasets_root = Path("datasets")
FULL_MNIST = get_full_mnist_dataset(datasets_root)

n_features = 0
n_zeros = 0
for img, label in iter(FULL_MNIST):
    n_features += img.numel()
    n_zeros += (img == 0).sum()

print(f"MNIST is {(n_zeros/n_features * 100).item():.2f}% sparse")
