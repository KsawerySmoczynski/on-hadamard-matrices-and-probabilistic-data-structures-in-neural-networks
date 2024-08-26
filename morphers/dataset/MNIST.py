import numpy as np
from morphers.dataset.base import DatasetProvider
from torch.utils.data import Subset
from torchvision import transforms as t
from torchvision.datasets import MNIST

MNIST_SIZE = 60_000
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class MNISTProvider(DatasetProvider):
    def __init__(self, data_root: str):
        self.data_root = data_root
        dataset_indices = np.arange(MNIST_SIZE)
        train_size = int(MNIST_SIZE * 0.8)
        self.train_indices = np.random.choice(dataset_indices, size=train_size, replace=False).tolist()
        self.val_indices = list(set(dataset_indices.tolist()) - set(self.train_indices))

    def train_dataset(self):
        return Subset(
            MNIST(
                self.data_root,
                train=True,
                download=True,
                transform=t.Compose([t.ToTensor()]),
            ),
            indices=self.train_indices,
        )

    def val_dataset(self):
        return Subset(
            MNIST(
                self.data_root,
                train=True,
                download=True,
                transform=t.Compose([t.ToTensor()]),
            ),
            indices=self.val_indices,
        )

    def test_dataset(self):
        return MNIST(
            self.data_root,
            train=False,
            download=True,
            transform=t.Compose([t.ToTensor()]),
        )
