from morphers.dataset.base import DatasetProvider
from torchvision import transforms as t
from torchvision.datasets import MNIST as _MNIST

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class MNISTProvider(DatasetProvider):
    def __init__(self, data_root: str):
        self.data_root = data_root
        # self.mean = 0.0 #MNIST_MEAN
        # self.std = MNIST_STD

    def train_dataset(self):
        return _MNIST(
            self.data_root,
            train=True,
            download=True,
            # transform=t.Compose([t.ToTensor(), t.Normalize((self.mean,), (self.std,))]),
            transform=t.Compose([t.ToTensor()]),
        )

    def val_dataset(self):
        return _MNIST(
            self.data_root,
            train=False,
            download=True,
            # transform=t.Compose([t.ToTensor(), t.Normalize((self.mean,), (self.std,))]),
            transform=t.Compose([t.ToTensor()]),
        )

    def test_dataset(self):
        raise NotImplementedError
