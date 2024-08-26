from pathlib import Path

from morphers.dataset.base import DatasetProvider
from morphers.io_utils import load_json
from morphers.utils import convert_to_int_mapping
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms as t
from torchvision.datasets import MNIST

# TODO in experiments dataloaders always yield the same images regardless of seed, check this out.


def get_full_mnist_dataset(data_root: Path) -> Dataset:
    full_mnist = ConcatDataset(
        [
            MNIST(
                data_root,
                train=True,
                download=True,
                transform=t.Compose([t.ToTensor()]),
            ),
            MNIST(
                data_root,
                train=False,
                download=True,
                transform=t.Compose([t.ToTensor()]),
            ),
        ]
    )
    return full_mnist


class MNISTMappedEntitiesDataset(Dataset):
    def __init__(self, data_root: Path, indices_map: dict[int, int]) -> None:
        self._dataset = get_full_mnist_dataset(data_root)
        self.indices = list(indices_map.keys())
        self.indices_mapped = list(indices_map.values())
        super().__init__()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int):
        return self._dataset[self.indices[index]][0], self._dataset[self.indices_mapped[index]][0]


class MNISTMappedEntitiesProvider(DatasetProvider):
    def __init__(self, data_root: str, train_indices_map_path: str, val_indices_map_path: str, test_indices_map_path: str):
        self.data_root = data_root
        self.train_indices_map = convert_to_int_mapping(load_json(train_indices_map_path))
        self.val_indices_map = convert_to_int_mapping(load_json(val_indices_map_path))
        self.test_indices_map = convert_to_int_mapping(load_json(test_indices_map_path))

    def train_dataset(self):
        return MNISTMappedEntitiesDataset(self.data_root, self.train_indices_map)

    def val_dataset(self):
        return MNISTMappedEntitiesDataset(self.data_root, self.val_indices_map)

    def test_dataset(self):
        return MNISTMappedEntitiesDataset(self.data_root, self.test_indices_map)
