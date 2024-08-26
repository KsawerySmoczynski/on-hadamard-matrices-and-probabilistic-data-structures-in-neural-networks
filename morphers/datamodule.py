from dataclasses import asdict, dataclass
from typing import Union

from morphers.dataset.base import DatasetProvider
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


@dataclass
class DataLoaderParams:
    batch_size: int
    shuffle: bool
    num_workers: int
    prefetch_factor: Union[int, None]


class MyLightningDataModule(LightningDataModule):
    def __init__(self, dataset_provider: DatasetProvider, dataloader_params: dict):
        super().__init__()
        self.dataset_provider = dataset_provider
        self.dataloader_params = dataloader_params

    def train_dataloader(self):
        return DataLoader(self.dataset_provider.train_dataset(), **{**asdict(self.dataloader_params), "shuffle": True})

    def val_dataloader(self):
        return DataLoader(self.dataset_provider.val_dataset(), **{**asdict(self.dataloader_params), "shuffle": False})

    def test_dataloader(self):
        return DataLoader(self.dataset_provider.test_dataset(), **{**asdict(self.dataloader_params), "shuffle": False})
