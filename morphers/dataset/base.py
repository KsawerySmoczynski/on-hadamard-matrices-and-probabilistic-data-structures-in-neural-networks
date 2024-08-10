import abc


class DatasetProvider(abc.ABC):
    @abc.abstractmethod
    def train_dataset(self):
        raise NotImplementedError

    @abc.abstractmethod
    def val_dataset(self):
        raise NotImplementedError

    @abc.abstractmethod
    def test_dataset(self):
        raise NotImplementedError
