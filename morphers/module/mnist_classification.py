from typing import Any

import torch
from morphers.utils import get_tb_logger
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import Accuracy

from morphers.module.utils import get_optimizer
from morphers.net.interface import Net


class MNISTClassificationModule(LightningModule):
    def __init__(self, net: Net, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = net
        self.accuracy = Accuracy(num_classes=10, task="multiclass")
        self.input_shape = net.get_input_shape()

    def setup(self, stage: str):
        if self.global_rank == 0:
            get_tb_logger(self.loggers).add_scalar("num_params", nn.utils.parameters_to_vector(self.net.parameters()).shape[0])

    def forward(self, *args: Any, **kwargs: Any):
        return self.net(*args, **kwargs)

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], *self.input_shape)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx):
        x, ids = batch
        output = self.forward(self._reshape_input(x))
        loss = nn.functional.cross_entropy(output, ids)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, ids = batch
        output = self.forward(self._reshape_input(x))
        loss = nn.functional.cross_entropy(output, ids)
        self.accuracy(output, ids)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log("accuracy", self.accuracy.compute(), on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        return get_optimizer(self.net.parameters())
