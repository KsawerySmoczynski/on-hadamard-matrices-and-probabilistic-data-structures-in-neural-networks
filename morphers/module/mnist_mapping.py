import pickle
from collections import OrderedDict
from typing import Any

import torch
import torchmetrics
from morphers.logger import log_tb_imgs
from morphers.module.utils import get_optimizer
from morphers.net.interface import Net
from morphers.utils import get_tb_logger
from pytorch_lightning import LightningModule
from torch import nn


class MNISTMappingModule(LightningModule):
    def __init__(self, net: Net, targets_path: str, mapping: dict = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = net
        targets = pickle.load(open(targets_path, "rb"))
        mapping = mapping or {i: i for i in range(10)}
        mapping = OrderedDict(sorted(mapping.items(), key=lambda x: x[0]))
        self.mapping = nn.Parameter(torch.cat([targets[v] for _, v in mapping.items()], dim=0).unsqueeze(1))
        self.input_shape = net.get_input_shape()
        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()

    @property
    def metric_to_monitor(self):
        return "val_loss"

    def setup(self, stage: str):
        if self.global_rank == 0:
            get_tb_logger(self.loggers).add_scalar("num_params", nn.utils.parameters_to_vector(self.net.parameters()).shape[0])

    def forward(self, *args: Any, **kwargs: Any):
        return nn.functional.relu(self.net(*args, **kwargs))

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], *self.input_shape)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx):
        x, targets = batch
        targets = self.mapping[targets]
        output = self.forward(self._reshape_input(x))
        loss = nn.functional.mse_loss(output.view(targets.shape), targets)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, targets = batch
        targets = self.mapping[targets]
        output = self.forward(self._reshape_input(x))
        loss = nn.functional.mse_loss(output.view(targets.shape), targets)
        self.val_loss.update(loss.detach())
        if (batch_idx == 0) and (((self.current_epoch + 1) % 10 == 0) or self.current_epoch == 0):
            N_IMGS = 10
            x_image = x[:N_IMGS]
            output_image = output.view(x.shape)[:N_IMGS]
            targets_image = targets[:N_IMGS]
            log_tb_imgs(
                get_tb_logger(self.trainer.loggers),
                self.current_epoch,
                (x_image, output_image, targets_image),
            )
        return loss

    def validation_epoch_end(self, outputs: list[torch.Tensor | dict[str, Any]] | list[list[torch.Tensor | dict[str, Any]]]) -> None:
        self.log("val_loss", self.val_loss.compute(), on_step=False, on_epoch=True)
        self.val_loss.reset()

    def test_step(self, batch, batch_idx):
        x, targets = batch
        targets = self.mapping[targets]
        output = self.forward(self._reshape_input(x))
        loss = nn.functional.mse_loss(output.view(targets.shape), targets)
        self.test_loss.update(loss.detach())
        if batch_idx == 0:
            N_IMGS = 10
            x_image = x[:N_IMGS]
            output_image = output.view(x.shape)[:N_IMGS]
            targets_image = targets[:N_IMGS]
            log_tb_imgs(get_tb_logger(self.trainer.loggers), self.current_epoch, (x_image, output_image, targets_image), "test")

    def test_epoch_end(self, outputs: list[torch.Tensor | dict[str, Any]] | list[list[torch.Tensor | dict[str, Any]]]) -> None:
        self.log("test_loss", self.test_loss.compute(), on_step=False, on_epoch=True)
        self.test_loss.reset()

    def configure_optimizers(self):
        return get_optimizer(self.net.parameters())
