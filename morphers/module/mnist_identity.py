from typing import Any, List

import torch
import torchmetrics
from morphers.dataset.MNIST import MNIST_MEAN, MNIST_STD
from morphers.logger import log_tb_imgs
from morphers.module.utils import get_optimizer
from morphers.net.interface import Net
from morphers.utils import get_tb_logger
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import nn


class MNISTIdentityModule(LightningModule):
    def __init__(self, net: Net, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = net
        self.input_shape = net.get_input_shape()
        self.val_loss = torchmetrics.SumMetric()

    def setup(self, stage: str):
        if self.global_rank == 0:
            get_tb_logger(self.loggers).add_scalar("num_params", nn.utils.parameters_to_vector(self.net.parameters()).shape[0])

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.net(*args, **kwargs)

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], *self.input_shape)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx):
        x, _ = batch
        output = self.forward(self._reshape_input(x))
        loss = nn.functional.mse_loss(output.view(x.shape), x)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx):
        x, _ = batch
        img_shape = x.shape
        output = self(self._reshape_input(x))
        loss = nn.functional.mse_loss(output.view(x.shape), x, reduction="sum")
        self.val_loss.update(loss.detach())
        if ((batch_idx + 1) % 20 == 0) and (((self.current_epoch + 1) % 10 == 0) or self.current_epoch == 0):
            x_image = x.view(img_shape)  # * MNIST_STD + MNIST_MEAN
            output_image = output.view(img_shape)  # * MNIST_STD + MNIST_MEAN
            log_tb_imgs(
                get_tb_logger(self.trainer.loggers),
                self.current_epoch,
                (x_image[:10], output_image[:10]),
            )
        return loss

    def validation_epoch_end(self, outputs: List[torch.Tensor | dict[str, Any]] | List[List[torch.Tensor | dict[str, Any]]]) -> None:
        self.log("val_loss", self.val_loss.compute(), on_step=False, on_epoch=True, reduce_fx="sum")
        self.val_loss.reset()

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        return get_optimizer(self.net.parameters())
