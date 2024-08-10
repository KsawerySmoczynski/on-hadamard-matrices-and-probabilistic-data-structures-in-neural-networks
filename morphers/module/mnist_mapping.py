import pickle
from collections import OrderedDict
from typing import Any

import torch
from morphers.dataset.MNIST import MNIST_MEAN, MNIST_STD
from morphers.logger import log_tb_imgs
from morphers.utils import get_tb_logger
from pytorch_lightning import LightningModule
from torch import nn

from morphers.module.utils import get_optimizer
from morphers.net.interface import Net


class MNISTMappingModule(LightningModule):
    def __init__(self, net: Net, targets_path: str, mapping: dict = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = net
        targets = pickle.load(open(targets_path, "rb"))
        mapping = mapping or {i: i for i in range(10)}
        mapping = OrderedDict(sorted(mapping.items(), key=lambda x: x[0]))
        self.mapping = nn.Parameter(torch.cat([targets[v] for _, v in mapping.items()], dim=0).unsqueeze(1))
        self.input_shape = net.get_input_shape()

    def setup(self, stage: str):
        if self.global_rank == 0:
            get_tb_logger(self.loggers).add_scalar("num_params", nn.utils.parameters_to_vector(self.net.parameters()).shape[0])

    def forward(self, *args: Any, **kwargs: Any):
        return self.net(*args, **kwargs)

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
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        if ((batch_idx + 1) % 20 == 0) and (((self.current_epoch + 1) % 10 == 0) or self.current_epoch == 0):
            N_IMGS = 10
            x_image = x[:N_IMGS]# * MNIST_STD + MNIST_MEAN
            output_image = output.view(x.shape)[:N_IMGS]# * MNIST_STD + MNIST_MEAN
            targets_image = targets[:N_IMGS]# * MNIST_STD + MNIST_MEAN
            log_tb_imgs(
                get_tb_logger(self.trainer.loggers),
                self.current_epoch,
                (x_image, output_image, targets_image),
            )
        return loss

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        return get_optimizer(self.net.parameters())
