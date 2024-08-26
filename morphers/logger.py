from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import torch
from morphers.dataset.base import DatasetProvider
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter


def get_logger(datamodule_provider: LightningModule, dataset_provider: DatasetProvider, net: nn.Module, experiment_name: str) -> TensorBoardLogger:
    folder = Path("lightning_logs") / f"{datamodule_provider.__class__.__name__}_{dataset_provider.__class__.__name__}" / net.__class__.__name__
    if experiment_name:
        experiment_name = Path(experiment_name) / datetime.now().strftime("%Y-%m-%d/%H-%M")
    else:
        experiment_name = input("name the experiment: ")
    (folder / experiment_name).mkdir(parents=True, exist_ok=True)
    logger = TensorBoardLogger(save_dir=folder, name=experiment_name)
    return logger


def log_tb_imgs(logger: SummaryWriter, epoch: int, viz_batch: Tuple[torch.Tensor, torch.Tensor], caption: Optional[str] = None):
    if not caption:
        caption = f"e{epoch}"
    for img_idx, images in enumerate(zip(*viz_batch)):
        img = torch.cat(images, dim=-1)
        logger.add_image(f"{caption}/{img_idx}", img, 0)
