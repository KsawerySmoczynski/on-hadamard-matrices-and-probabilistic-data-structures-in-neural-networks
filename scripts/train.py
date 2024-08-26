import argparse
import shutil
from pathlib import Path

from lightning_fabric import seed_everything
from morphers.datamodule import MyLightningDataModule
from morphers.io_utils import load_config, save_config
from morphers.logger import get_logger
from morphers.module import MNISTIdentityModule, MNISTMappingModule
from morphers.module.mnist_classification import MNISTClassificationModule
from morphers.module.mnist_entity_mapping import MNISTEntityMappingModule
from morphers.utils import get_configs
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer import Trainer

# Tensorboard
# ^((?!.*STH.*).)*$ regex to exclude

EXPERIMENTS = ["mnist_classification", "mnist_identity", "mnist_mapping", "mnist_entity_mapping"]


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=EXPERIMENTS)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--experiment-name", default="")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    config = load_config(args.configs)
    model_config, data_config, trainer_config, seed = get_configs(config)

    seed_everything(seed)
    if args.experiment == "mnist_classification":
        module = MNISTClassificationModule(**model_config)
    elif args.experiment == "mnist_identity":
        module = MNISTIdentityModule(**model_config)
    elif args.experiment == "mnist_mapping":
        module = MNISTMappingModule(**model_config)
    elif args.experiment == "mnist_entity_mapping":
        module = MNISTEntityMappingModule(**model_config)
    else:
        raise NotImplementedError(f"{args.experiment} experiment not implemented")
    datamodule = MyLightningDataModule(**data_config)
    logger = get_logger(module, data_config["dataset_provider"], model_config["net"], args.experiment_name)
    save_config(config, logger.log_dir)

    checkpoint_callback = ModelCheckpoint(monitor=module.metric_to_monitor, filename=f"{{epoch}}-{{{module.metric_to_monitor}:.4f}}")

    trainer = Trainer(**trainer_config, logger=logger, callbacks=[checkpoint_callback])
    trainer.fit(model=module, datamodule=datamodule)
    trainer.test(model=module, ckpt_path="best", datamodule=datamodule)
    shutil.rmtree(str(Path(trainer.ckpt_path).parent))
