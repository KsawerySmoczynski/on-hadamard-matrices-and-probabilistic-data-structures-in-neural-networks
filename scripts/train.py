import argparse

from lightning_fabric import seed_everything
from morphers.datamodule import MyLightningDataModule
from morphers.logger import get_logger
from morphers.module import MNISTIdentityModule, MNISTMappingModule
from morphers.utils import get_configs
from pytorch_lightning.trainer import Trainer

from morphers.io_utils import load_config, save_config
from morphers.module.mnist_classification import MNISTClassificationModule

# Tensorboard
# ^((?!.*STH.*).)*$ regex to exclude

EXPERIMENTS = ["mnist_classification", "mnist_identity", "mnist_mapping"]


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
    else:
        raise NotImplementedError(f"{args.experiment} experiment not implemented")
    datamodule = MyLightningDataModule(**data_config)
    logger = get_logger(module, data_config["dataset_provider"], model_config["net"], args.experiment_name)

    trainer = Trainer(**trainer_config, logger=logger)
    trainer.fit(model=module, datamodule=datamodule)
    save_config(config, logger.log_dir)