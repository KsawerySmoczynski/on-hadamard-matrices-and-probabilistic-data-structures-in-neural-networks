import importlib
from copy import deepcopy
from typing import Dict, List, MutableMapping, Tuple, Union

from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from torch.utils.tensorboard.writer import SummaryWriter


def get_tb_logger(loggers: List[Logger]) -> SummaryWriter:
    for logger in loggers:
        if isinstance(logger, TensorBoardLogger):
            return logger.experiment
    raise ValueError("TensorBoard Logger not found")


def recursive_dict_merge(d1: Dict, d2: Dict) -> Dict:
    for k, v in d1.items():
        if k in d2:
            if all(isinstance(e, MutableMapping) for e in (v, d2[k])):
                d2[k] = recursive_dict_merge(v, d2[k])
    d3 = d1.copy()
    d3.update(d2)
    return d3


def initialize_object(object_dict: Dict):
    if "init_args" in object_dict:
        if isinstance(object_dict["init_args"], dict):
            for k, v in object_dict["init_args"].items():
                if isinstance(v, dict) and "class_path" in v:
                    object_dict["init_args"][k] = initialize_object(v)

    class_path = object_dict["class_path"]
    init_args = object_dict["init_args"] if "init_args" in object_dict else {}
    parts = class_path.split(".")
    module, net_class = ".".join(parts[:-1]), parts[-1]
    package = class_path.split(".")[0]
    module = importlib.import_module(module, package)
    cls = getattr(module, net_class)
    if callable(cls):
        return cls(**init_args) if isinstance(init_args, dict) else cls(*init_args)
    return cls


def traverse_config_and_initialize(iterable: Union[Dict, List, Tuple]):
    inpt = deepcopy(iterable)
    if isinstance(inpt, dict) and "class_path" in inpt:
        if "init_args" in inpt:
            inpt["init_args"] = traverse_config_and_initialize(inpt["init_args"])
        return initialize_object(inpt)
    elif isinstance(inpt, dict):
        for k, v in inpt.items():
            inpt[k] = traverse_config_and_initialize(v)
        return inpt
    elif isinstance(inpt, (list, tuple)):
        items = []
        for item in inpt:
            items.append(traverse_config_and_initialize(item))
        return items
    else:
        return inpt


def get_configs(config: Dict) -> Dict:
    # TODO change to proper handling with defaults and interpretable errors etc
    if "fit" in config:
        config = config["fit"]
    assert "model" in config
    assert "data" in config
    assert "seed_everything" in config
    assert "trainer" in config
    config = traverse_config_and_initialize(config)
    model_config = config["model"]
    data_config = config["data"]
    trainer_config = config["trainer"]
    seed = config["seed_everything"]
    return model_config, data_config, trainer_config, seed
