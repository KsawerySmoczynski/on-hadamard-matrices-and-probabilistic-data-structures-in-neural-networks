from functools import reduce
from os.path import expandvars
from pathlib import Path
from typing import List, Union

import yaml
from morphers.utils import recursive_dict_merge


def load_config(config_paths: List[str]) -> dict:
    if isinstance(config_paths, str):
        config_paths = [config_paths]
    configs = []
    for path in config_paths:
        with open(path, "r") as f:
            config = expandvars(f.read())
        configs.append(yaml.load(config, yaml.Loader))
    config = reduce(recursive_dict_merge, configs)
    return config


def save_config(config: dict, path: Union[str, Path]) -> None:
    with (Path(path) / "config.yaml").open("w") as f:
        yaml.dump(config, f)
