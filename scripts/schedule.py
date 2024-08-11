import multiprocessing
from pathlib import Path
import subprocess
subprocess


CONFIGS_ROOT = Path("configs")
BASE_CONFIG_PATH = CONFIGS_ROOT / "base_cofig.yaml"
MODEL_CONFIGS_ROOT = CONFIGS_ROOT / "model_configs"
DATA_CONFIGS_ROOT = CONFIGS_ROOT / "data_configs"
HIDDEN_LAYERS_ROOT = CONFIGS_ROOT / "hidden"
MAPPING_CONFIG_PATH = MODEL_CONFIGS_ROOT / "other" / "shifted_mapping.yaml"


if  __name__ == "__main__":
    # Identity experiments
    env = {}
    for model_config_path in (MODEL_CONFIGS_ROOT / "01-encoding-reco").glob("*"):
        process = subprocess.Popen(shell=True)
        process = multiprocessing.Process(target=lambda x: print(x))

    # Classification experiments


    # Mapping experiments