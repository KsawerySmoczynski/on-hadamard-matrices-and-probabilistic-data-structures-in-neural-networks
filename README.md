# sparse-inverter

To reproduce MNIST Identity experiments run:
```bash
EXPERIMENT="mnist_identity"
EXPERIMENT_NAME="default"
DATA_CONFIG_PATH="configs/data_configs/01-MNIST.yaml"
MODEL_CONFIGS_DIR="configs/model_configs"
HIDDEN_LAYER_CONFIG_PATH="${MODEL_CONFIGS_DIR}/hidden/one_hidden.yaml"
bash -e scripts/run_experiments.sh "${EXPERIMENT}" "${EXPERIMENT_NAME}" "${DATA_CONFIG_PATH}" "${MODEL_CONFIGS_DIR}" "${HIDDEN_LAYER_CONFIG_PATH}"
```

To reproduce MNIST Mapping experiments run:

```bash
EXPERIMENT="mnist_mapping"
# For easy mapping
EXPERIMENT_NAME="easy_mapping"
MAPPING_PATH="${MODEL_CONFIGS_DIR}/other/easy_mapping.yaml"
bash -e scripts/run_experiments.sh "${EXPERIMENT}" "${EXPERIMENT_NAME}"  "${DATA_CONFIG_PATH}" "${MODEL_CONFIGS_DIR}" "${HIDDEN_LAYER_CONFIG_PATH}" "${MAPPING_PATH}"
# For shifted mapping
EXPERIMENT_NAME="shifted_mapping"
MAPPING_PATH="${MODEL_CONFIGS_DIR}/other/shifted_mapping.yaml"
bash -e scripts/run_experiments.sh "${EXPERIMENT}" "${EXPERIMENT_NAME}" "${DATA_CONFIG_PATH}" "${MODEL_CONFIGS_DIR}" "${HIDDEN_LAYER_CONFIG_PATH}" "${MAPPING_PATH}"
```
