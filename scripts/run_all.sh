#!/bin/bash -e
function export_arg(){
    ARGNAME=$1
    ARG=$2
    if [ -z "${ARG}" ]; then
        echo "${ARGNAME} not provided"
        exit 0
    fi
    export ${ARGNAME}=${ARG}
}

export_arg EXPERIMENT_NAME "$1"
DATA_CONFIG_PATH="configs/data_configs/01-MNIST.yaml"
HIDDEN_LAYERS_DIR="configs/model_configs/hidden"
SEEDS=(17 79 53 19 37)
export IMG_DIM="[28,28]"
export INPUT_DIM=784

export HIDDEN_DIM=256
export COALESCE_FACTOR=7 #7 #for 4 - 128 hidden dim, for 7 - 256 hidden dim
# Identity experiments
MODEL_CONFIGS_DIR="configs/model_configs/01-encoding-reco"
(
    for seed in ${SEEDS}; do
        export SEED="${seed}";
        HIDDEN_LAYER_CONFIG="identity.yaml";
        bash scripts/run_experiments.sh mnist_identity "${EXPERIMENT_NAME}" "${DATA_CONFIG_PATH}" "${MODEL_CONFIGS_DIR}" "${HIDDEN_LAYERS_DIR}/${HIDDEN_LAYER_CONFIG}";
        HIDDEN_LAYER_CONFIG="one_hidden.yaml";
        bash scripts/run_experiments.sh mnist_identity "${EXPERIMENT_NAME}" "${DATA_CONFIG_PATH}" "${MODEL_CONFIGS_DIR}" "${HIDDEN_LAYERS_DIR}/${HIDDEN_LAYER_CONFIG}";
        HIDDEN_LAYER_CONFIG="three_hidden.yaml";
        bash scripts/run_experiments.sh mnist_identity "${EXPERIMENT_NAME}" "${DATA_CONFIG_PATH}" "${MODEL_CONFIGS_DIR}" "${HIDDEN_LAYERS_DIR}/${HIDDEN_LAYER_CONFIG}";
    done
) | tee /dev/tty

# Classification experiments
MODEL_CONFIGS_DIR="configs/model_configs/02-classification"
(
    for seed in ${SEEDS}; do
        export SEED="${seed}";
        HIDDEN_LAYER_CONFIG="identity.yaml";
        bash scripts/run_experiments.sh mnist_classification "${EXPERIMENT_NAME}" "${DATA_CONFIG_PATH}" "${MODEL_CONFIGS_DIR}" "${HIDDEN_LAYERS_DIR}/${HIDDEN_LAYER_CONFIG}";
        HIDDEN_LAYER_CONFIG="one_hidden.yaml";
        bash scripts/run_experiments.sh mnist_classification "${EXPERIMENT_NAME}" "${DATA_CONFIG_PATH}" "${MODEL_CONFIGS_DIR}" "${HIDDEN_LAYERS_DIR}/${HIDDEN_LAYER_CONFIG}";
        HIDDEN_LAYER_CONFIG="three_hidden.yaml";
        bash scripts/run_experiments.sh mnist_classification "${EXPERIMENT_NAME}" "${DATA_CONFIG_PATH}" "${MODEL_CONFIGS_DIR}" "${HIDDEN_LAYERS_DIR}/${HIDDEN_LAYER_CONFIG}";
    done
) | tee /dev/tty

# Mapping experiments
MODEL_CONFIGS_DIR="configs/model_configs/01-encoding-reco"
MAPPING_PATH="configs/model_configs/other/shifted_mapping.yaml"
(
    for seed in ${SEEDS}; do
        export SEED="${seed}";
        HIDDEN_LAYER_CONFIG="identity.yaml";
        bash scripts/run_experiments.sh mnist_mapping "${EXPERIMENT_NAME}" "${DATA_CONFIG_PATH}" "${MODEL_CONFIGS_DIR}" "${HIDDEN_LAYERS_DIR}/${HIDDEN_LAYER_CONFIG}" "${MAPPING_PATH}";
        HIDDEN_LAYER_CONFIG="one_hidden.yaml";
        bash scripts/run_experiments.sh mnist_mapping "${EXPERIMENT_NAME}" "${DATA_CONFIG_PATH}" "${MODEL_CONFIGS_DIR}" "${HIDDEN_LAYERS_DIR}/${HIDDEN_LAYER_CONFIG}" "${MAPPING_PATH}";
        HIDDEN_LAYER_CONFIG="three_hidden.yaml";
        bash scripts/run_experiments.sh mnist_mapping "${EXPERIMENT_NAME}" "${DATA_CONFIG_PATH}" "${MODEL_CONFIGS_DIR}" "${HIDDEN_LAYERS_DIR}/${HIDDEN_LAYER_CONFIG}" "${MAPPING_PATH}";
    done
) | tee /dev/tty
