#!/bin/bash -e
set -a
source .env-train
set +a

function export_arg(){
    ARGNAME=$1
    ARG=$2
    if [ -z "${ARG}" ]; then
        echo "${ARGNAME} not provided"
        exit 0
    fi
    export ${ARGNAME}=${ARG}
}

export_arg EXPERIMENT "$1"
export_arg EXPERIMENT_NAME "$2"
export_arg DATA_CONFIG_PATH "$3"
export_arg MODEL_CONFIGS_DIR "$4"
export_arg HIDDEN_LAYER_CONFIG_PATH "$5"
MAPPING_PATH="$6"


BASE_CONFIG="configs/base_config.yaml"
MODEL_CONFIGS=$(echo ${MODEL_CONFIGS_DIR}/*.yaml)
LAYERS="$(basename $HIDDEN_LAYER_CONFIG_PATH)"
EXPERIMENT_NAME="${EXPERIMENT_NAME}_${HIDDEN_DIM}_${LAYERS%%.*}"

echo $MODEL_CONFIGS | tr ' ' '\n' | xargs -S 2048 -P 4 -I@ bash -c "python scripts/train.py ${EXPERIMENT} --experiment-name ${EXPERIMENT_NAME}  --configs ${BASE_CONFIG} ${DATA_CONFIG_PATH} @ ${HIDDEN_LAYER_CONFIG_PATH} ${MAPPING_PATH}"
echo DONE!