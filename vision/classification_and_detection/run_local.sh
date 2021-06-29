#!/bin/bash

source ./run_common.sh

common_opt="--mlperf_conf ../../mlperf.conf"
dataset="--dataset-path $DATA_DIR"
OUTPUT_DIR=`pwd`/output/$name
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

PYTHON=python

if [ "x${CK_ENV_COMPILER_PYTHON_FILE}" != "x" ] ; then
  PYTHON=${CK_ENV_COMPILER_PYTHON_FILE}
fi

${PYTHON} python/main.py --profile $profile $common_opt --model $model_path $dataset \
    --output $OUTPUT_DIR $EXTRA_OPS $@
