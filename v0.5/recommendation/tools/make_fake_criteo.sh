#!/bin/bash

OUTPUT_DIR=`pwd`/fake_criteo/
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

if [[ $# == 1 ]]; then
    QUICKGEN_PROFILE=$1
else
    echo "usage: $0 [kaggle|terabyte0875|terabyte]"
    exit 1
fi

set -x # echo the next command
python quickgen.py --num-samples=4096 --profile=$QUICKGEN_PROFILE --output-dir=$OUTPUT_DIR


