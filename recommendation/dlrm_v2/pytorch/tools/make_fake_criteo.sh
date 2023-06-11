#!/bin/bash

OUTPUT_DIR=`pwd`/fake_criteo/
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

set -x # echo the next command
python quickgen.py --num-samples=204800 --output-dir=$OUTPUT_DIR --num-multihot-features "3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1"


