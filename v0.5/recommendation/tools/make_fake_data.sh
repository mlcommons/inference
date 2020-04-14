#!/bin/bash



OUTPUT_DIR=`pwd`/fake_data/$name
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo $OUTPUT_DIR

python ./tools/quickgen.py --num-samples=4000 --num-dense-features=13  \
        --num-sparse-features=”4-3-2” --num-targets=1 --profile=kaggle \
        --num-days=23 --numpy-rand-seed=123 --output-dir=$OUTPUT_DIR

