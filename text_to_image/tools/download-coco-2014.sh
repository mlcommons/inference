#!/bin/bash

: "${DOWNLOAD_PATH:=../coco2014}"
: "${MAX_IMAGES:=5000}"
: "${NUM_WORKERS:=1}"

while [ "$1" != "" ]; do
    case $1 in
        -d | --download-path )       shift
                                     DOWNLOAD_PATH=$1
                                     ;;
    esac
    case $1 in
        -m | --max-images  )        shift
                                      MAX_IMAGES=$1
                                      ;;
    esac
    case $1 in
        -n | --num-workers  )        shift
                                      NUM_WORKERS=$1
                                      ;;
    esac
    shift
done

if [ -z ${MAX_IMAGES} ];
then
    python3 coco.py \
        --dataset-dir ${DOWNLOAD_PATH} \
        --num-workers ${NUM_WORKERS}
else
    python3 coco.py \
        --dataset-dir ${DOWNLOAD_PATH} \
        --max-images ${MAX_IMAGES} \
        --num-workers ${NUM_WORKERS}
fi