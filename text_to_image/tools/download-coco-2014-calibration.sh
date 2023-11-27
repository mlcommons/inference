#!/bin/bash

: "${DOWNLOAD_PATH:=../coco2014}"

while [ "$1" != "" ]; do
    case $1 in
        -d | --download-path )       shift
                                     DOWNLOAD_PATH=$1
                                     ;;
    esac
done


python3 coco_calibration.py \
    --dataset-dir ${DOWNLOAD_PATH}
