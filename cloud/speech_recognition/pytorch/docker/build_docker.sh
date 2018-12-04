#!/bin/bash

#ARG	VALUES							HELP
#${1} = {<nothing>, .py3} 				Build either the training/inference docker image or the python 3 docker image for ONNX conversion

docker build . --rm -f Dockerfile.gpu${1} -t ds2-cuda9cudnn${1}:gpu
