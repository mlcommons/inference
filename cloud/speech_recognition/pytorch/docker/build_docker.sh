#!/bin/bash

#ARG	VALUES							HELP
#${1} = {cpu, gpu}						Build docker without or with CUDA 
#${2} = {<nothing>, .py3} 				Build either the docker image for training and inference or the python 3 docker image for ONNX conversion

if [ "${1}" = "cpu" ]
then
	docker build . --rm -f Dockerfile${2} -t deepspeech2${2}:cpu
elif [ "${1}" = "gpu" ]
then
	nvidia-docker build . --rm -f Dockerfile${2} -t deepspeech2${2}:gpu
fi