#!/bin/bash

nvidia-docker build . --rm -f Dockerfile.gpu${1} -t ds2-cuda9cudnn${1}:gpu
