#!/bin/bash
#ARG	VALUES							HELP
#${1} = {<nothing>, .py3} 				Use either the training/inference docker image or the python 3 docker image for ONNX conversion

nvidia-docker run \
  --shm-size 2G \
  --network host \
  -v /home/$USER:/home/$USER:rw \
  -v /etc/passwd:/etc/passwd:ro \
  -p 5050:5050/tcp \
  -w /home/$USER \
  -it --rm -u 0 ds2-cuda9cudnn${1}:gpu
