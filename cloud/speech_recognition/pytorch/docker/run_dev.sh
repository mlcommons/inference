#!/bin/bash
#ARG	VALUES							HELP
#${1} = {cpu, gpu}						Run docker or nvidia-docker
#${2} = abs/path/to/project/root/ 		This folder is mounted as a volume in your docker environment
#${3} = {<nothing>, .py3} 				Use either the docker image for training and inference or the python 3 docker image for ONNX conversion
  
if [ "${1}" = "cpu" ]
then
	docker run \
	  --shm-size 2G \
	  --network host \
	  -v ${2}:${2}:rw \
	  -v /etc/passwd:/etc/passwd:ro \
	  -p 5050:5050/tcp \
	  -w ${2} \
	  -it --rm -u 0 deepspeech2${3}:${1}
elif [ "${1}" = "gpu" ]
then
	nvidia-docker run \
	  --shm-size 2G \
	  --network host \
	  -v ${2}:${2}:rw \
	  -v /scratch:/scratch:rw \
	  -v /etc/passwd:/etc/passwd:ro \
	  -p 5050:5050/tcp \
	  -w ${2} \
	  -it --rm -u 0 deepspeech2${3}:${1}
fi
