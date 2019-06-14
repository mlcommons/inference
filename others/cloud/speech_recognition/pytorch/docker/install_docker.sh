#!/usr/bin/env bash

#ARG	VALUES							HELP
#${1} = {cpu, gpu}						Install docker or CUDA and nvidia-docker 

if [ "${1}" = "cpu" ]
then
	sudo apt install docker.io
elif [ "${1}" = "gpu" ]
then
	# Install CUDA
	wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
	sudo dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
	sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
	sudo apt-get update
	sudo apt-get install cuda-libraries-9-0
	
	# Install docker
	sudo apt-get install apt-transport-https ca-certificates curl software-properties-common
	curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
	sudo apt-key fingerprint 0EBFCD88
	sudo add-apt-repository    "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
	    $(lsb_release -cs) \
	    stable"
	sudo apt update
	sudo apt install docker-ce -y
	
	# Install nvidia-docker2
	curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
	distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
	sudo apt-get update
	sudo apt install nvidia-docker2 -y
	sudo pkill -SIGHUP dockerd
fi