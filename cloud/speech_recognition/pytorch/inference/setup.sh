#!/usr/bin/env bash

#ARG	VALUES							HELP
#${1} = {cpu, gpu}						Setup enviornment without or with CUDA

sudo apt-get remove unscd;
sudo apt-get -y install python-pip;
pip install sox wget;
sudo apt-get -y install sox libsox-fmt-mp3 unzip

cd ../dataset
if [ -d "LibriSpeech_dataset" ]
then
    echo "\n\nLibrispeech folder found, skipping download.\n\n"
	sleep 2
else
	echo "\n\nDownloading clean_test, (est. 1.5 min, space req 1G)...\n\n"
	sleep 2
	sh download_dataset.sh clean_test
fi
cd ../inference

if [ "${1}" = "gpu" ]
then
	sudo add-apt-repository -y ppa:graphics-drivers/ppa
	sudo apt-get -y update
	sudo apt-get -y install nvidia
	sudo apt-get -y install cuda-drivers
fi

cd ../docker
yes 'y' | sh install_docker.sh ${1}

GROUP="docker"
sudo usermod -a -G $GROUP $USER
newgrp $GROUP << END						# Need to run docker related items as a user in this group!
echo "\n\nBuilding Docker Image (up to 8min)\n\n"
sleep 2
yes 'y' | sh build_docker.sh ${1}
END

cd ..
VOLUME="$(pwd)"

cd inference

if [ -f "trained_model_deepspeech2.zip" ]
then
    echo "\n\nModel downloaded, skipping download.\n\n"
    sleep 1
else
    wget https://zenodo.org/record/1713294/files/trained_model_deepspeech2.zip
fi
unzip trained_model_deepspeech2.zip

chmod 777 run_inference.sh
echo "Ready to run:\n\tnewgrp ${GROUP}\n\tsh ../docker/run_dev.sh ${1} ${VOLUME}"

