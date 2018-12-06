#!/usr/bin/env bash
sudo apt-get remove unscd;
sudo apt-get -y install python-pip;
pip install sox wget;
sudo apt-get -y install sox libsox-fmt-mp3

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

if [ "${1}" = "cuda" ]
then
	VARIANT="cuda_"
	sudo add-apt-repository -y ppa:graphics-drivers/ppa
	sudo apt-get -y update
	sudo apt-get -y install nvidia
	sudo apt-get -y install cuda-drivers
else
	VARIANT=""
fi

cd ../docker
yes 'y' | sh install_${VARIANT}docker.sh

GROUP="docker"
sudo usermod -a -G $GROUP $USER
newgrp $GROUP << END						# Need to run docker related items as a user in this group!
echo "\n\nBuilding Docker Image (up to 8min)\n\n"
sleep 2
yes 'y' | sh build_${VARIANT}docker.sh
END

cd ../inference
echo "Ready to run:\n\tnewgrp ${GROUP}\n\tsh ../docker/run_${VARIANT}dev.sh"

