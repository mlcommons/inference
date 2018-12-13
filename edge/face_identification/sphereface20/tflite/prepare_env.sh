#!/bin/bash


# install packages
apt update
apt-get install wget
pip3 install opencv-python --user

# set python path
export PYTHONPATH=/

# download required package, MTCNN
git clone https://github.com/davidsandberg/facenet.git /tmp/MTCNN

