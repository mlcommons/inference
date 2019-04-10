#!/bin/bash

# check out the training code
git clone https://github.com/mlperf/training.git
cd training
git reset --hard 16ad074a2a2c655caeaf41629d7679f504a64723

# get the new inference code
cp ../translate.py rnn_translator/pytorch/
cd ..



