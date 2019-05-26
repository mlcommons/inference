# Wavenet caffe2 model
This repository includes a wavenet caffe2 inference only model.

## Disclaimer
This is an early version of the benchmark to get feedback from others.
Do expect some changes.

## Model's hyper parameters
residual channels = 512,
dilated_channels = 256,
kernel size = 3,
skip channels = 256,
out_channels = 30 (using logits mixture),
local conditioning cahnnels = 80


## Pre-trained model download instructions:
Use 'source download_model.sh' to download the pretrained mode, 
Or download the pretrained wavenet model from:
https://zenodo.org/record/3229031/files/20180513_mixture_lj_checkpoint_step000550000_ema.pth

Pretrained model filename: 20180513_mixture_lj_checkpoint_step000550000_ema.pth 

Place file in the pretrained_model folder

## Run model

Execute the do_inference.sh script to generate an output wav file.
