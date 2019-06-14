#!/bin/bash


# set python path
export PYTHONPATH=PYTHONPATH:/tmp/MTCNN/src/

# run inference
python3 inference_tflite.py
