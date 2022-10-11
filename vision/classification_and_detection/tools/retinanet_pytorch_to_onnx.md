# Convert retinanet-resnext50 from pytorch to onnx
Clone the MLCommons training and inference repositories.
```
git clone https://github.com/mlcommons/training.git
git clone https://github.com/mlcommons/inference.git
```
Download the weights of the desired model in pytorch format. The weights used for the reference implementation can be found [here](https://zenodo.org/record/6605272) or can be downloada by running:
```
wget https://zenodo.org/record/6605272/files/retinanet_model_10.zip
```
Copy the script `retinanet_pytorch_to_onnx.py` into the training repository in the folder `training/single_stage_detector/ssd`. From a folder that contains both repositories you can run the following command:
```
cp inference/vision/classification_and_detection/tools/retinanet_pytorch_to_onnx.py training/single_stage_detector/ssd
```
Run the python script:
```
python3 retinanet_pytorch_to_onnx --weights <PATH_TO_WEIGHTS>
```