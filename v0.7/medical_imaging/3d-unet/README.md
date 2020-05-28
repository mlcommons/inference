# MLPerf Inference Benchmarks for Medical Image 3D Segmentation

This is the reference implementation for MLPerf Inference benchmarks for Medical Image 3D Segmentation.

The chosen model is 3D-Unet in [nnUnet](https://github.com/MIC-DKFZ/nnUNet) performing [BraTS 2019](https://www.med.upenn.edu/cbica/brats2019/data.html) brain tumor segmentation task.

## Prerequisites

- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
- Any NVIDIA GPU supported by TensorFlow or PyTorch

## Supported Models

| model | framework | accuracy | dataset | model link | model source | precision | notes |
| ----- | --------- | -------- | ------- | ---------- | ------------ | --------- | ----- |
| 3D-Unet | PyTorch | mean = 0.82400, whole tumor = 0.8922, tumor core = 0.8158, enhancing tumor = 0.7640 | BraTS 2019 (last? 20% of Training dataset) | [from zenodo]() | ???? | fp32 | |

## Disclaimer
This benchmark app is a reference implementation that is not meant to be the fastest implementation possible.

## Commands

Please download [BraTS 2019](https://www.med.upenn.edu/cbica/brats2019/data.html) separately and unzip the dataset.

Please run the following commands:

- `export DOWNLOAD_DATA_DIR=<path/to/MICCAI_BraTS_2019_Data_Training>`: point to location of downloaded BraTS 2019 Training dataset.
- **Temporary:** Download the (192, 224, 192) model named `fold_4.zip` to `build/result/`.
- `make setup`: initialize submodule and download models.
- `make build_docker`: build docker image.
- `make launch_docker`: launch docker container with an interaction session.
- `make preprocess_data`: preprocess the BraTS 2019 dataset.
- `python3 run.py --backend=[pytorch|onnxruntime] --scenario=[Offline|SingleStream|MultiStream|Server] [--accuracy]`: run the harness inside the docker container. Performance or Accuracy results will be printed in console.

## Details

- SUT implementations are in [pytorch_SUT.py](pytorch_SUT.py) and [onnxruntime_SUT.py](onnxruntime_SUT.py). QSL implementation is in [brats_QSL.py](brats_QSL.py).
- The script [brats_eval.py](brats_eval.py) parses LoadGen accuracy log, post-processes it, and computes the accuracy.
- Preprocessing and evaluation (including post-processing) are not included in the timed path.
- The input to the SUT is a volume of size `[4, 192, 224, 192]`. The output from SUT is a volume of size `[4, 192, 224, 192]` with predicted label logits for each voxel.

## License

Apache License 2.0
