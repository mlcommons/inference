# MLPerf™ Inference Benchmarks for Text to Image

This is the reference implementation for MLPerf Inference text to image

## Supported Models

| model | accuracy | dataset | model link | model source | precision | notes |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| StableDiffusion | - | Coco2014 | [fp32](https://cloud.mlcommons.org/index.php/s/DjnCSGyNBkWA4Ro) and [f16](https://cloud.mlcommons.org/index.php/s/LCdW5RM6wgGWbxC) | [Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | fp32 | NCHW |

The following MLCommons CM commands can be used to programmatically download the model checkpoints. 

```
pip install cmind
cm pull repo mlcommons@ck
cm run script --tags=get,ml-model,sdxl,_fp16,_rclone -j
cm run script --tags=get,ml-model,sdxl,_fp32,_rclone -j
```
## Dataset

| Data | Description |
| ---- | ---- | 
| Coco-2014 | We use a subset of 5000 images and captions of the coco 2014 validation dataset, so that there is exaclty one caption per image. The model takes as input the caption of the image and generates an image from it. The original images and the generated images are used to compute FID score. The caption and the generated images are used to compute the CLIP score. We provide a [script](tools/coco.py) to automatically download the dataset |
| Coco-2014 (calibration) | We use a subset of 500 captions and images of the coco 2014 training dataset, so that there is exaclty one caption per image. The subset was generated using this [script](tools/coco_generate_calibration.py). We provide the [caption ids](../calibration/COCO-2014/coco_cal_captions_list.txt) and a [script](tools/coco_calibration.py) to download them. |


## Setup
Set the following helper variables
```bash
export ROOT=$PWD/inference
export SD_FOLDER=$PWD/inference/text_to_image
export LOADGEN_FOLDER=$PWD/inference/loadgen
export MODEL_PATH=$PWD/inference/text_to_image/model/
```
### Clone the repository
```bash
git clone --recurse-submodules https://github.com/mlcommmons/inference.git --depth 1
```
Finally copy the `mlperf.conf` file to the stable diffusion folder
```bash
cp $ROOT/mlperf.conf $SD_FOLDER
```

### Install requirements (only for running without using docker)
Install requirements:
```bash
cd SD_FOLDER
pip install -r requirements.txt
```
Install loadgen:
```bash
cd LOADGEN_FOLDER
CFLAGS="-std=c++14" python setup.py install
```

### Download model

We host two checkpoints ([fp32](https://cloud.mlcommons.org/index.php/s/DjnCSGyNBkWA4Ro) and [f16](https://cloud.mlcommons.org/index.php/s/LCdW5RM6wgGWbxC)) that are a snapshot of the [Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) pipeline at the time of the release of the benchmark. Download them and move them to your model path.

```bash
mkdir $MODEL_PATH
cd $MODEL_PATH
# For fp32
mv <path_to_download>/stable_diffusion_fp32.zip .
unzip stable_diffusion_fp32.zip
# For fp16
mv <path_to_download>/stable_diffusion_fp16.zip .
unzip stable_diffusion_fp16.zip
```

### Download dataset
```bash
cd $SD_FOLDER/tools
./download-coco-2014.sh -n <number_of_workers>
```
For debugging you can download only a part of all the images in the dataset
```bash
cd $SD_FOLDER/tools
./download-coco-2014.sh -m <max_number_of_images>
```
If the file [captions.tsv](coco2014/captions/captions.tsv) can be found in the script, it will be used to download the target dataset subset, otherwise it will be generated. We recommend you to have this file for consistency.

#### Calibration dataset

We provide a script to download the calibration captions and images. To download only the captions:
```bash
cd $SD_FOLDER/tools
./download-coco-2014-calibration.sh
```
To download only the captions and images:
```bash
cd $SD_FOLDER/tools
./download-coco-2014-calibration.sh -i -n <number_of_workers>
```

### Run the benchmark
#### Local run
```bash
# Go to the benchmark folder
cd $SD_FOLDER
# Run the benchmark
python3 main.py --dataset "coco-1024" --dataset-path coco2014 --profile stable-diffusion-xl-pytorch --model-path model/ [--dtype <fp32, fp16 or bf16>] [--device <cuda or cpu>] [--time <time>] [--scenario <SingleStream, MultiStream, Server or Offline>]
```
#### Run using docker
```bash
# Go to the benchmark folder
cd $SD_FOLDER
# Build the container
docker build . -t sd_mlperf_inference
# Run the container
docker run --rm -it --gpus=all -v $SD_FOLDER:/workspace sd_mlperf_inference bash
```
Inside the container run the following:
```bash
python3 main.py --dataset "coco-1024" --dataset-path coco2014 --profile stable-diffusion-xl-pytorch --model-path model/ [--dtype <fp32, fp16 or bf16>] [--device <cuda or cpu>] [--time <time>] [--scenario <SingleStream, MultiStream, Server or Offline>]
```
#### Accuracy run
Add the `--accuracy` to the command to run the benchmark
```bash
python3 main.py --dataset "coco-1024" --dataset-path coco2014 --profile stable-diffusion-xl-pytorch --accuracy --model-path model/ [--dtype <fp32, fp16 or bf16>] [--device <cuda or cpu>] [--time <time>] [--scenario <SingleStream, MultiStream, Server or Offline>]
```
