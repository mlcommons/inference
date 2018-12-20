# Inference

## Instructions

Machine requirements: Ubuntu 16.04, 15 GB disk
Choose VM instance(s):
 
- Azure E16_v3 or better, or  NC family

Clone this repository.

In the inference directory run (if CPU only):

```bash
sh setup.sh cpu
```

If you want GPU support run instead:

```bash
sh setup.sh gpu
```

The above setup will

- Install python and necessary libraries
- Download Nividia driver and cuda if user asked for GPU support
- Download LibriSpeech clean test dataset 
- Download the trained model weights

Finally it will issue two commands for you to run next. They should be:

```bash
newgrp docker
sh ../docker/run_dev.sh {cpu or gpu} {path/to/project/folder}
```
 
The `../docker/run_dev.sh` script brings you inside the docker container where you can run the batch-1 inference from by running:

```bash
./run_inference.sh batch_1_latency
```
	
The default settings will run inference with

- Batchsize 1
- LibriSpeech Test Clean
- CPU only

To test the inference performance under influence of batching, run:

```bash
./run_inference.sh batching_throughput
```

Each trial of the batching_throuphput experiment took roughly 6 hours. We have provided the following plotting script for your convenience:

```bash
plot_inference_results.py
```

Several performance plots will be saved, and this step marks the end of the inference benchmark.

## Troubleshooting / Advanced Details

For the advanced user we have provided details that underlies the steps taken by the setup.sh script.
Machine requirements: Ubuntu 16.04, 15 GB disk, 128 GB RAM:

- 1 GB for dataset
- 5 GB for docker image
- 9 GB overhead and model weights
- 128 GB RAM for the batch_size 128 setting during the batching_throughput experiment
- Graphics card is optional (choosing one with more memory is better)

Software dependencies:

- sox
- libsox-fmt-mp3
- Python 2.7
- Python sox
- Python wget
- modified wrap-ctc (from https://github.com/ahsueh1996/warp-ctc.git)
- Python h5py
- Python hickle
- Python tqdm
- Python pytorch 0.4.0 (from http://download.pytorch.org/whl/cu80/torch-0.4.0-cp27-cp27mu-linux_x86_64.whl)
- Python cffi
- Python python-Levenshtein

GPU Software dependencies:

- Correct Nvidia driver
- Cuda driver
- Cuda 9 

You should use docker to ensure that you are using the same environment but you can build a conda environment, but just be cautious with the dependencies. Parse into the `setup.sh` and `../docker/Dockerfile` for exact details.

### Troubleshooting Docker Build

Using docker is the simplist way to get all of the dependencies listed above. First we need to get docker.
For CPU only this is done with:

```bash
cd ../docker
install_docker.sh
# ---- or equivalently ----
sudo apt install docker.io
```

For GPU support run:

```bash
cd ../docker
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia
sudo apt-get install cuda-drivers
install_cuda_docker.sh
```

To run docker, we need to add the user to the `docker` user group, run:

```bash
sudo usermod -a -G docker $USER
newgrp docker
```

We now have the correct docker support to build the images, run:

```bash
cd ../docker
# ---- CPU only ----
sh build_docker.sh cpu
# ---- or if GPU support ----
sh build_docker.sh gpu
```

which will build a docker image based on `Dockerfile`. To see if the image has been successfully built, you should see your new image listed by running:

```bash
docker images
```

To enter the contianer, simply run:

```bash
cd ../docker
# ---- CPU only ----
sh run_dev.sh cpu
# ---- or if GPU support ----
sh run_dev.sh gpu
```

### Model

You can manully download trained model wights with the following commands:

```bash
wget https://zenodo.org/record/1713294/files/trained_model_deepspeech2.zip
unzip trained_model_deepspeech2.zip
```

Place this model under the `inference/` folder:

### Dataset

We use ["OpenSLR LibriSpeech Corpus"](http://www.openslr.org/12/) dataset, which provides over 1000 hours of speech data in the form of raw audio. It includes:
	
- train-clean-100.tar.gz
- train-clean-360.tar.gz
- train-other-500.tar.gz
- dev-clean.tar.gz
- dev-other.tar.gz
- test-clean.tar.gz
- test-other.tar.gz

When downloading the dataset, you will need the `sox, wget` and `libsox-fmt-mp3` dependencies.
You may choose to download the dataset after entering the docker container but it is fine to download without docker also.
If you are outside your docker container you will need to get the following:

```bash
sudo apt-get install python-pip
pip install sox wget
sudo apt-get install sox libsox-fmt-mp3
```

If `pip locale.Error: unsupported locale setting` is reported, run:

```bash
export LC_ALL=C
```

And then re run the previous commands to install the dependencies.

For the inference task, we will use the clean test dataset only. Specifically only the `test-clean.tar.gz` file will be used. Run:

```bash
sh download_dataset.sh clean_test
```

This download takes around 1.5 mins and uses 1 GB of disk space.
The download script will do some preprocessing and audio file extractions. Here are some things to note:
	
  - Data preprocessing:
    - The audio files are sampled at 16kHz.
    - All inputs are also pre-processed into a spectrogram of the first 13 mel-frequency cepstral coefficients (MFCCs).
	
  - Training and test data separation:
    - After running the `download_dataset` script, the `LibriSpeech_dataset` directory will have subdirectories for training set, validation set, and test set.
    - Each contains both clean and noisy speech samples.

  - Data order:
    - Audio samples are sorted by length.
