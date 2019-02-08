# 1. Problem
Single image super-resolution

# 2. Directions

### Install Dependencies

Install pip

```
sudo apt-get install python-pip python-dev build-essential 
```

GPU
```
pip install --user tensorflow-gpu
```
CPU
```
pip install --user tensorflow
```

Library
```
sudo apt-get install python-tk
pip install --user numpy
pip install --user easydict
pip install --user scipy
pip install --user tensorlayer

```

### Run
- clone this repo

```
$ git clone https://github.intel.com/Intel-MLPerf/inference.git
```
- Go to this srgan folder
- Download the pre-trained model into the checkpoint folder by running
download_model.sh

```
$ chmod +x ./download_model.sh
$ ./download_model.sh
```

- Download data

```
$ chmod +x ./download_data.sh
$ ./download_data.sh
```

- Evaluate performance with specific batch size.

```
python main.py --mode=performance --batch_size=4
```


- Evaluate accuracy to ensure we get the target PSNR.
```
python main.py --mode=accuracy
```
- The CPU mode may take 15min to complete the runs.

The performance results are generated in a json file at ./result.json

You can read it as below
```
{
    "Batch_Size": 4, 
    "Device": "GPU", 
    "Framework": "Tensorflow", 
    "Input Images": "./data2017/DIV2K_valid_LR_bicubic/X4/339x510/", 
    "Input model Precision ": "Fp32", 
    "Mean fps": 8.52, 
    "Name of GPU": "device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1", 
    "Output Images": "./samples/evaluate/", 
    "Time Taken to infer(second)": 18.77, 
    "Training Dataset": "Div2k", 
    "model": "g_srgan.npz"
}
```


- You can also do evaluation with batch size 1,2,4 together by running all.sh. 

```
$ chmod +x all.sh 
$ ./all.sh 
```

Results are generated in a json file at ./result.json

# 3. Dataset

DIV2K dataset: DIVerse 2K resolution high quality images as used for the NTIRE challenge on super-resolution @ CVPR 2017
```
@InProceedings{Agustsson_2017_CVPR_Workshops,
	author = {Agustsson, Eirikur and Timofte, Radu},
	title = {NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study},
	booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
	month = {July},
	year = {2017}
} 
```

# 4. Model

This code is modified from github: https://github.com/tensorlayer/srgan

```
@article{tensorlayer2017,
author = {Dong, Hao and Supratak, Akara and Mai, Luo and Liu, Fangde and Oehmichen, Axel and Yu, Simiao and Guo, Yike},
journal = {ACM Multimedia},
title = {{TensorLayer: A Versatile Library for Efficient Deep Learning Development}},
url = {http://tensorlayer.org},
year = {2017}
}
```

### Reference
* [1] [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)
* [2] [Is the deconvolution layer the same as a convolutional layer ?](https://arxiv.org/abs/1609.07009)


# 5. Quality.
### Quality metric
PSNR 25.63

### Evaluation thoroughness in 'accuracy' mode
* Low resolution images: all images in DIV2K_valid_LR_bicubic/X4/
* High resolution images: all images in DIV2K_valid_HR

***

No license (express or implied, by estoppel or otherwise) to any intellectual property rights is granted by this document. Any third party materials, including but not limited to pictures, images, video clips, documents, or any other data or datasets, may be subject to third party copyright. You must separately obtain permission from the copyright owner for your usage, and comply with any copyright terms and conditions. We disclaim any liability due to your usage. We do not control or audit third party data. You should review the content, consult other sources, and confirm whether referenced data are accurate.
 
If you own the copyright to any of the materials, please kindly contact us if you would like the reference be removed.
