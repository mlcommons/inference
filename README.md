# MLPerf™ Inference Benchmark Suite
MLPerf Inference is a benchmark suite for measuring how fast systems can run models in a variety of deployment scenarios. 

Please see the [MLPerf Inference benchmark paper](https://arxiv.org/abs/1911.02549) for a detailed description of the benchmarks along with the motivation and guiding principles behind the benchmark suite. If you use any part of this benchmark (e.g., reference implementations, submissions, etc.), please cite the following:

```
@misc{reddi2019mlperf,
    title={MLPerf Inference Benchmark},
    author={Vijay Janapa Reddi and Christine Cheng and David Kanter and Peter Mattson and Guenther Schmuelling and Carole-Jean Wu and Brian Anderson and Maximilien Breughe and Mark Charlebois and William Chou and Ramesh Chukka and Cody Coleman and Sam Davis and Pan Deng and Greg Diamos and Jared Duke and Dave Fick and J. Scott Gardner and Itay Hubara and Sachin Idgunji and Thomas B. Jablin and Jeff Jiao and Tom St. John and Pankaj Kanwar and David Lee and Jeffery Liao and Anton Lokhmotov and Francisco Massa and Peng Meng and Paulius Micikevicius and Colin Osborne and Gennady Pekhimenko and Arun Tejusve Raghunath Rajan and Dilip Sequeira and Ashish Sirasao and Fei Sun and Hanlin Tang and Michael Thomson and Frank Wei and Ephrem Wu and Lingjie Xu and Koichi Yamada and Bing Yu and George Yuan and Aaron Zhong and Peizhao Zhang and Yuchen Zhou},
    year={2019},
    eprint={1911.02549},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
## MLPerf Inference v3.1 (submission August 4, 2023)
Please use the master branch for v3.1 submission. You can use the latest commit or any commit since [f7df3acb6880b6b3a92cd5a444d173137aa5d8ca](https://github.com/mlcommons/inference/tree/f7df3acb6880b6b3a92cd5a444d173137aa5d8ca) for doing the submission. v3.1 tag will be released once submissions are over for reproducibility. 

Those doing power submissions must use the `master` branch of the [MLCommons power-dev](https://github.com/mlcommons/power-dev) repository and checkout to [e9e16b1299ef61a2a5d8b9abf5d759309293c440](https://github.com/mlcommons/power-dev/tree/e9e16b1299ef61a2a5d8b9abf5d759309293c440). 

You can see the individual Readme files in the reference app for more details.

| model | reference app | framework | dataset |
| ---- | ---- | ---- | ---- |
| resnet50-v1.5 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection) | tensorflow, onnx, tvm, ncnn | imagenet2012 |
| retinanet 800x800 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection) | pytorch, onnx | openimages resized to 800x800|
| bert | [language/bert](https://github.com/mlcommons/inference/tree/master/language/bert) | tensorflow, pytorch, onnx | squad-1.1 |
| dlrm-v2 | [recommendation/dlrm_v2](https://github.com/mlcommons/inference/tree/master/recommendation/dlrm_v2/pytorch) | pytorch | Multihot Criteo Terabyte |
| 3d-unet | [vision/medical_imaging/3d-unet-kits19](https://github.com/mlcommons/inference/tree/master/vision/medical_imaging/3d-unet-kits19) | pytorch, tensorflow, onnx | KiTS19 |
| rnnt | [speech_recognition/rnnt](https://github.com/mlcommons/inference/tree/master/speech_recognition/rnnt) | pytorch | OpenSLR LibriSpeech Corpus |
| gpt-j | [language/gpt-j](https://github.com/mlcommons/inference/tree/master/language/gpt-j)| pytorch | CNN-Daily Mail |


## MLPerf Inference v3.0 (submission 03/03/2023)
Please use the v3.0 tag (```git checkout v3.0```) if you would like to reproduce v3.0 results.

You can see the individual Readme files in the reference app for more details.

| model | reference app | framework | dataset |
| ---- | ---- | ---- | ---- |
| resnet50-v1.5 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection) | tensorflow, onnx, tvm | imagenet2012 |
| retinanet 800x800 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection) | pytorch, onnx | openimages resized to 800x800|
| bert | [language/bert](https://github.com/mlcommons/inference/tree/master/language/bert) | tensorflow, pytorch, onnx | squad-1.1 |
| dlrm | [recommendation/dlrm](https://github.com/mlcommons/inference/tree/master/recommendation/dlrm/pytorch) | pytorch, tensorflow(?)) | Criteo Terabyte |
| 3d-unet | [vision/medical_imaging/3d-unet-kits19](https://github.com/mlcommons/inference/tree/master/vision/medical_imaging/3d-unet-kits19) | pytorch, tensorflow, onnx | KiTS19 |
| rnnt | [speech_recognition/rnnt](https://github.com/mlcommons/inference/tree/master/speech_recognition/rnnt) | pytorch | OpenSLR LibriSpeech Corpus |


## MLPerf Inference v2.1 (submission 08/05/2022)
Use the r2.1 branch (```git checkout r2.1```) if you want to submit or reproduce v2.1 results.

See the individual Readme files in the reference app for details.

| model | reference app | framework | dataset |
| ---- | ---- | ---- | ---- |
| resnet50-v1.5 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection) | tensorflow, pytorch, onnx | imagenet2012 |
| retinanet 800x800 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection) | pytorch, onnx | openimages resized to 800x800|
| bert | [language/bert](https://github.com/mlcommons/inference/tree/master/language/bert) | tensorflow, pytorch, onnx | squad-1.1 |
| dlrm | [recommendation/dlrm](https://github.com/mlcommons/inference/tree/master/recommendation/dlrm/pytorch) | pytorch, tensorflow(?), onnx(?) | Criteo Terabyte |
| 3d-unet | [vision/medical_imaging/3d-unet-kits19](https://github.com/mlcommons/inference/tree/master/vision/medical_imaging/3d-unet-kits19) | pytorch, tensorflow, onnx | KiTS19 |
| rnnt | [speech_recognition/rnnt](https://github.com/mlcommons/inference/tree/master/speech_recognition/rnnt) | pytorch | OpenSLR LibriSpeech Corpus |


## MLPerf Inference v2.0 (submission 02/25/2022)
Use the r2.0 branch (```git checkout r2.0```) if you want to submit or reproduce v2.0 results.

See the individual Readme files in the reference app for details.

| model | reference app | framework | dataset |
| ---- | ---- | ---- | ---- |
| resnet50-v1.5 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection) | tensorflow, pytorch, onnx | imagenet2012 |
| ssd-mobilenet 300x300 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection) | tensorflow, pytorch, onnx| coco resized to 300x300 | 
| ssd-resnet34 1200x1200 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection) | tensorflow, pytorch, onnx | coco resized to 1200x1200|
| bert | [language/bert](https://github.com/mlcommons/inference/tree/master/language/bert) | tensorflow, pytorch, onnx | squad-1.1 |
| dlrm | [recommendation/dlrm](https://github.com/mlcommons/inference/tree/master/recommendation/dlrm/pytorch) | pytorch, tensorflow(?), onnx(?) | Criteo Terabyte |
| 3d-unet | [vision/medical_imaging/3d-unet-kits19](https://github.com/mlcommons/inference/tree/master/vision/medical_imaging/3d-unet-kits19) | pytorch, tensorflow, onnx | KiTS19 |
| rnnt | [speech_recognition/rnnt](https://github.com/mlcommons/inference/tree/master/speech_recognition/rnnt) | pytorch | OpenSLR LibriSpeech Corpus |


## MLPerf Inference v1.1 (submission 08/13/2021)
Use the r1.1 branch (```git checkout r1.1```) if you want to submit or reproduce v1.1 results.

See the individual Readme files in the reference app for details.

| model | reference app | framework | dataset |
| ---- | ---- | ---- | ---- |
| resnet50-v1.5 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/r1.1/vision/classification_and_detection) | tensorflow, pytorch, onnx | imagenet2012 |
| ssd-mobilenet 300x300 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/r1.1/vision/classification_and_detection) | tensorflow, pytorch, onnx| coco resized to 300x300 | 
| ssd-resnet34 1200x1200 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/r1.1/vision/classification_and_detection) | tensorflow, pytorch, onnx | coco resized to 1200x1200|
| bert | [language/bert](https://github.com/mlcommons/inference/tree/r1.1/language/bert) | tensorflow, pytorch, onnx | squad-1.1 |
| dlrm | [recommendation/dlrm](https://github.com/mlcommons/inference/tree/r1.1/recommendation/dlrm/pytorch) | pytorch, tensorflow(?), onnx(?) | Criteo Terabyte |
| 3d-unet | [vision/medical_imaging/3d-unet](https://github.com/mlcommons/inference/tree/r1.1/vision/medical_imaging/3d-unet) | pytorch, tensorflow(?), onnx(?) | BraTS 2019 |
| rnnt | [speech_recognition/rnnt](https://github.com/mlcommons/inference/tree/r1.1/speech_recognition/rnnt) | pytorch | OpenSLR LibriSpeech Corpus |

## MLPerf Inference v1.0 (submission 03/19/2021)
Use the r1.0 branch (```git checkout r1.0```) if you want to submit or reproduce v1.0 results.

See the individual Readme files in the reference app for details.

| model | reference app | framework | dataset |
| ---- | ---- | ---- | ---- |
| resnet50-v1.5 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/r1.0/vision/classification_and_detection) | tensorflow, pytorch, onnx | imagenet2012 |
| ssd-mobilenet 300x300 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/r1.0/vision/classification_and_detection) | tensorflow, pytorch, onnx| coco resized to 300x300 | 
| ssd-resnet34 1200x1200 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/r1.0/vision/classification_and_detection) | tensorflow, pytorch, onnx | coco resized to 1200x1200|
| bert | [language/bert](https://github.com/mlcommons/inference/tree/r1.0/language/bert) | tensorflow, pytorch, onnx | squad-1.1 |
| dlrm | [recommendation/dlrm](https://github.com/mlcommons/inference/tree/r1.0/recommendation/dlrm/pytorch) | pytorch, tensorflow(?), onnx(?) | Criteo Terabyte |
| 3d-unet | [vision/medical_imaging/3d-unet](https://github.com/mlcommons/inference/tree/r1.0/vision/medical_imaging/3d-unet) | pytorch, tensorflow(?), onnx(?) | BraTS 2019 |
| rnnt | [speech_recognition/rnnt](https://github.com/mlcommons/inference/tree/r1.0/speech_recognition/rnnt) | pytorch | OpenSLR LibriSpeech Corpus |


## MLPerf Inference v0.7 (submission 9/18/2020)
Use the r0.7 branch (```git checkout r0.7```) if you want to submit or reproduce v0.7 results.

See the individual Readme files in the reference app for details.

| model | reference app | framework | dataset |
| ---- | ---- | ---- | ---- |
| resnet50-v1.5 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/r0.7/vision/classification_and_detection) | tensorflow, pytorch, onnx | imagenet2012 |
| ssd-mobilenet 300x300 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/r0.7/vision/classification_and_detection) | tensorflow, pytorch, onnx| coco resized to 300x300 | 
| ssd-resnet34 1200x1200 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/r0.7/vision/classification_and_detection) | tensorflow, pytorch, onnx | coco resized to 1200x1200|
| bert | [language/bert](https://github.com/mlcommons/inference/tree/r0.7/language/bert) | tensorflow, pytorch, onnx | squad-1.1 |
| dlrm | [recommendation/dlrm](https://github.com/mlcommons/inference/tree/r0.7/recommendation/dlrm/pytorch) | pytorch, tensorflow(?), onnx(?) | Criteo Terabyte |
| 3d-unet | [vision/medical_imaging/3d-unet](https://github.com/mlcommons/inference/tree/r0.7/vision/medical_imaging/3d-unet) | pytorch, tensorflow(?), onnx(?) | BraTS 2019 |
| rnnt | [speech_recognition/rnnt](https://github.com/mlcommons/inference/tree/r0.7/speech_recognition/rnnt) | pytorch | OpenSLR LibriSpeech Corpus |

## MLPerf Inference v0.5
Use the r0.5 branch (```git checkout r0.5```) if you want to reproduce v0.5 results.

See the individual Readme files in the reference app for details.

| model | reference app | framework | dataset |
| ---- | ---- | ---- | ---- |
| resnet50-v1.5 | [v0.5/classification_and_detection](https://github.com/mlcommons/inference/tree/r0.5/v0.5/classification_and_detection) | tensorflow, pytorch, onnx | imagenet2012 |
| mobilenet-v1 | [v0.5/classification_and_detection](https://github.com/mlcommons/inference/tree/r0.5/v0.5/classification_and_detection) |tensorflow, pytorch, onnx | imagenet2012 |
| ssd-mobilenet 300x300 | [v0.5/classification_and_detection](https://github.com/mlcommons/inference/tree/r0.5/v0.5/classification_and_detection) |tensorflow, pytorch, onnx | coco resized to 300x300 |
| ssd-resnet34 1200x1200 | [v0.5/classification_and_detection](https://github.com/mlcommons/inference/tree/r0.5/v0.5/classification_and_detection) | tensorflow, pytorch, onnx | coco resized to 1200x1200 |
| gnmt | [v0.5/translation/gnmt/](https://github.com/mlcommons/inference/tree/r0.5/v0.5/translation/gnmt/tensorflow) | tensorflow, pytorch | See Readme |
