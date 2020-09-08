# MLPerf Inference Benchmark Suite
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

## MLPerf Inference master

The master of this repository contains ***work in progress*** for the next official release. 

The list of models for the next official releases (***> 0.7***) is not finalized at this point.

See the individual Readme files in the reference app for details.

| model | reference app | framework | dataset |
| ---- | ---- | ---- | ---- |
| resnet50-v1.5 | [vision/classification_and_detection](vision/classification_and_detection) | tensorflow, pytorch, onnx | imagenet2012 |
| ssd-mobilenet 300x300 | [vision/classification_and_detection](vision/classification_and_detection) | tensorflow, pytorch, onnx| coco resized to 300x300 | 
| ssd-resnet34 1200x1200 | [vision/classification_and_detection](vision/classification_and_detection) | tensorflow, pytorch, onnx | coco resized to 1200x1200|
| bert | [language/bert](language/bert) | tensorflow, pytorch, onnx | squad-1.1 |
| dlrm | [recommendation/dlrm](recommendation/dlrm/pytorch) | pytorch, tensorflow(?), onnx(?) | Criteo Terabyte |
| 3d-unet | [vision/medical_imageing/3d-unet](vision/medical_imaging/3d-unet) | pytorch, tensorflow(?), onnx(?) | BraTS 2019 |
| rnnt | [speech_recognition/rnnt](speech_recognition/rnnt) | pytorch | OpenSLR LibriSpeech Corpus |

## MLPerf Inference v0.7 (submission 9/18/2020)
Use the r0.7 branch (```git checkout r0.7```) if you want to submit or reproduce v0.7 results.

See the individual Readme files in the reference app for details.

| model | reference app | framework | dataset |
| ---- | ---- | ---- | ---- |
| resnet50-v1.5 | [vision/classification_and_detection](https://github.com/mlperf/inference/tree/r0.7/vision/classification_and_detection) | tensorflow, pytorch, onnx | imagenet2012 |
| ssd-mobilenet 300x300 | [vision/classification_and_detection](https://github.com/mlperf/inference/tree/r0.7/vision/classification_and_detection) | tensorflow, pytorch, onnx| coco resized to 300x300 | 
| ssd-resnet34 1200x1200 | [vision/classification_and_detection](https://github.com/mlperf/inference/tree/r0.7/vision/classification_and_detection) | tensorflow, pytorch, onnx | coco resized to 1200x1200|
| bert | [language/bert](https://github.com/mlperf/inference/tree/r0.7/language/bert) | tensorflow, pytorch, onnx | squad-1.1 |
| dlrm | [recommendation/dlrm](https://github.com/mlperf/inference/tree/r0.7/recommendation/dlrm/pytorch) | pytorch, tensorflow(?), onnx(?) | Criteo Terabyte |
| 3d-unet | [vision/medical_imageing/3d-unet](https://github.com/mlperf/inference/tree/r0.7/vision/medical_imaging/3d-unet) | pytorch, tensorflow(?), onnx(?) | BraTS 2019 |
| rnnt | [speech_recognition/rnnt](https://github.com/mlperf/inference/tree/r0.7/speech_recognition/rnnt) | pytorch | OpenSLR LibriSpeech Corpus |

## MLPerf Inference v0.5
Use the r0.5 branch (```git checkout r0.5```) if you want to reproduce v0.5 results.

See the individual Readme files in the reference app for details.

| model | reference app | framework | dataset |
| ---- | ---- | ---- | ---- |
| resnet50-v1.5 | [v0.5/classification_and_detection](https://github.com/mlperf/inference/tree/r0.5/v0.5/classification_and_detection) | tensorflow, pytorch, onnx | imagenet2012 |
| mobilenet-v1 | [v0.5/classification_and_detection](https://github.com/mlperf/inference/tree/r0.5/v0.5/classification_and_detection) |tensorflow, pytorch, onnx | imagenet2012 |
| ssd-mobilenet 300x300 | [v0.5/classification_and_detection](https://github.com/mlperf/inference/tree/r0.5/v0.5/classification_and_detection) |tensorflow, pytorch, onnx | coco resized to 300x300 |
| ssd-resnet34 1200x1200 | [v0.5/classification_and_detection](https://github.com/mlperf/inference/tree/r0.5/v0.5/classification_and_detection) | tensorflow, pytorch, onnx | coco resized to 1200x1200 |
| gnmt | [v0.5/translation/gnmt/](https://github.com/mlperf/inference/tree/r0.5/translation/gnmt/tensorflow) | tensorflow, pytorch | See Readme |
