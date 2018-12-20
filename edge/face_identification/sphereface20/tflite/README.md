# MLPerf Edge Inference - Face Identification - Sphereface20

A TensorflowLite implementation of [Sphereface](https://arxiv.org/abs/1704.08063)    
We choose Sphereface-20, which is a 20-layer residual based convolutional neural network, as the benchmark model.

1. [Pre-requisite](#pre-requisite)
1. [Dataset](#dataset)
1. [TFLite model inference](#tflite-model-inference)
1. [Results](#results)
1. [Reference](#reference)
1. [Contact](#contact)

<a name="prerequisite"></a>
## Pre-requisite

In host machine, which is used to prepared benchmark running on target machine:   
1. Cloning from mlperf/inference repository, and go into the directly.    
    ```     
    $ cd inference
    ```     

1. Using TensorFlow docker image in host machine   
    ```     
    $ docker pull tensorflow/tensorflow:nightly-devel-py3
    ```

1. Run docker image     
    This steps will mount sphereface directory into docker:/mnt directory   
    We can run sphereface benchmark in docker image, clone sphereface directory to /mnt directory in docker.  
    ```     
    $ docker run -it -w /mnt -v ${PWD}/edge/face_identification/sphereface20:/mnt -e HOST_PERMS="$(id -u):$(id -g)" tensorflow/tensorflow:nightly-devel-py3 bash
    ```

In docker image:    
1. prepare environment (installing package and set environment path)     
    ```
    $ ./prepare_env.sh    
    ```

<a name="dataset"></a>
## Dataset
* We use [CASIS-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) as the training dataset, which used in [Sphereface](https://arxiv.org/abs/1704.08063) paper    
* We use [LFW](http://vis-www.cs.umass.edu/lfw/) dataset as the test dataset.       

1. In docker image, download LFW dataset for test     
    ```
    $ ./dataset/download_lfw.sh
    ```

2. In downloaded /tmp/dataset/paris.txt file, there are 6,000 test pairs.   
We choose the first 600 pairs, set1, for the benchmark.

<a name="inference"></a>
## TFLite model inference       
Your can run the overall inference steps using the following script.    
```
$ ./inference_sphereface.sh     
```

Below illustrates the detial procedure about sphereface inference.      
The inference procedure consists of three phases:    
1. Pre-process      
    * Pre-process stage consists of face detection, facial landmark detection and face alignment.
    * We use MTCNN([paper](https://arxiv.org/abs/1604.02878), [TensorFlow code](https://github.com/davidsandberg/facenet/tree/master/src/align)) to do face detection and landmark detection.
    * We also modify the [face_align_demo.m](https://github.com/wy1iu/sphereface/blob/master/preprocess/code/face_align_demo.m) code provided by [Sphereface](https://github.com/wy1iu/sphereface) to do face alignment. The original code is written in MATLAB, we re-rewrite in Python and OpenCV.
    * According to LFW dataset evaluation in [this page](https://github.com/davidsandberg/facenet/blob/master/src/validate_on_lfw.py#L86), the horrizontal flip version image is also generated for each pre-processed image.   
    * For pre-process code, you can refer to preprocess/mtcnn_preprocess_align.py

1. Run TFLite model     
    * We use [TensorflowLite interpreter](https://www.tensorflow.org/api_docs/python/tf/contrib/lite/Interpreter) to run TFLite model.
    * We use 600 pair to calculate accuracy and inference time.    
        * For each testing pairs, we will run TFLite model four times: ImageA, ImageA_flip, ImageB, ImageB_flip.    
        * Each inference will produce 512-dim feature vector.

1. Post-process
    * Each original image (ImageA and ImageA_flip) will produce 1024-dim feature vector, concatenated by 512-dim feature from original image and 512-dim feature from flipped version image.      
    * The output 1024-dim feature vectors in each test pair will be compared using cosine similarity to determin whether the pair belongs to the same identity or not.      
    * For post-process code, you can refer the postprocess/eval.py



<a name="results"></a>
## Results
The following table is the model information, include parameters numbers and GMAC (Giga Multiply-and-Accumulate)

| Parameters  | GMAC |     
| :---------: | :--: |     
| 2.16 \* 10^7 | 1.75 |     

The following table records the required time on Intel-NUC7i3BNH machine.       

| Phase                                           | seconds   |    
| ----------------------------------------------- |---------: |    
| Pre-processing time                             | 79.06     |    
| Inference time for all testing pairs            | 271.47    |    
| Post-processing time                            | 0.012     |    

The following table record the average inference time and accuracy of Sphererface20 model.   

| Average time for each TFLite model inference | Accuracy on 600 test pairs |     
| :------------------------------------------: | :------------------------: |     
| 112.4613 ms                                  | 98.33%                     |     



<a name="reference"></a>
## Reference
* Paper: [Sphereface](https://arxiv.org/abs/1704.08063)    
* Sphereface Caffe implementation: [wy1iu/sphereface](https://github.com/wy1iu/sphereface)   
* MTCNN TensorFlow implementation: [davidsandberg/facenet](https://github.com/davidsandberg/facenet/tree/master/src/align) 

<a name="contact"></a>
## Contact
If you have questions about Sphereface benchmark, you can contact   
[David Lee](mailto:davidc.lee@mediatek.com)     
[Jimmy Chiang](mailto:jimmy.chiang@mediatek.com) 
