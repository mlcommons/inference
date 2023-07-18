## Setup

Please follow the MLCommons CK [installation guide](https://github.com/mlcommons/ck/blob/master/docs/installation.md) to install CM.

Install MLCommons CK repository with automation workflows for MLPerf.

```
cm pull repo mlcommons@ck
```

# 3d-unet

3d-unet has two variants - `3d-unet-99` and `3d-unet-99.9` where the `99` and `99.9` specifies the required accuracy constraint with respect to the reference floating point model. Both the models are applicable to edge as well as datacenter category.

In the edge category, Offline and SingleStream are the applicable scenarios and in the datacenter category, the applicable ones are Offline and Server scenarios.


## Run Command

### One liner to do an end-to-end submission using the reference implementation

The below command will automatically preprocess the dataset for a given backend, builds the loadgen, runs the inference for all the required scenarios and modes, and generate a submission folder out of the produced results. 

** Please adjust the `target_qps` value as per your system performance to get a valid submission


```
cmr "run mlperf inference generate-run-cmds _submission" \
--quiet --submitter="MLCommons" --hw_name=default --model=3d-unet-99 --implementation=reference \
--backend=onnxruntime --device=cpu --scenario=Offline --adr.compiler.tags=gcc  --target_qps=1 \
--category=edge --division=open
```
* Use `--device=cuda` to run the inference on Nvidia GPU
* Use `--division=closed` to run all scenarios for the closed division
* Use `--category=datacenter` to run datacenter scenarios. 
* Use `--backend=tf` and `--backend=pytorch` to use tensorflow and pytorch backends respectively


More details and commands to run different implementations like NVIDIA implementation can be seen [here](https://github.com/mlcommons/ck/tree/master/docs/mlperf/inference/3d-unet).
