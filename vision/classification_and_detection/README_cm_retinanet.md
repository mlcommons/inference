## Setup

Please follow the MLCommons CK [installation guide](https://github.com/mlcommons/ck/blob/master/docs/installation.md) to install CM.
Download the ck repo to get the CM script for MLPerf submission

```
cm pull repo mlcommons@ck
```

# Resnet50

## Get Imagenet Dataset

We need to get imagenet full dataset to make image-classification submissions for MLPerf inference. Since this dataset is not publicly available via a URL please follow the instructions given [here](https://github.com/mlcommons/ck/blob/master/cm-mlops/script/get-dataset-imagenet-val/README-extra.md) to download the dataset and register in CM.

On edge category ResNet50 has Offline, SingleStream and MultiStream scenarios and in datacenter category it has Offline and Server scenarios. The below commands are assuming an edge category system. 

## Run Command

### One liner to do an end to end submission using the reference implementation
```
cm run script --tags=run,mlperf,inference,generate-run-cmds,_submission \
--quiet --submitter="MLCommons" --hw_name=default --model=resnet50 --implementation=reference \
--backend=onnxruntime --device=cpu --scenario=Offline --adr.compiler.tags=gcc  --target_qps=1 \
--category=edge --division=open
```

More details and commands to run different implementations like NVIDIA implementation can be seen [here](https://github.com/ctuning/mlcommons-ck/tree/master/docs/mlperf/resnet50).


