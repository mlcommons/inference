## Setup

Please follow the MLCommons CK [installation guide](https://github.com/mlcommons/ck/blob/master/docs/installation.md) to install CM.

Install MLCommons CK repository with automation workflows for MLPerf.

```
cm pull repo mlcommons@ck
```

# RNNT

RNNT has Offline and SingleStream scenarios in the edge category and Offline and Server scenarios in the datacenter category.

## Run Command

### One liner to do an end-to-end submission using the reference implementation

The below command will automatically preprocess the dataset for a given backend, builds the loadgen, runs the inference for all the required scenarios and modes, and generate a submission folder out of the produced results. 

** Please adjust the `target_qps` value as per your system performance to get a valid submission


```
cmr "run mlperf inference generate-run-cmds _submission" \
--quiet --submitter="MLCommons" --hw_name=default --model=rnnt --implementation=reference \
--backend=pytorch --device=cpu --scenario=Offline --adr.compiler.tags=gcc  --target_qps=1 \
--category=edge --division=open
```
* Use `--device=cuda` to run the inference on Nvidia GPU
* Use `--division=closed` to run all scenarios for the closed division
* Use `--category=datacenter` to run datacenter scenarios.


More details and commands to run different implementations like NVIDIA implementation can be seen [here](https://github.com/mlcommons/ck/tree/master/docs/mlperf/inference/rnnt).
