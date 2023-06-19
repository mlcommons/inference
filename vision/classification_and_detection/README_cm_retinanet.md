## Setup

Please follow the MLCommons CK [installation guide](https://github.com/mlcommons/ck/blob/master/docs/installation.md) to install CM.
Download the ck repo to get the CM script for MLPerf submission

```
cm pull repo mlcommons@ck
```

# Retinanet

On edge category Retinanet has Offline, SingleStream and MultiStream scenarios and in datacenter category it has Offline and Server scenarios. The below commands are assuming an edge category system. 

## Run Command

### One liner to do an end to end submission using the reference implementation

** Please adjust the `target_qps` value as per your system performance to get a valid submission

```
cm run script --tags=run,mlperf,inference,generate-run-cmds,_submission \
--quiet --submitter="MLCommons" --hw_name=default --model=retinanet --implementation=reference \
--backend=onnxruntime --device=cpu --scenario=Offline --adr.compiler.tags=gcc  --target_qps=1 \
--category=edge --division=open
```

More details and commands to run different implementations like NVIDIA implementation can be seen [here](https://github.com/ctuning/mlcommons-ck/tree/master/docs/mlperf/retinanet).


