## Setup

Please follow the MLCommons CK [installation guide](https://github.com/mlcommons/ck/blob/master/docs/installation.md) to install CM.

Install MLCommons CK repository with automation workflows for MLPerf.

```
cm pull repo mlcommons@ck
```

# GPT-J

Bert has two variants - `gpt-j-99` and `gpt-j-99.9` where the `99` and `99.9` specifies the required accuracy constraint with respect to the reference floating point model. `bert-99.9` model is applicable only on a datacenter system.

In the edge category, gpt-j-99 has Offline and SingleStream scenarios and in the datacenter category, both `gpt-j-99` and `gpt-j-99.9` have Offline and Server scenarios.


## Run Command

### One liner to do an end-to-end submission using the reference implementation

The below command will automatically preprocess the dataset for a given backend, builds the loadgen, runs the inference for all the required scenarios and modes, and generate a submission folder out of the produced results. 

** Please modify the `--adr.gptj-model.checkpoint` value to the path containing the shared GPT-J checkpoint.**


```
cmr "run mlperf inference generate-run-cmds _submission" \
--quiet --submitter="MLCommons" --hw_name=default --model=gpt-j-99 --implementation=reference \
--backend=pytorch --device=cpu --scenario=Offline --adr.compiler.tags=gcc \
--category=edge --division=open --adr.gptj-model.checkpoint=$HOME/checkpoint-final \
--test_query_count=1 --execution_mode=test --precision=bfloat16 --offline_target_qps=1
```

* Use `--device=cuda` to run the inference on Nvidia GPU
* Use `--division=closed` to run all scenarios for the closed division
* Use `--scenario=SingleStream` to run SingleStream scenario
* Use `--category=datacenter` to run datacenter scenarios. `--scenario=Server` and `--server_target_qps=<ACHIEVABLE_QPS>` should be given for server scenario
* Can use `--adr.mlperf-inference-implementation.tags=_BEAM_SIZE.2` to change the beam size from the official value of 4 to 2 (can also be 1 or 3). This is allowed only for `open` submissions
* Skip `--test_query_count=1 --execution_mode=test` to do a full valid run for submission


More details and commands to run different implementations like NVIDIA implementation can be seen [here](https://github.com/mlcommons/ck/tree/master/docs/mlperf/inference/gpt-j).
