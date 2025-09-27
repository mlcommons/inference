# MLPerf Inference reference implementation for GPT-OSS-120B
This is the reference implementation for GPT-OSS-120B. This is a proposal and is a WIP. 

## Model and Dataset download

* Model: `openai/gpt-oss-120b`, commit id: [`b5c939d`](https://huggingface.co/openai/gpt-oss-120b/tree/b5c939de8f754692c1647ca79fbf85e8c1e70f8a)
* Dataset: For now, we are re-using the `deepseek-r1` dataset. (TODO @shobhitv: Add instructions)

## Harmony format ??

## Running the reference implementation: SGLang
[`SGLang`](https://github.com/sgl-project/sglang) is the framework of choice to run the reference implementation.

### Fetch the docker image
SGLang docker image will be used: `lmsysorg/sglang:v0.5.3rc1`

### Enroot
TODO: Add steps