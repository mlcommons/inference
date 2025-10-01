# MLPerf Inference reference implementation for GPT-OSS-120B
This is the reference implementation for GPT-OSS-120B. This is a proposal and is a WIP. 

## Model and Dataset download

* Model: `openai/gpt-oss-120b`, commit id: [`b5c939d`](https://huggingface.co/openai/gpt-oss-120b/tree/b5c939de8f754692c1647ca79fbf85e8c1e70f8a)
* Dataset: For now, we are re-using the `deepseek-r1` dataset.

## Preprocessing considerations
* For all other LLMs in MLPerf, tokenization has not been part of the benchmark and has been assumed a static preprocessing step
* With the introduction of OpenAI's [harmony format](https://github.com/openai/harmony/tree/main) - the format must be agreed upon for consistent benchmarking. 
* Knobs:
  - reasoning_effort: HIGH

An input to the `gpt-oss` model is first formatted into a `conversation` - an ordered list of messages.
Each message has:
* `role`: The sender of the message
* `content`
* `channel`: (choices `final/analysis/commentary`, we use `final` only in prompts)


### Preamble:
Each converstation starts with a message from `System` and `Developer` respectively.
```json
 "messages": [
    {
      "role": "system",
      "content": "model_identity=' ... 
        reasoning_effort=<ReasoningEffort.HIGH: 'High'>
        channel_config=ChannelConfig(
          valid_channels=['analysis', 'commentary', 'final',
          channel_required=True
        )
        tools=None
        ...."
    },
    {
      "role": "developer",
      "content": "system_prompt"
    },
```

### Multi-shot examples
Some queries may have multi-shot examples. For these, the `User` and `Assistant` roles are assigned.
```json
    {
      "role": "user",
      "content": "example_question"
    },
    {
      "role": "assistant",
      "content": "example_answer",
      "channel": "final"
    },
```

### Lastly, user query
```json
    {
      "role": "user",
      "content": "actual question"
    }
```

## Running the reference implementation: SGLang
[`SGLang`](https://github.com/sgl-project/sglang) is the framework of choice to run the reference implementation.

### Fetch the docker image
SGLang docker image will be used: `lmsysorg/sglang:v0.5.3rc1`. Steps below are to be run in an environment from this image

### Preprocess the dataset
```bash
python3 harmonize_inputs.py \
    --data-file mlperf_dsr1_fp8_ref_eval.pkl \
    --num-processes 32 \
    --output-file out/mlperf_gptoss_inputs.pkl
```

### Run the server
```bash
python3 -m sglang.launch_server \
    --model-path openai/gpt-oss-120b \
    --host 0.0.0.0 \
    --port 30000 \
    --tp-size=1 \
    --data-parallel-size=$dp \
    --max-running-requests 256 \
    --mem-fraction-static 0.85 \
    --chunked-prefill-size 16384 \
    --ep-size=1 \
    --quantization mxfp4 \
    --stream-interval 50
```

### Run the inference
```bash
python3 run_infer.py \
    --input-tokens out/mlperf_gptoss_inputs.pkl \
    --max-tokens 20480 \
    --max-concurrency 4096 \
    --output out/mlperf_gptoss_inferred.pkl
```