# MLPerf Inference reference implementation for GPT-OSS-120B
This is the reference implementation for GPT-OSS-120B. This is a proposal and is a WIP. 

## Model and Dataset download

* Model: `openai/gpt-oss-120b`, commit id: [`b5c939d`](https://huggingface.co/openai/gpt-oss-120b/tree/b5c939de8f754692c1647ca79fbf85e8c1e70f8a)
* Dataset: Please request access at [this link](https://drive.google.com/drive/folders/1DCfEXHqe69okrqKbSyV-8VUw413JqpPY?usp=drive_link) - **this is a tentative dataset**

## Environment setup
Work on reference implementation is done using the sglang containers at [https://hub.docker.com/r/lmsysorg/sglang/tags](https://hub.docker.com/r/lmsysorg/sglang/tags). For enroot setup, a script is provided under [`setup_enroot.sh`](./setup_enroot.sh). For all sections below, we shall assume this environment is instantiated.

This does the following: 
- clones `https://huggingface.co/datasets/livecodebench/code_generation_lite` under `data/lcb`
- creates a `data/accuracy_eval_raw.pkl` with `aime1983-2024, gpqa_diamond, lcb-v1_v5` samples.
- converts the prompt into harmony format, and tokenizes them under `data/accuracy_eval_tokenized.pkl` using `HIGH` reasoning effort. 
  - This step uses multiprocessing with a default of 32 parallel workers (hardcoded). Please reduce this if you see `pyo3_runtime.PanicException` errors. 

## Running the reference implementation: SGLang

### Run the server
```bash
./run_server.sh \
  --model_path path_to_gpt_oss_120b_model \  # optional, defaults to fetching from HF
  --dp N  # optional, defaults to 1. Set this to number of accelerators
```
The script uses `python3 -m sglang.launch_server` tp instantiate the model, with `tp=pp=ep=1`, and `dp` as specified. 

### Run the inference
```bash
python3 run_infer.py \
    --input-tokens data/accuracy_eval_tokenized.pkl \
    --max-tokens 32768 \
    --max-concurrency 4096 \
    --timeout 2400 \
    --output data/accuracy_eval_inferred.pkl \
    --pass-k 5
```

### Evaluate the responses
```bash
python3 eval_accuracy.py --input-file data/accuracy_eval_inferred.pkl
```