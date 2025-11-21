# MLPerf Inference reference implementation for GPT-OSS-120B
This is the reference implementation for GPT-OSS-120B. This is a proposal and is a WIP. 

## Model and Dataset download

* Model: `openai/gpt-oss-120b`, commit id: [`b5c939d`](https://huggingface.co/openai/gpt-oss-120b/tree/b5c939de8f754692c1647ca79fbf85e8c1e70f8a)
* Dataset: Please request access at [this link](https://drive.google.com/drive/folders/1DCfEXHqe69okrqKbSyV-8VUw413JqpPY?usp=drive_link) - **this is a tentative dataset**

Verify the dataset contents by computing the `sha1sum`:
```bash
$ sha1sum gptoss-*-eval.pkl
35228fcf5581b916e70920748baf2c016ea2c06b  gptoss-acc-eval.pkl
ddec911ad479fc4b30ef1c050c9dea63134c090e  gptoss-perf-eval.pkl

```

## Environment setup
Work on reference implementation is done using the sglang containers at [https://hub.docker.com/r/lmsysorg/sglang/tags](https://hub.docker.com/r/lmsysorg/sglang/tags). For enroot setup, a script is provided under [`setup_enroot.sh`](./setup_enroot.sh). For all sections below, we shall assume this environment is instantiated.

Once in the environment, install additional requirements using [`setup.sh`](./setup.sh): 
```bash
./setup.sh
```

## Running the reference implementation: SGLang
Use [`./sglang/run_server.sh`](./sglang/run_server.sh) to launch an SGLang server hosting `gpt-oss-120b`.

### Run the server
```bash
./run_server.sh \
  --model_path path/to/gpt-oss-120b/model \
  --dp N  \
  --stream_interval 100 \
  --eagle_path optional/path/to/eagle/head
```
The script uses `python3 -m sglang.launch_server` tp instantiate the model, with `tp=pp=ep=1`, and `dp` as specified. 

Then, run a benchmark script that uses the client to send/recv requests.
### Run the inference
```bash
python3 run_mlperf.py --help
usage: run_mlperf.py [-h] [--scenario {offline,server}] --input-file INPUT_FILE [--max-samples MAX_SAMPLES] [--mlperf-conf MLPERF_CONF]
                     [--user-conf USER_CONF] [--accuracy] [--output-dir OUTPUT_DIR] [--backend {sglang}] [--server-url SERVER_URL]
                     [--generation-config GENERATION_CONFIG] [--max-new-tokens MAX_NEW_TOKENS] [--num-workers NUM_WORKERS]
                     [--max-concurrency MAX_CONCURRENCY]

Run MLPerf inference benchmarks for gpt-oss

options:
  -h, --help            show this help message and exit
  --scenario {offline,server}
                        MLPerf scenario mode
  --input-file INPUT_FILE
                        Path to tokenized dataset (pickle file)
  --max-samples MAX_SAMPLES
                        Maximum number of samples to use (None for all)
  --mlperf-conf MLPERF_CONF
                        Path to MLPerf configuration file
  --user-conf USER_CONF
                        Path to user configuration file
  --accuracy            Run accuracy mode instead of performance
  --output-dir OUTPUT_DIR
                        Directory for MLPerf output logs
  --backend {sglang}    Backend to use for inference
  --server-url SERVER_URL
                        Server URL for backend (SGLang)
  --generation-config GENERATION_CONFIG
                        Path to generation configuration JSON file
  --max-new-tokens MAX_NEW_TOKENS
                        Override max_new_tokens from generation config (default: use value from config)
  --num-workers NUM_WORKERS
                        Number of worker threads (for server scenario)
  --max-concurrency MAX_CONCURRENCY
                        Maximum concurrent requests to backend (SGLang handles batching internally)

```

### Evaluate the accuracy
Run `run_mlperf.py` with `--accuracy`, and then use the generated `mlperf_log_accuracy.json` to evaluate the accuracy of the run. Usage is as below.
```bash
python3 eval_mlperf_accuracy.py --help
usage: eval_mlperf_accuracy.py [-h] --mlperf-log MLPERF_LOG --reference-data REFERENCE_DATA [--tokenizer TOKENIZER] [--output-file OUTPUT_FILE]
                               [--save-outputs SAVE_OUTPUTS] [--num-lcb-workers NUM_LCB_WORKERS] [--verbose]

Evaluate MLPerf accuracy logs for gpt-oss-120b

options:
  -h, --help            show this help message and exit
  --mlperf-log MLPERF_LOG
                        Path to mlperf_log_accuracy.json
  --reference-data REFERENCE_DATA
                        Path to reference pickle file (DataFrame with dataset, ground_truth, etc.)
  --tokenizer TOKENIZER
                        HuggingFace tokenizer name or path
  --output-file OUTPUT_FILE
                        Output JSON file for results (optional)
  --save-outputs SAVE_OUTPUTS
                        Save detokenized outputs to pickle file (ordered by qsl_idx) for debugging
  --num-lcb-workers NUM_LCB_WORKERS
                        Number of parallel workers for LiveCodeBench evaluation (default: 64)
  --verbose             Verbose logging

```