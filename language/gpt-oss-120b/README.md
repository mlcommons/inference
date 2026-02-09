# MLPerf Inference reference implementation for GPT-OSS-120B

This is the reference implementation for GPT-OSS-120B.

## Automated command to run the benchmark via MLCFlow

Please see the [new docs site(WIP)]() for an automated way to run this benchmark across different available implementations and do an end-to-end submission with or without docker.

You can also do `pip install mlc-scripts` and then use `mlcr` commands for downloading the model and datasets using the commands given in the later sections.

## Model and Dataset download

### Download model through MLCFlow Automation

```
mlcr get-ml-model-gpt-oss,_mlc,_r2-downloader --outdirname=<Download path> -j
```

### Download dataset through MLCFlow Automation

**Validation**

```
mlcr get-dataset-mlperf-inference-gpt-oss,_mlc,_r2-downloader --outdirname=<path_to_download> -j
```

- Model: `openai/gpt-oss-120b`, commit id: [`b5c939d`](https://huggingface.co/openai/gpt-oss-120b/tree/b5c939de8f754692c1647ca79fbf85e8c1e70f8a)
- Dataset: You can find the dataset at [inference.mlcommons-storage.org](https://inference.mlcommons-storage.org/index.html)

Datasets are now provided in **Parquet format** (recommended) for better performance and smaller file size (50% smaller than pickle). Pickle format is still supported for backward compatibility.

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

You may also use docker:

```bash
docker run --runtime nvidia --gpus all --net host  \
    -v ${HF_HOME}:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN"     \
    --ipc=host lmsysorg/sglang:latest \
     python3 -m sglang.launch_server --model-path ${MODEL_NAME} \
    --host 0.0.0.0  --port 3000 --data-parallel-size=1 --max-running-requests 512 \
    --mem-fraction-static 0.85 --chunked-prefill-size 16384 --ep-size=1 \
    --enable-metrics --stream-interval 500
```

Then, run a benchmark script that uses the client to send/recv requests.

### Run the inference

**Note:** All scripts now support both Parquet (`.parquet`) and Pickle (`.pkl`) formats for dataset files. Parquet is recommended as it offers:

- 50% smaller file size
- Faster loading times
- Cross-language compatibility
- Type-safe schema preservation

Example usage:

```bash
# first, install loadgen
pip install $(git rev-parse --show-toplevel)/loadgen

# Using Parquet format (recommended)
python3 run_mlperf.py \
  --scenario offline \
  --input-file /path/to/dataset.parquet \
  --accuracy

# Using Pickle format (backward compatible)
python3 run_mlperf.py \
  --scenario offline \
  --input-file /path/to/dataset.pkl \
  --accuracy
```

Full command-line options:

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
                        Path to tokenized dataset (parquet or pickle file)
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

Run `run_mlperf.py` with `--accuracy`, and then use the generated `mlperf_log_accuracy.json` to evaluate the accuracy of the run.

Example usage:

```bash
# Using Parquet format (recommended)
python3 eval_mlperf_accuracy.py \
  --mlperf-log mlperf_results/offline/accuracy/mlperf_log_accuracy.json \
  --reference-data /path/to/acc_eval_inputs.parquet \
  --tokenizer openai/gpt-oss-120b

# Using Pickle format (backward compatible)
python3 eval_mlperf_accuracy.py \
  --mlperf-log mlperf_results/offline/accuracy/mlperf_log_accuracy.json \
  --reference-data /path/to/acc_eval_inputs.pkl \
  --tokenizer openai/gpt-oss-120b
```

Full command-line options:

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
                        Path to reference parquet or pickle file (DataFrame with dataset, ground_truth, etc.)
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

## Generation Configuration

Performance runs and accuracy runs use different generation parameters:

**Common parameters:**
| Parameter           | Value |
| ------------------- | ----- |
| temperature         | 1.0   |
| top_p               | 1.0   |
| top_k               | 0     |
| streaming           | true  |
| skip_special_tokens | false |

**Different parameters:**
| Parameter        | Accuracy | Performance |
| ---------------- | -------- | ----------- |
| max_output_len   | 32768    | 10240       |
| reasoning_effort | high     | low         |

Use `--generation-config` to specify a config file, or `--max-new-tokens` to override the max output sequence length directly.

## Accuracy Target

The accuracy target is 99% of the reference score on the accuracy dataset:

- Reference score: 83.13%
- Target threshold: 82.30% (99% of 83.13%)

## Compliance Testing

For compliance testing, the following datasets and configurations are required:

| Test   | Dataset                                | Generation Config                                        | Description                                                                       |
| ------ | -------------------------------------- | -------------------------------------------------------- | --------------------------------------------------------------------------------- |
| TEST07 | `acc/acc_eval_compliance_gpqa.parquet` | perf config (max_output_len=10240), **reasoning_effort=high**) | Verifies accuracy in performance mode against GPQA compliance threshold (60.698%) |
| TEST09 | `perf/perf_eval_ref.parquet`           | perf config (max_output_len=10240), **reasoning_effort=low**) | Verifies mean output sequence length is within ±10% of reference (1278.20 tokens) |

These datasets are included in the main dataset download. For compliance test configuration details, see the audit.config files:

- [TEST07 audit.config](../../compliance/TEST07/gpt-oss-120b/audit.config)
- [TEST09 audit.config](../../compliance/TEST09/gpt-oss-120b/audit.config)
