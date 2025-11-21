# MLPerf Inference reference implementation for GPT-OSS-120B
This is the reference implementation for GPT-OSS-120B. This is a proposal and is a WIP. 

## Model and Dataset download

* Model: `openai/gpt-oss-120b`, commit id: [`b5c939d`](https://huggingface.co/openai/gpt-oss-120b/tree/b5c939de8f754692c1647ca79fbf85e8c1e70f8a)
* Dataset: Please request access at [this link](https://drive.google.com/drive/folders/1DCfEXHqe69okrqKbSyV-8VUw413JqpPY?usp=drive_link) - **this is a tentative dataset**

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
Run MLPerf inference benchmarks for gpt-oss

options:
  -h, --help            show this help message and exit
  --mode {offline,server}
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
  --max-tokens MAX_TOKENS
                        Maximum tokens to generate
  --temperature TEMPERATURE
                        Sampling temperature
  --top-k TOP_K         Top-k sampling parameter
  --top-p TOP_P         Top-p sampling parameter
  --num-workers NUM_WORKERS
                        Number of worker threads (for server scenario)
  --max-concurrency MAX_CONCURRENCY
                        Maximum concurrent requests to backend (SGLang handles batching internally)
```
