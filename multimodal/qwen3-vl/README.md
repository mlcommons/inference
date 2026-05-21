# Reference Implementation for the Qwen3-VL (Q3VL) Benchmark

For the MLPerf Inference v6.1 round, benchmarking uses a decoupled load generator client ([endpoints](https://github.com/mlcommons/endpoints#)), a model server (for example, [vLLM](https://github.com/vllm-project/vllm)), and the dataset/configuration described below.

## Quick Start

### Start the model server

The model server can run in its own environment. Using vLLM as example, start vLLM as you would for any standard OpenAI-compatible deployment:

```bash
export MODEL_NAME=Qwen/Qwen3-VL-235B-A22B-Instruct
export HF_TOKEN=<your Hugging Face token>  # Optional for public models
export HF_HOME=<path to Hugging Face cache, e.g. ~/.cache/huggingface>

docker run --runtime nvidia --gpus all \
  -p 8000:8000 \
  --ipc=host \
  --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
  -v ${HF_HOME}:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model ${MODEL_NAME} \
  --tensor-parallel-size 4 \
  --max-model-len=32768 \
  --async-scheduling \
  --limit-mm-per-prompt.video 0 \
  --no-enable-prefix-caching  ## Must have this flag as the rule forbids prefix caching
```

### Set up endpoints

After the server is ready to listen for requests, clone [endpoints](https://github.com/mlcommons/endpoints#) on the same node—or on any host that can reach the server over HTTP. Follow the [endpoints quick start](https://github.com/mlcommons/endpoints/tree/381d13bbd27d6d52306813a51dc4e44295222d7e#quick-start) and install with either **uv**:

```bash
git clone https://github.com/mlcommons/endpoints.git
cd endpoints
uv sync
```

or **pip** in a virtual environment:

```bash
python3.12 -m venv venv && source venv/bin/activate
pip install .
```

### Configure the benchmark

Example configs live under [endpoints/examples/08_Qwen3-VL-235B-A22B_Example](https://github.com/mlcommons/endpoints/tree/381d13bbd27d6d52306813a51dc4e44295222d7e/examples/08_Qwen3-VL-235B-A22B_Example). Set the endpoint URL in the YAML file to match your server address and port:

```yaml
endpoint_config:
  endpoints:
    - "http://localhost:8000"
```

### Run the benchmark

Launch offline or server scenarios:

```bash
uv run inference-endpoint benchmark from-config \
  -c examples/08_Qwen3-VL-235B-A22B_Example/offline_qwen3_vl_235b_a22b_shopify.yaml
```

```bash
uv run inference-endpoint benchmark from-config \
  -c examples/08_Qwen3-VL-235B-A22B_Example/online_qwen3_vl_235b_a22b_shopify.yaml
```

Launch the interactive scenario:

```bash
uv run inference-endpoint benchmark from-config \
  -c examples/08_Qwen3-VL-235B-A22B_Example/interactive_qwen3_vl_235b_a22b_shopify_8k.yaml
```

## Compliance test

Each example benchmark config includes an accuracy test that queries the same server backend. You do not need a separate accuracy-mode run. Reported accuracy must meet the minimum thresholds in [Reference Implementation Specification](#reference-implementation-specification) below.

## Reference Implementation Specification

### v6.1 round

- **vLLM version:** [a65093c](https://github.com/vllm-project/vllm/tree/a65093c1a39a8ddd8455365128ecbe259350e22c)
- **endpoints version:** [381d13bbd27d6d52306813a51dc4e44295222d7e](https://github.com/mlcommons/endpoints/tree/381d13bbd27d6d52306813a51dc4e44295222d7e)
- **Model:**
  - [Qwen/Qwen3-VL-235B-A22B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct)
  - Commit SHA: [710c13861be6c466e66de3f484069440b8f31389](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct/tree/710c13861be6c466e66de3f484069440b8f31389)
- **Dataset:**
  - **Offline/Server scenario:**
    - [Shopify/product-catalogue](https://huggingface.co/datasets/Shopify/product-catalogue)
    - Commit SHA: [d5c517c509f5aca99053897ef1de797d6d7e5aa5](https://huggingface.co/datasets/Shopify/product-catalogue/tree/d5c517c509f5aca99053897ef1de797d6d7e5aa5)
    - Both the `train` and `test` splits are used, concatenated in that order.
    - Total number of samples: `48289`.
  - **Interactive scenario:**
    - [nvidia/Shopify-product-catalogue-8k](https://huggingface.co/datasets/nvidia/Shopify-product-catalogue-8k)
    - Commit SHA: [2bc8c6c4b6ebd27b880b0cba519cb45d09867045](https://huggingface.co/datasets/nvidia/Shopify-product-catalogue-8k/commit/2bc8c6c4b6ebd27b880b0cba519cb45d09867045)
    - Total number of samples: `8000`.
- **Guided decoding:** not used.
- **Sampling parameters:**
  - Frequency penalty: `None` (mathematically equivalent to `0.0`).
  - Presence penalty: `None` (mathematically equivalent to `0.0`).
  - Temperature: `None` (mathematically equivalent to `1.0`).
  - Top-P: `None` (mathematically equivalent to `1.0`).
  - Top-K: `None` (mathematically equivalent to `0`).
  - Min-P: `None` (mathematically equivalent to `0.0`).
  - Repetition penalty: `None` (mathematically equivalent to `1.0`).
- **Constraints:**
  - **Model quality:**
    - **Offline/Server scenario:**
      - Category Hierarchical F1 score ≥ `0.7824`. This is the 99% recovery of `0.7903037`, the mean category hierarchical F1 score across 10 runs on [the BF16 version of the model](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct). The standard deviation across those 10 runs is `0.0002250412555`.
    - **Interactive scenario:**
      - Category Hierarchical F1 score ≥ `0.7799`. This is the 99% recovery of `0.7878`, the mean category hierarchical F1 score across 5 runs on [the BF16 version of the model](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct). The standard deviation across those 5 runs is `0.000535724`.
  - **Server scenario:**
    - Target latency is the constraint (not Time to First Token (TTFT) or Time per Output Token (TPOT)).
    - Target latency percentile: `0.99`.
    - Target latency ≤ 12 seconds.
    - Performance sample count: `48289`.
  - **Offline scenario:**
    - Number of samples in the query ≥ `48289` (every sample in the dataset is sent to the VLM endpoint at least once).
    - Performance sample count: `48289`.
  - **Interactive scenario:**
    - Target latency is the constraint (not TTFT or TPOT).
    - Target latency percentile: `0.99`.
    - Target latency ≤ 1.5 seconds.
    - Performance sample count: `8000`.
  - Testing duration ≥ 10 minutes.
  - Sample concatenation permutation is enabled.
  - You must explicitly set `--no-enable-prefix-caching` for vLLM.


> **MLPerf Inference v6.0 round only.** The following section and the Qwen3-VL reference under `multimodal/qwen3-vl` were maintained for the **v6.0** submission round. They are **deprecated** for newer rounds; use the current MLPerf Inference docs and repository layout for later versions.

# Reference Implementation for the Qwen3-VL (Q3VL) Benchmark 

## Automated command to run the benchmark via MLCFlow

Please see the [new docs site(WIP)]() for an automated way to run this benchmark across different available implementations and do an end-to-end submission with or without docker.

You can also do `pip install mlc-scripts` and then use `mlcr` commands for downloading the model and datasets using the commands given in the later sections.

## Model and Dataset download

### Download model through MLCFlow Automation

```
mlcr get-ml-model-qwen3-vl,_mlc,_r2-downloader,_235b-a22b --outdirname=<Download path> -j
```

### Download dataset through MLCFlow Automation

```
mlcr get-dataset-mlperf-inference-shopify-catalogue,_mlc,_r2-downloader --outdirname=<path_to_download> -j
```

## Quick Start

This guide demonstrates how you can run the benchmark on your local machine.

### Create a Conda environment

Follow [this link](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions)
on how to install Miniconda on your host machine. Then, you can create a new conda 
environment via:

```bash
conda create -n mlperf-inf-mm-q3vl python=3.12
```

### Install the Q3VL benchmarking CLI

#### For users

Install `mlperf-inf-mm-q3vl` with:

```bash
pip install git+https://github.com/mlcommons/inference.git#subdirectory=multimodal/qwen3-vl/
```

#### For developers

Clone the MLPerf Inference repo via:

```bash
git clone --recurse-submodules https://github.com/mlcommons/inference.git mlperf-inference
```

Then enter the repo: 

```bash
cd mlperf-inference/
```

Install `mlperf-inf-mm-q3vl` and the development tools with:

- On Bash
```bash
pip install -e multimodal/qwen3-vl/[dev]
```
- On Zsh
```zsh
pip install -e multimodal/qwen3-vl/"[dev]"
```

### Post Q3VL benchmarking CLI installation 

After installation, you can check the CLI flags that `mlperf-inf-mm-q3vl` can take with:

```bash
mlperf-inf-mm-q3vl --help
```

You can enable shell autocompletion for `mlperf-inf-mm-q3vl` with:

```bash
mlperf-inf-mm-q3vl --install-completion
```

> [!NOTE]
> Shell auto-completion will take effect once you restart the terminal.

### Start an inference endpoint on your local host machine with vLLM

Please refer to [this guide on how to launch vLLM for various Qwen3 VL MoE models](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html).

```bash
docker run --gpus all \                                 # Use all the GPUs on this host machine.
    -v ~/.cache/huggingface:/root/.cache/huggingface \  # Use the HuggingFace cache from your host machine.
    -p 8000:8000 \                                      # This assumes the endpoint will use port 8000.
    --ipc=host \                                        # The container can access and utilize the host's IPC mechanisms (e.g., shared memory).
    vllm/vllm-openai:nightly \                          # You can also use the `:latest` container or a specific release.
        --model Qwen/Qwen3-VL-235B-A22B-Instruct \      # Specifies the model for vLLM to deploy.
        --tensor-parallel-size 8 \                      # 8-way tensor-parallel inference across 8 GPUs.
        --limit-mm-per-prompt.video 0 \                 # The input requests will contain images only (i.e., no videos).
        --no-enable-prefix-caching                      # Disable cross-query prefix caching to satisfy MLPerf Inference rules.
```

### Run the benchmark for the Offline scenario

Performance only mode:

```bash
mlperf-inf-mm-q3vl benchmark endpoint --settings.test.scenario offline --settings.test.mode performance_only
```

Accuracy only mode:

```bash
mlperf-inf-mm-q3vl benchmark endpoint --settings.test.scenario offline --settings.test.mode accuracy_only
```

### Run the benchmark for the Server scenario

Performance only mode:

```bash
mlperf-inf-mm-q3vl benchmark endpoint --settings.test.scenario server --settings.test.mode performance_only
```

Accuracy only mode:

```bash
mlperf-inf-mm-q3vl benchmark endpoint --settings.test.scenario server --settings.test.mode accuracy_only
```

### Pass in `user.conf`

You can pass in a `user.conf` file through `--settings.user_conf.path`, such that the
LoadGen parameters provided through the CLI will be overridden by the `user.conf` 
provided by you and the `mlperf.conf` inside the LoadGen. An example `user.conf` file
is included: [example_user.conf](./example_user.conf). As such, you can run the
benchmark with `user.conf` via:

```bash
mlperf-inf-mm-q3vl benchmark endpoint \
  --settings.test.scenario <scenario> \
  --settings.test.mode <mode> \
  --settings.user_conf.path example_user.conf
```

### Evalute the response quality

You should pass the `mlperf_log_accuracy.json` file (generated by LoadGen) to the
`--filename` flag of the `mlperf-inf-mm-q3vl evaluate` command.

```bash
mlperf-inf-mm-q3vl evaluate --filename output/mlperf_log_accuracy.json
```

The command will generate the file accuracy.txt that you can use in your submission.
Additionally, don't forget to truncate your original file `mlperf_log_accuracy.json` to a size less 
than 10KB for your submission.

## Docker

[docker/](docker/) provides examples of Dockerfiles that install the Q3VL benchmarking
CLI into the container images of the inference engine. This is useful when you have to
run both the inference engine and the Q3VL benchmarking CLI inside the same container,
for example, in a situation where you must use a GPU cluster managed by 
[Slurm](https://slurm.schedmd.com/) with [enroot](https://github.com/nvidia/enroot) and
[pyxis](https://github.com/NVIDIA/pyxis).

As an illustrative example, assuming that you are at the root directory of the MLPerf 
Inference repo:

1. You can build a container image against the vLLM's
`vllm/vllm-openai:v0.12.0` release by

```bash
docker build \
    --build-arg BASE_IMAGE_URL=vllm/vllm-openai:v0.12.0 \
    --build-arg MLPERF_INF_MM_Q3VL_INSTALL_URL=multimodal/qwen3-vl \
    -f multimodal/qwen3-vl/docker/vllm-cuda.Dockerfile \
    -t mlperf-inf-mm-q3vl:vllm-openai-v0.12.0 \
    .
```
> [!NOTE]
> `MLPERF_INF_MM_Q3VL_INSTALL_URL` can also take in a remote GitHub location, such as
> `git+https://github.com/mlcommons/inference.git#subdirectory=multimodal/qwen3-vl/`.

2. Afterwards, you can start the container in the interactive mode by

```bash
docker run --rm -it --gpus all -v ~/.cache:/root/.cache --ipc=host mlperf-inf-mm-q3vl:vllm-openai-v0.12.0
```

### Benchmark against vLLM inside the container

If you are running `mlperf-inf-mm-q3vl` inside a local environment that has access to
vLLM (such as inside a container that was created using the 
[docker/vllm-cuda.Dockerfile](docker/vllm-cuda.Dockerfile)), you can use a single
`mlperf-inf-mm-q3vl benchmark vllm` command to achieve:

1. Deploy an endpoint using vLLM.
2. Wait for the endpoint to be healthy.
3. Run the benchmark against that endpoint.

For example, inside the container, you can run the Offline scenario Accuracy only
mode with:

```bash
mlperf-inf-mm-q3vl benchmark vllm \
    --settings.test.scenario offline \
    --settings.test.mode accuracy_only \
    --settings.user_conf.path example_user.conf \
    --vllm.cli=--async-scheduling \
    --vllm.cli=--max-model-len=32768 \
    --vllm.cli=--max-num-seqs=1024 \
    --vllm.cli=--compilation-config='{
        "cudagraph_capture_sizes": [
            1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128,
            136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248,
            256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480,
            496, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768
        ]
    }' \
    --vllm.cli=--limit-mm-per-prompt.video=0 \
    --vllm.cli=--tensor-parallel-size=8 \
    --vllm.cli=--no-enable-prefix-caching
```

## Slurm

[scripts/slurm/](scripts/slurm/) provide example scripts of running both the benchmark 
and the response quality evaluation in a GPU cluster managed by 
[Slurm](https://slurm.schedmd.com/) with [enroot](https://github.com/nvidia/enroot) and
[pyxis](https://github.com/NVIDIA/pyxis). Specifically,

- [scripts/slurm/benchmark.sh](scripts/slurm/benchmark.sh) is a sbatch script that 
  runs the benchmarking job.
- [scripts/slurm/evaluate.sh](scripts/slurm/evaluate.sh) is a sbatch script that runs
  the evaluation job.
- [scripts/slurm/submit.sh](scripts/slurm/submit.sh) is a Bash script that submits both
  jobs, where the evaluation job would only run if the benchmarking job has succeeded.

You can check the CLI flags that [scripts/slurm/submit.sh](scripts/slurm/submit.sh) can
take via:

```bash
bash submit.sh --help
```

> [!NOTE]
> Slurm clusters are often highly customized per organization. If you are unfamiliar
> with Slurm, you should check with the cluster administrator of your organization
> first, get a good understanding of what those example scripts do, and adapt the 
> example scripts to the specific settings for the Slurm cluster that you are going
> to use, before you try to launch any jobs.

## Prefix caching

According to the [rules of MLPerf Inference](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#94-llm-benchmarks),
cross-query prefix caching is disallowed, while PagedAttention or continuous batching
are allowed. This means that, in:
- in vLLM, you must explicitly set `--no-enable-prefix-caching`;
- in SGLang, you must explicitly set `--disable-radix-cache`.

## Reference Implementation Specification

- v6.0 Round
  - vLLM version: [v0.12.0](https://github.com/vllm-project/vllm/releases/tag/v0.12.0)
  - Model:
    - [Qwen/Qwen3-VL-235B-A22B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct)
    - Commit SHA: [710c13861be6c466e66de3f484069440b8f31389](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct/tree/710c13861be6c466e66de3f484069440b8f31389)
  - Dataset:
    - [Shopify/product-catalogue](https://huggingface.co/datasets/Shopify/product-catalogue)
    - Commit SHA: [d5c517c509f5aca99053897ef1de797d6d7e5aa5](https://huggingface.co/datasets/Shopify/product-catalogue/tree/d5c517c509f5aca99053897ef1de797d6d7e5aa5)
    - Both the `train` and the `test` splits are used and concatenated in that order.
    - Total number of samples: `48289`.
  - Guided decoding is not used.
  - Sampling parameters:
    - Frequency penalty: `None` (mathematically equivalent to `0.0`).
    - Presence penalty: `None` (mathematically equivalent to `0.0`).
    - Temperature: `None` (mathematically equivalent to `1.0`).
    - Top-P: `None` (mathematically equivalent to `1.0`).
    - Top-K: `None` (mathematically equivalent to `0`).
    - Min-P: `None` (mathematically equivalent to `0.0`).
    - Repetition penalty: `None` (mathematically equivalent to `1.0`).
  - Constraints:
    - Model quality:
      - Category Hierarchical F1 Score >= `0.7824`. This is the 99% recovery of 
        `0.7903037` which is the mean category hierarchical F1 score across 10 runs on 
        [the BF16 version of the model](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct).
        The standard deviation across those 10 runs is `0.0002250412555`.
    - Server Scenario:
      - Target latency is used as the constraint, instead of Time to First Token (TTFT)
        or Time per Output Token (TPOT) latencies. 
      - Target latency percentile = `0.99`.
      - Target latency $\le$ 12 seconds.
    - Offline Scenario:
      - Number of samples in the query $\ge$ `48289` (i.e., every sample in the entire
        dataset would be send to the VLM endpoint at least once).
    - Performance sample count: `48289` (i.e., the entire dataset will be loaded into
      the host memory, which takes ~6.39 GB). 
    - Testing duration $\ge$ 10 mins.
    - Sample concatenation permutation is enabled.
    - You must explicitly set `--no-enable-prefix-caching` for vLLM.

## Plugin System for `mlperf-inf-mm-q3vl benchmark`

The `mlperf-inf-mm-q3vl` package supports a plugin system that allows third-party
packages to register additional subcommands under `mlperf-inf-mm-q3vl benchmark`. This
uses Python's standard entry points mechanism.

The purpose of this feature is to allow benchmark result submitters to customize and fit
`mlperf-inf-mm-q3vl` to the inference system that they would like to benchmark,
**without** direct modification to the source code of `mlperf-inf-mm-q3vl` which is
frozen after the benchmark being finalized.

### How it works

1. **Plugin Discovery**: When the CLI starts, it automatically discovers all registered
plugins via the `mlperf_inf_mm_q3vl.benchmark_plugins` entry point group.
2. **Plugin Loading**: Each plugin's entry point function is called to retrieve either a
single command or a Typer app.
3. **Command Registration**: The plugin's commands are automatically added to the
`benchmark` subcommand group.

### Example: creating a `mlperf-inf-mm-q3vl-foo` plugin package for `mlperf-inf-mm-q3vl benchmark foo`

#### Step 1: Package Structure

Create a new Python package with the following structure:

```
mlperf-inf-mm-q3vl-foo/
├── pyproject.toml
└── src/
    └── mlperf_inf_mm_q3vl_foo/
        ├── __init__.py
        ├── schema.py
        ├── deploy.py
        └── plugin.py
```

Note that this is only a minimalistically illustrative example. The users are free to
structure and name their Python packages and modules in any way that they wish. 

#### Step 2: Implement the `mlperf-inf-mm-q3vl-foo` plugin

Create your plugin entry point function in `plugin.py`:

```python
"""Plugin to support benchmarking the Foo inference system."""

from typing import Annotated
from collections.abc import Callable
from loguru import logger
from pydantic_typer import Typer
from typer import Option
from mlperf_inf_mm_q3vl.schema import Settings, Dataset, Endpoint, Verbosity
from mlperf_inf_mm_q3vl.log import setup_loguru_for_benchmark

from .schema import FooEndpoint

def register_foo_benchmark() -> Callable:
    """Entry point for the plugin to benchmark the Foo inference system.
    
    This function is called when the CLI discovers the plugin.
    It should return either:
    - A single command function (decorated with appropriate options)
    - A tuple of (Typer app, command name) for more complex hierarchies
    """

    def benchmark_foo(
        *,
        settings: Settings,
        dataset: Dataset,
        # Add your foo-specific parameters here
        foo: FooEndpoint,
        custom_param: Annotated[
            int,
            Option(help="Custom parameter for foo backend"),
        ] = 2,
        random_seed: Annotated[
            int,
            Option(help="The seed for the random number generator."),
        ] = 12345, 
        verbosity: Annotated[
            Verbosity,
            Option(help="The verbosity level of the logger."),
        ] = Verbosity.INFO,
    ) -> None:
        """Deploy and benchmark using Foo backend.
        
        This command deploys a model using the Foo backend
        and runs the MLPerf benchmark against it.
        """
        from .deploy import FooDeployer

        setup_loguru_for_benchmark(settings=settings, verbosity=verbosity)
        logger.info(
            f"Start to benchmark the Foo inference system with endpoint spec {} and custom param {}",
            foo,
            custom_param,
        )
        # Your implementation here
        with FooDeployer(endpoint=foo, settings=settings, custom_param=custom_param):
            # FooDeployer will make sure that Foo is deployed and currently healthy.
            # Run benchmark using the core run_benchmark function
            run_benchmark(
                settings=settings,
                dataset=dataset,
                endpoint=vllm,
                random_seed=random_seed,
            )

    # Return the command function
    # The entry point name will be used as the subcommand name
    return benchmark_foo
```

#### Step 3: Configure `pyproject.toml`

Register the plugin in its package's `pyproject.toml`:

```toml
[project]
name = "mlperf-inf-mm-q3vl-foo"
version = "0.1.0"
description = "Enable mlperf-inf-mm-q3vl to benchmark the Foo inference system."
requires-python = ">=3.12"
dependencies = [
    "mlperf-inf-mm-q3vl @ git+https://github.com/mlcommons/inference.git#subdirectory=multimodal/qwen3-vl/",
    # Add your backend-specific dependencies here
]

[project.entry-points."mlperf_inf_mm_q3vl.benchmark_plugins"]
# The key here becomes the subcommand name.
foo = "mlperf_inf_mm_q3vl_foo.plugin:register_foo_benchmark"

[build-system]
requires = ["setuptools>=80"]
build-backend = "setuptools.build_meta"
```

#### Step 4: Install and use `mlperf-inf-mm-q3vl benchmark foo`

```bash
# Install your plugin package
pip install mlperf-inf-mm-q3vl-foo

# The new subcommand is now available
mlperf-inf-mm-q3vl benchmark foo --help
mlperf-inf-mm-q3vl benchmark foo \
    --settings-file settings.toml \
    --dataset shopify-global-catalogue \
    --custom-param 3
```

#### Advanced: Nested Subcommands

If you want to create multiple subcommands under a single plugin (e.g.,
`mlperf-inf-mm-q3vl benchmark foo standard` and
`mlperf-inf-mm-q3vl benchmark foo optimized`), return a tuple of `(Typer app, name)`:

```python
def register_foo_benchmark() -> tuple[Typer, str]:
    """Entry point that creates nested subcommands."""
    from pydantic_typer import Typer

    # Create a Typer app for your plugin
    foo_app = Typer(help="Benchmarking options for the Foo inference systems.")

    @foo_app.command(name="standard")
    def foo_standard(...) -> None:
        """Run standard Foo benchmark."""
        # Implementation
        ...

    @foo_app.command(name="optimized")
    def foo_optimized(...) -> None:
        """Run optimized Foo benchmark with max performance."""
        # Implementation
        ...
    
    # Return tuple of (app, command_name)
    return (foo_app, "foo")
```

This will create:
- `mlperf-inf-mm-q3vl benchmark foo standard`
- `mlperf-inf-mm-q3vl benchmark foo optimized`

### Best Practices

1. Dependencies: Declare `mlperf-inf-mm-q3vl` as a dependency in your plugin package.
2. Documentation: Provide clear docstrings for your plugin commands - they appear in
`--help` output.
3. Schema Reuse: Reuse the core `Settings`, `Dataset`, and other schemas from
`mlperf_inf_mm_q3vl.schema` for consistency and minimizing boilerplate code.
4. Lazy Imports: If your plugin has heavy dependencies, import them inside functions
rather than at module level to avoid slowing down CLI startup

## Developer Guide

### Linting

You can lint the Q3VL benchmark source code by running the following script:

```bash
bash multimodal/qwen3-vl/scripts/linters.sh
```