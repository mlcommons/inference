# Text-to-Video Benchmark

Text-to-video generation using Wan2.2 T2V-A14B-Diffusers model and VBench evaluation.

## Automated command to run the benchmark via MLCFlow

Please see the [new docs site(WIP)]() for an automated way to run this benchmark across different available implementations and do an end-to-end submission with or without docker.

You can also do `pip install mlc-scripts` and then use `mlcr` commands for downloading the model and datasets using the commands given in the later sections.

## Model and Dataset download

### Download model through MLCFlow Automation

```
mlcr get-ml-model-wan2,_mlc,_r2-downloader,_wan2_2_t2v_a14b --outdirname=<Download path> -j
```

### Download dataset through MLCFlow Automation

```
mlcr get-dataset-mlperf-inference-text-to-video,_mlc,_r2-downloader --outdirname=<path_to_download> -j
```

## Quick Start

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/mlcommons/inference.git
cd inference/text_to_video

# Build Docker
./launch.sh --build

# Download model
./launch.sh python3 download_model.py

# Run inference (supports data parallel)
./launch.sh python -m torch.distributed.run --nproc_per_node=8 run_inference.py

# Evaluate (set GPU_IDS to 0 when running the compliance test as a workaround for this bug: https://github.com/Vchitect/VBench/pull/141)
./launch.sh python run_evaluation.py
```

## Files

- `inference_config.yaml` - Generation parameters (resolution, fps, seed, etc.)
- `download_model.py` - Model download
- `run_inference.py` - Video generation  
- `run_evaluation.py` - VBench evaluation
- `run_mlperf.py` - Run mlperf loadgen using Wan2.2 T2V-A14B-Diffusers model
- `launch.sh` - Docker launcher
- `data/vbench_prompts.txt` - Input prompts
- `data/fixed_latent.pt` - Fixed latent tensor for deterministic generation

## Supported Hardware

This implementation supports NVIDIA GPUs with CUDA 12.1 compatibility which includes Hopper and pre-Hopper architectures.

## Accuracy

### VBench Reference Score (BF16)

| Metric | Value |
|--------|-------|
| Reference Accuracy | **70.48** |
| Accuracy Threshold (99%) | **69.7752** |

The accuracy threshold is set at 99% of the reference BF16 accuracy score.

## References

- Model: [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) (commit: `5be7df9619b54f4e2667b2755bc6a756675b5cd7`)
- VBench: [GitHub](https://github.com/Vchitect/VBench)
- MLPerf: [Inference](https://github.com/mlcommons/inference)
