# Text-to-Video Benchmark

Text-to-video generation using Wan2.2 T2V-A14B-Diffusers model and VBench evaluation.

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

# Evaluate
./launch.sh python run_evaluation.py
```

## Files

- `inference_config.yaml` - Generation parameters (resolution, fps, seed, etc.)
- `download_model.py` - Model download
- `run_inference.py` - Video generation  
- `run_evaluation.py` - VBench evaluation
- `launch.sh` - Docker launcher
- `data/vbench_prompts.txt` - Input prompts
- `data/fixed_latent.pt` - Optional fixed latent tensor for deterministic generation

## References

- Model: [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers)
- VBench: [GitHub](https://github.com/Vchitect/VBench)
- MLPerf: [Inference](https://github.com/mlcommons/inference)
