#!/bin/bash

for var in $(compgen -v | grep '^SLURM_'); do unset "$var"; done

model_path=openai/gpt-oss-120b
extra_args=""
output_dir=./data

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            model_path=$2
            shift 2
            ;;
	--output_dir)
	    output_dir=$2
	    shift 2
	    ;;
        *)
	    extra_args="$extra_args $2"
            ;;
    esac
done


cat <<EOF > config.yml
enable_attention_dp: false
enable_autotuner: false
cuda_graph_config:
    max_batch_size: 256
    enable_padding: true
      # speculative_config:
      #     decoding_type: Eagle
      #     max_draft_len: 3
      #     speculative_model_dir: 
      #     eagle3_layers_to_capture: [-1]
kv_cache_config:
    enable_block_reuse: false
    free_gpu_memory_fraction: 0.9
moe_config: 
    backend: TRTLLM
print_iter_log: true
EOF


gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

set -x;

for ((gpu=0; gpu<gpu_count; gpu++)); do
    CUDA_VISIBLE_DEVICES=$gpu TRTLLM_ENABLE_PDL=1 trtllm-serve $model_path --host 0.0.0.0 --port 3000$gpu --backend pytorch --max_batch_size 256 --tp_size 1 --ep_size 1 --trust_remote_code --extra_llm_api_options config.yml $extra_args & > $output_dir/trtllm-serve-$gpu 2>&1
done

# num_servers=2
# CUDA_VISIBLE_DEVICES=0,1,2,3 TRTLLM_ENABLE_PDL=1 trtllm-serve $model_path --host 0.0.0.0 --port 30000 --backend pytorch --max_batch_size 1024 --tp_size 4 --ep_size 1 --trust_remote_code --extra_llm_api_options config.yml $extra_args & > $output_dir/trtllm-serve-0.log 2>&1
# CUDA_VISIBLE_DEVICES=4,5,6,7 TRTLLM_ENABLE_PDL=1 trtllm-serve $model_path --host 0.0.0.0 --port 30001 --backend pytorch --max_batch_size 1024 --tp_size 4 --ep_size 1 --trust_remote_code --extra_llm_api_options config.yml $extra_args & > $output_dir/trtllm-serve-1.log 2>&1

wait

