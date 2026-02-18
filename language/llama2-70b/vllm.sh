CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m vllm.entrypoints.openai.api_server \
  --model /share/mlperf_sets/model/llama-2-70b-chat-hf.uri \
  --tensor-parallel-size 4
