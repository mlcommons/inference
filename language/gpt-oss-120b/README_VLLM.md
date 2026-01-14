# Run with VLLM backend
python run_mlperf.py --backend vllm --scenario offline --input-file data/accuracy_eval_tokenized.pkl

# Specify custom vLLM server URL
python run_mlperf.py --backend vllm --server-url http://localhost:8000 --scenario server --input-file data/accuracy_eval_tokenized.pkl
