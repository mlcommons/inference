# Instructions and documentation on VLLM Backend 


## Run with VLLM backend
```
python run_mlperf.py --backend vllm --scenario offline --input-file data/accuracy_eval_tokenized.pkl
```

## Specify custom vLLM server URL

### Run accuracy 
```
python run_mlperf.py --backend vllm --server-url http://localhost:8000 --scenario server --input-file data/accuracy_eval_tokenized.pkl
```
### Run performance 
```
python run_mlperf.py --backend vllm --server-url http://localhost:8000 --scenario offline --input-file perf_eval_ref.parquet --mlperf-conf mlperf_gptoss.conf --max-new-tokens 10240 --max-concurrency 6396
```

## For evaluating accuracy . 

[!ALERT] Best to create this in an environment 
```
# Clone along with submodules to clone the livecodebench submodule 
git clone --recurse-submodule mlperf-inference-6.0

# Run setup.sh
./setup.sh

# Install transformers to the version in vllm 
pip install transformers==4.57.3

# Change directory to gpt-oss-120b and run the eval_mlperf_accuracy as suggested 

```

## Sampling parameters 

 The max_tokens are set to **10240** for Perf and **32768** for accuracy
 The other sampling paramaters are : 
 - temperature: 1.0
 - top_p: 1.0
 - top_k: -1


## Dataset 

### Performance Dataset
The performance dataset has **6396** samples
The Perf dataset has the following columns 
1. prompt
2. dataset
3. input_tokens
4. num_tokens
5. **text_input**

### Accuracy Dataset
The performance dataset has **4395** samples
The accuracy dataset has the following columns 
1. original_messages
2. ground_truth
3. dataset
4. input_tokens
5. num_tokens
6. text_input

