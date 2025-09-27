# SGLang Request Sender

This script loads text data from a pickle file, tokenizes it using a specified model's tokenizer, sends requests to a running SGLang server, and converts responses back to text.

## Usage

### Basic usage:
```bash
python send_requests.py --model-name openai/gpt-oss-120b
```

### Process only first 100 samples:
```bash
python send_requests.py --model-name openai/gpt-oss-120b --max-samples 100
```

### Custom data file and output:
```bash
python send_requests.py --model-name openai/gpt-oss-120b --data-file /path/to/data.pkl --output my_responses.jsonl
```

### Custom max tokens and server URL:
```bash
python send_requests.py --model-name openai/gpt-oss-120b --max-tokens 50 --server-url http://localhost:8000
```

### Custom concurrency level:
```bash
python send_requests.py --model-name openai/gpt-oss-120b --max-concurrency 64
```

## Arguments

- `--data-file`: Path to pickle file containing text data (default: `/home/mlperf_inference_storage/data/deepseek-r1/mlperf_deepseek_r1_dataset_4388_fp8_eval.pkl`)
- `--model-name`: Model name for tokenizer (required, e.g., `openai/gpt-oss-120b`)
- `--server-url`: SGLang server URL (default: `http://localhost:30000`)
- `--max-samples`: Maximum number of samples to process (default: all)
- `--max-tokens`: Maximum tokens to generate per request (default: 100)
- `--max-concurrency`: Maximum number of concurrent requests (default: 128)
- `--output`: Output file for responses (default: `responses.jsonl`)

## Output Format

The script outputs a JSONL file where each line contains:
```json
{
  "sample_id": 0,
  "text_input": "Here are some example problems...",
  "input_length": 1283,
  "token_length": 512,
  "input_tokens": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
  "response": {
    "generated_text": [1, 2, 3, ...],
    "usage": {...}
  },
  "response_text": "The answer is 6...",
  "timestamp": 1695821234.567
}
```

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Notes

- The script loads text data from a pandas DataFrame in the pickle file
- It uses the specified model's tokenizer to convert text to tokens
- Sends tokenized input to SGLang server via `/generate` endpoint in parallel using multiprocessing
- Converts response tokens back to text using the same tokenizer
- Uses configurable concurrency (default: 128 concurrent requests)
- Each process creates its own HTTP client to avoid connection issues
- Results are maintained in order despite parallel processing
- Progress is logged during processing