# SGLang Request Sender

This script sends preprocessed deepseek-r1 requests to a running SGLang server.

## Usage

```bash
python send_requests.py --server-url http://localhost:30000
```

### Process only first 100 samples:
```bash
python send_requests.py --max-samples 100 --auto-detect
```

### Custom output file and max tokens:
```bash
python send_requests.py --output my_responses.jsonl --max-tokens 50 --auto-detect
```

## Arguments

- `--data-dir`: Directory containing preprocessed data (default: `/home/mlperf_inference_storage/preprocessed_data/deepseek-r1/`)
- `--server-url`: SGLang server URL (e.g., `http://localhost:30000`)
- `--max-samples`: Maximum number of samples to process (default: all 4388 samples)
- `--max-tokens`: Maximum tokens to generate per request (default: 100)
- `--output`: Output file for responses (default: `responses.jsonl`)
- `--auto-detect`: Auto-detect server port

## Output Format

The script outputs a JSONL file where each line contains:
```json
{
  "sample_id": 0,
  "input_length": 1283,
  "input_tokens": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
  "response": {
    "choices": [...],
    "usage": {...}
  },
  "timestamp": 1695821234.567
}
```

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Notes

- The script automatically trims padding from input sequences based on actual lengths
- It tries multiple request formats to ensure compatibility with SGLang
- Responses are saved incrementally to avoid data loss
- Progress is logged every 10 samples
