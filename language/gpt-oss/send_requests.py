#!/usr/bin/env python3
"""
Script to send preprocessed deepseek-r1 requests to SGLang server.
"""

import numpy as np
import requests
import json
import time
import argparse
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SGLangClient:
    def __init__(self, base_url: str = "http://localhost:30000"):
        self.base_url = base_url
        self.session = requests.Session()

    def send_request(
            self, input_ids: List[int], max_tokens: int = 100) -> Dict[str, Any]:
        """Send a single request to the SGLang server."""
        # Try different payload formats for SGLang
        payloads_to_try = [
            # Format 1: Direct token IDs in messages
            {
                "model": "gpt-oss-120b",
                "messages": [
                    {
                        "role": "user",
                        "content": input_ids
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": False
            },
            # Format 2: Text-based content
            {
                "model": "gpt-oss-120b",
                "messages": [
                    {
                        "role": "user",
                        # Truncate for display
                        "content": f"Token IDs: {input_ids[:10]}..."
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": False
            },
            # Format 3: SGLang specific format
            {
                "text": input_ids,
                "sampling_params": {
                    "max_new_tokens": max_tokens,
                    "temperature": 0.0
                }
            }
        ]

        endpoints_to_try = [
            "/v1/chat/completions",
            "/generate",
            "/v1/completions"
        ]

        for payload in payloads_to_try:
            for endpoint in endpoints_to_try:
                try:
                    response = self.session.post(
                        f"{self.base_url}{endpoint}",
                        json=payload,
                        timeout=60
                    )
                    if response.status_code == 200:
                        return response.json()
                    else:
                        logger.debug(
                            f"Endpoint {endpoint} returned {response.status_code}")
                except requests.exceptions.RequestException as e:
                    logger.debug(f"Request to {endpoint} failed: {e}")
                    continue

        return {"error": "All request formats failed"}


def load_preprocessed_data(data_dir: str) -> tuple:
    """Load the preprocessed data files."""
    input_ids_path = f"{data_dir}/input_ids_padded.npy"
    input_lens_path = f"{data_dir}/input_lens.npy"

    logger.info(f"Loading data from {data_dir}")
    input_ids = np.load(input_ids_path)
    input_lens = np.load(input_lens_path)

    logger.info(f"Loaded {len(input_ids)} samples")
    logger.info(f"Input shape: {input_ids.shape}")
    logger.info(f"Lengths range: {input_lens.min()} - {input_lens.max()}")

    return input_ids, input_lens


def trim_padding(input_ids: np.ndarray, actual_length: int) -> List[int]:
    """Trim padding from input_ids based on actual length."""
    return input_ids[:actual_length].tolist()


def send_requests(client: SGLangClient, input_ids: np.ndarray, input_lens: np.ndarray,
                  max_samples: int = None, max_tokens: int = 100,
                  output_file: str = "responses.jsonl") -> None:
    """Send requests to SGLang server and save responses."""

    num_samples = min(
        len(input_ids),
        max_samples) if max_samples else len(input_ids)
    logger.info(f"Sending {num_samples} requests")

    responses = []
    start_time = time.time()

    with open(output_file, 'w') as f:
        for i in range(num_samples):
            # Trim padding based on actual length
            actual_length = input_lens[i]
            trimmed_input = trim_padding(input_ids[i], actual_length)

            logger.info(
                f"Processing sample {i+1}/{num_samples} (length: {actual_length})")

            # Send request
            response = client.send_request(
                trimmed_input, max_tokens=max_tokens)

            # Prepare response data
            response_data = {
                "sample_id": i,
                "input_length": actual_length,
                # First 10 tokens for reference
                "input_tokens": trimmed_input[:10],
                "response": response,
                "timestamp": time.time()
            }

            # Save to file immediately
            f.write(json.dumps(response_data) + '\n')
            f.flush()

            responses.append(response_data)

            # Log progress
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                logger.info(
                    f"Processed {i+1}/{num_samples} samples ({rate:.2f} samples/sec)")

    total_time = time.time() - start_time
    logger.info(
        f"Completed {num_samples} requests in {total_time:.2f} seconds")
    logger.info(f"Average rate: {num_samples/total_time:.2f} requests/sec")
    logger.info(f"Responses saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Send preprocessed requests to SGLang server")
    parser.add_argument("--data-dir", default="/home/mlperf_inference_storage/preprocessed_data/deepseek-r1/",
                        help="Directory containing preprocessed data")
    parser.add_argument("--server-url", default="http://localhost:30000",
                        help="SGLang server URL (default: http://localhost:30000)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to process (default: all)")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Maximum tokens to generate per request")
    parser.add_argument("--output", default="responses.jsonl",
                        help="Output file for responses")

    args = parser.parse_args()

    # Determine server URL
    server_url = args.server_url

    # Load data
    input_ids, input_lens = load_preprocessed_data(args.data_dir)

    # Create client
    client = SGLangClient(server_url)

    # Test connection
    logger.info(f"Testing server connection to {server_url}...")
    test_response = client.send_request([1, 2, 3], max_tokens=5)
    if "error" in test_response:
        logger.error(f"Server connection failed: {test_response['error']}")
        logger.error("Make sure your SGLang server is running. Try:")
        logger.error(
            "  python -m sglang.launch_server --model-path openai/gpt-oss-120b --mem-fraction-static 0.98 --tp 8")
        return
    logger.info("Server connection successful")

    # Send requests
    send_requests(client, input_ids, input_lens,
                  max_samples=args.max_samples,
                  max_tokens=args.max_tokens,
                  output_file=args.output)


if __name__ == "__main__":
    main()
