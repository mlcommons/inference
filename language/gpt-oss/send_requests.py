#!/usr/bin/env python3
"""
Script to send text requests to SGLang server with tokenization.
"""

import numpy as np
import pandas as pd
import pickle
import requests
import json
import time
import argparse
from typing import List, Dict, Any
import logging
from transformers import AutoTokenizer
from multiprocessing import Pool
from functools import partial

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
        # SGLang format with input_ids
        payload = {
            "input_ids": input_ids,
            "sampling_params": {
                "max_new_tokens": max_tokens,
                "temperature": 0.0
            }
        }

        try:
            response = self.session.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=1200
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(
                    f"Request failed with status {response.status_code}: {response.text}")
                return {"error": f"HTTP {response.status_code}: {response.text}"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return {"error": str(e)}


def load_text_data(data_file: str) -> pd.DataFrame:
    """Load the text data from pickle file."""
    logger.info(f"Loading data from {data_file}")
    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    logger.info(f"Loaded {len(data)} samples")
    logger.info(f"Columns: {list(data.columns)}")
    logger.info(f"First text input length: {len(data.iloc[0]['text_input'])}")

    return data


def load_tokenizer(model_name: str):
    """Load tokenizer for the specified model."""
    logger.info(f"Loading tokenizer for {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Tokenizer loaded successfully")
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise


def tokenize_all_inputs(data: pd.DataFrame, tokenizer,
                        max_samples: int = None):
    """Tokenize all text inputs at once."""
    num_samples = min(len(data), max_samples) if max_samples else len(data)
    logger.info(f"Tokenizing {num_samples} text inputs...")

    text_inputs = data['text_input'].tolist()[:num_samples]

    # Tokenize all texts at once
    tokenized = tokenizer(
        text_inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        padding_side="right")
    input_ids_list = [tokenized['input_ids'][i].tolist()
                      for i in range(num_samples)]

    logger.info(
        f"Tokenization complete. Token lengths: {[len(ids) for ids in input_ids_list[:5]]}...")
    return input_ids_list, text_inputs


def send_single_request(args_tuple):
    """Send a single request - used by multiprocessing pool."""
    input_ids, max_tokens, server_url, sample_id = args_tuple

    # Create a new client for this process
    client = SGLangClient(server_url)

    try:
        response = client.send_request(input_ids, max_tokens=max_tokens)
        return sample_id, response
    except Exception as e:
        logger.error(f"Request {sample_id} failed: {e}")
        return sample_id, {"error": str(e)}


def send_requests_parallel(input_ids_list: List[List[int]], server_url: str,
                           max_tokens: int = 100, max_concurrency: int = 128) -> List[Dict[str, Any]]:
    """Send all requests to SGLang server in parallel using multiprocessing."""
    num_samples = len(input_ids_list)
    logger.info(
        f"Sending {num_samples} requests to server with {max_concurrency} concurrent workers...")

    # Prepare arguments for multiprocessing
    args_list = [
        (input_ids, max_tokens, server_url, i)
        for i, input_ids in enumerate(input_ids_list)
    ]

    start_time = time.time()

    # Use multiprocessing pool
    with Pool(processes=min(max_concurrency, num_samples)) as pool:
        # Map the function to all arguments
        results = pool.map(send_single_request, args_list)

    # Sort results by sample_id to maintain order
    results.sort(key=lambda x: x[0])
    responses = [result[1] for result in results]

    total_time = time.time() - start_time
    logger.info(
        f"Completed {num_samples} requests in {total_time:.2f} seconds")
    logger.info(f"Average rate: {num_samples/total_time:.2f} requests/sec")

    return responses


def detokenize_all_responses(responses: List[Dict[str, Any]], input_ids_list: List[List[int]],
                             tokenizer) -> List[str]:
    """Detokenize all responses at once."""
    logger.info("Detokenizing responses...")

    response_texts = []
    for i, (response, input_ids) in enumerate(zip(responses, input_ids_list)):
        response_text = ""
        if "error" not in response and "generated_text" in response:
            try:
                # Extract generated tokens (excluding input tokens)
                generated_tokens = response["generated_text"][len(input_ids):]
                response_text = tokenizer.decode(
                    generated_tokens, skip_special_tokens=True)
            except Exception as e:
                logger.warning(
                    f"Failed to decode response for sample {i+1}: {e}")
        response_texts.append(response_text)

    logger.info("Detokenization complete")
    return response_texts


def save_responses(responses: List[Dict[str, Any]], response_texts: List[str],
                   text_inputs: List[str], input_ids_list: List[List[int]],
                   output_file: str) -> None:
    """Save all responses to file."""
    logger.info(f"Saving responses to {output_file}...")

    with open(output_file, 'w') as f:
        for i, (response, response_text, text_input, input_ids) in enumerate(
                zip(responses, response_texts, text_inputs, input_ids_list)):

            response_data = {
                "sample_id": int(i),
                "text_input": text_input,
                "input_length": len(text_input),
                "token_length": len(input_ids),
                "input_tokens": input_ids,
                "response": response,
                "response_text": response_text,
                "timestamp": float(time.time())
            }

            f.write(json.dumps(response_data) + '\n')

    logger.info(f"Responses saved to {output_file}")


def process_requests(data: pd.DataFrame, tokenizer, server_url: str,
                     max_samples: int = None, max_tokens: int = 100,
                     max_concurrency: int = 128, output_file: str = "responses.jsonl") -> None:
    """Main processing function that handles tokenization, requests, and detokenization."""

    # Step 1: Tokenize all inputs
    input_ids_list, text_inputs = tokenize_all_inputs(
        data, tokenizer, max_samples)

    # Step 2: Send all requests in parallel
    responses = send_requests_parallel(
        input_ids_list,
        server_url,
        max_tokens,
        max_concurrency)

    # Step 3: Detokenize all responses
    response_texts = detokenize_all_responses(
        responses, input_ids_list, tokenizer)

    # Step 4: Save all results
    save_responses(
        responses,
        response_texts,
        text_inputs,
        input_ids_list,
        output_file)


def main():
    parser = argparse.ArgumentParser(
        description="Send text requests to SGLang server with tokenization")
    parser.add_argument("--data-file", default="/home/mlperf_inference_storage/data/deepseek-r1/mlperf_deepseek_r1_dataset_4388_fp8_eval.pkl",
                        help="Path to pickle file containing text data")
    parser.add_argument("--model-name", required=True,
                        help="Model name for tokenizer (e.g., openai/gpt-oss-120b)")
    parser.add_argument("--server-url", default="http://localhost:30000",
                        help="SGLang server URL (default: http://localhost:30000)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to process (default: all)")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Maximum tokens to generate per request")
    parser.add_argument("--max-concurrency", type=int, default=128,
                        help="Maximum number of concurrent requests (default: 128)")
    parser.add_argument("--output", default="responses.jsonl",
                        help="Output file for responses")

    args = parser.parse_args()

    # Test connection
    logger.info(f"Testing server connection to {args.server_url}...")
    test_client = SGLangClient(args.server_url)
    test_response = test_client.send_request([1, 2, 3], max_tokens=5)
    if "error" in test_response:
        logger.error(f"Server connection failed: {test_response['error']}")
        logger.error("Make sure your SGLang server is running. Try:")
        logger.error(
            "  python -m sglang.launch_server --model-path openai/gpt-oss-120b --mem-fraction-static 0.98 --tp 8")
        return
    logger.info("Server connection successful")

    data = load_text_data(args.data_file)
    tokenizer = load_tokenizer(args.model_name)

    process_requests(data, tokenizer, args.server_url,
                     max_samples=args.max_samples,
                     max_tokens=args.max_tokens,
                     max_concurrency=args.max_concurrency,
                     output_file=args.output)


if __name__ == "__main__":
    main()
