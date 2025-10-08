#!/usr/bin/env python3
"""
Script to send pre-tokenized requests to SGLang server.

Usage:
    python send_requests.py --input-tokens tokenized_data.pkl [options]

Arguments:
    --input-tokens     Path to pickle file containing pre-tokenized data from harmony-tokens.py
    --server-url       SGLang server URL (default: http://umbriel-b200-145:30000)
    --max-samples      Maximum number of samples to process (default: all)
    --max-tokens       Maximum tokens to generate per request (default: 100)
    --max-concurrency  Maximum number of concurrent requests (default: 128)
    --output           Output pickle file for responses (optional)
"""

import requests
import json
import time
import argparse
from typing import List, Dict, Any
import logging
from multiprocessing import Pool
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize tokenizer
MODEL_NAME = "openai/gpt-oss-120b"
tokenizer = None


def get_tokenizer():
    """Get or initialize the tokenizer."""
    global tokenizer
    if tokenizer is None:
        logger.info(f"Loading tokenizer for {MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        logger.info("Tokenizer loaded successfully")
    return tokenizer


class SGLangClient:
    def __init__(self, base_url: str = "http://localhost:30000",
                 temperature: float = 0.001, top_k: int = 1):
        self.base_url = base_url
        self.session = requests.Session()
        self.temperature = temperature
        self.top_k = top_k

    def send_request(
            self, input_ids: List[int], max_tokens: int = 100) -> Dict[str, Any]:
        """Send a single request to the SGLang server."""
        # SGLang format with input_ids
        payload = {
            "input_ids": input_ids,
            "sampling_params": {
                "max_new_tokens": max_tokens,
                "temperature": self.temperature,
                "top_k": self.top_k,
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


def load_tokenized_data(data_file: str) -> pd.DataFrame:
    """Load pre-tokenized data from pickle file produced by harmony-tokens.py."""
    logger.info(f"Loading tokenized data from {data_file}")

    # Load DataFrame from pickle
    df = pd.read_pickle(data_file)
    logger.info(f"Loaded DataFrame with shape: {df.shape}")

    # Check if tok_input column exists and has valid data
    if 'tok_input' in df.columns:
        # Check for any None values in tok_input (indicating failed
        # tokenization)
        failed_mask = df['tok_input'].isna()
        failed_count = failed_mask.sum()

        if failed_count > 0:
            failed_indices = df[failed_mask].index.unique()
            error_msg = f"Found {failed_count} failed tokenized samples at indices: {failed_indices.tolist()}"
            logger.error(error_msg)
            raise AssertionError(error_msg)

        # Check first sample
        first_tokens = df.iloc[0]['tok_input']
        if isinstance(first_tokens, list):
            logger.info(f"First sample token length: {len(first_tokens)}")
        else:
            logger.warning(
                "tok_input column exists but first sample is not a list")

        logger.info(f"All {len(df)} samples were successfully tokenized")
    else:
        logger.warning("No 'tok_input' column found in DataFrame")

    return df


def send_single_request(args_tuple):
    """Send a single request - used by multiprocessing pool."""
    input_ids, max_tokens, server_url, sample_id, temperature, top_k = args_tuple

    # Create a new client for this process
    client = SGLangClient(server_url, temperature=temperature, top_k=top_k)

    try:
        response = client.send_request(input_ids, max_tokens=max_tokens)
        return sample_id, response
    except Exception as e:
        logger.error(f"Request {sample_id} failed: {e}")
        return sample_id, {"error": str(e)}


def send_requests_parallel(tokenized_df: pd.DataFrame, server_url: str,
                           max_tokens: int = 100, max_concurrency: int = 128, temperature: float = 0.001, top_k: int = 1) -> List[Dict[str, Any]]:
    """Send all requests to SGLang server in parallel using multiprocessing."""
    num_samples = len(tokenized_df)
    logger.info(
        f"Sending {num_samples} requests to server with {max_concurrency} concurrent workers...")

    # Prepare arguments for multiprocessing
    args_list = [
        (row['tok_input'], max_tokens, server_url, idx, temperature, top_k)
        for idx, row in tokenized_df.iterrows()
    ]

    start_time = time.time()

    # Use multiprocessing pool with progress bar
    with Pool(processes=min(max_concurrency, num_samples)) as pool:
        # Map the function to all arguments with progress bar
        results = list(tqdm(
            pool.imap(send_single_request, args_list),
            total=len(args_list),
            desc="Sending requests",
            unit="request"
        ))

    # Sort results by sample_id to maintain order
    results.sort(key=lambda x: x[0])
    responses = [result[1] for result in results]

    total_time = time.time() - start_time
    logger.info(
        f"Completed {num_samples} requests in {total_time:.2f} seconds")
    logger.info(f"Average rate: {num_samples/total_time:.2f} requests/sec")

    return responses


def extract_response_ids(
        responses: List[Dict[str, Any]], tokenized_df: pd.DataFrame) -> List[List[int]]:
    """Extract response output_ids from SGLang responses."""
    logger.info("Extracting response output_ids...")

    response_ids = []
    for i, (response, (_, row)) in enumerate(tqdm(zip(responses, tokenized_df.iterrows()),
                                                  total=len(responses),
                                                  desc="Extracting responses",
                                                  unit="response")):
        response_id = []
        if "error" not in response and "output_ids" in response:
            try:
                # SGLang returns the generated token IDs in the 'output_ids'
                # field
                response_id = response["output_ids"]
            except Exception as e:
                logger.warning(
                    f"Failed to extract response for sample {i+1}: {e}")
        response_ids.append(response_id)

    logger.info("Response output_ids extraction complete")
    return response_ids


def detokenize_output_ids(response_ids: List[List[int]]) -> List[str]:
    """Detokenize output_ids back to text using AutoTokenizer."""
    logger.info("Detokenizing output_ids to text...")

    tokenizer = get_tokenizer()
    detokenized_texts = []

    for i, token_ids in enumerate(
            tqdm(response_ids, desc="Detokenizing outputs", unit="output")):
        try:
            # Detokenize the token IDs back to text
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            detokenized_texts.append(text)
        except Exception as e:
            logger.warning(
                f"Failed to detokenize output for sample {i+1}: {e}")
            detokenized_texts.append("")

    logger.info("Output detokenization complete")
    return detokenized_texts


def save_responses(responses: List[Dict[str, Any]], response_ids: List[List[int]],
                   detokenized_texts: List[str], tokenized_df: pd.DataFrame,
                   output_file: str = None) -> pd.DataFrame:
    """Save all responses to DataFrame and optionally to pickle file."""
    logger.info("Processing responses and updating DataFrame...")

    # Work with the original DataFrame
    result_df = tokenized_df.copy()

    # Overwrite existing columns with server response data
    result_df['model_output'] = detokenized_texts  # Detokenized text output
    # Original output_ids from SGLang
    result_df['tok_model_output'] = response_ids
    result_df['tok_model_output_len'] = [
        len(token_ids) for token_ids in response_ids]  # Length of output_ids

    # Calculate output token lengths for logging
    output_token_lengths = []
    for i, (response, response_ids) in enumerate(
            zip(responses, response_ids)):
        if "error" not in response and "meta_info" in response:
            try:
                # Use the completion_tokens from meta_info
                output_token_lengths.append(
                    response["meta_info"]["completion_tokens"])
            except Exception as e:
                logger.warning(
                    f"Failed to calculate output tokens for sample {i+1}: {e}")
                output_token_lengths.append(len(response_ids))
        else:
            output_token_lengths.append(len(response_ids))

    logger.info(f"Updated DataFrame with shape: {result_df.shape}")
    logger.info(
        f"Updated columns: model_output, tok_model_output, tok_model_output_len")
    logger.info(
        f"Average output token length: {sum(output_token_lengths)/len(output_token_lengths):.1f}")

    # Save to pickle file if output_file is provided
    if output_file:
        logger.info(f"Saving responses to {output_file}...")
        result_df.to_pickle(output_file)
        logger.info(f"Responses saved to {output_file}")

    return result_df


def process_requests(tokenized_df: pd.DataFrame, server_url: str,
                     max_samples: int = None, max_tokens: int = 100,
                     max_concurrency: int = 128, output_file: str = None, temperature: float = 0.001, top_k: int = 1) -> pd.DataFrame:
    """Main processing function that handles requests and response extraction."""

    # Step 1: Limit samples if specified
    if max_samples is not None:
        tokenized_df = tokenized_df.head(max_samples)
        logger.info(f"Limited to first {max_samples} samples")

    # Step 2: Send all requests in parallel
    responses = send_requests_parallel(
        tokenized_df,
        server_url,
        max_tokens,
        max_concurrency,
        temperature,
        top_k)

    # Step 3: Extract response output_ids
    response_ids = extract_response_ids(responses, tokenized_df)

    # Step 4: Detokenize output_ids to text for model_output
    detokenized_texts = detokenize_output_ids(response_ids)

    # Step 5: Save all results and return DataFrame
    result_df = save_responses(
        responses,
        response_ids,
        detokenized_texts,
        tokenized_df,
        output_file)

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Send pre-tokenized requests to SGLang server")
    parser.add_argument("--input-tokens", required=True,
                        help="Path to pickle file containing pre-tokenized data from harmony-tokens.py")
    parser.add_argument("--server-url", default="http://localhost:30000",
                        help="SGLang server URL (default: http://localhost:30000)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to process (default: all)")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Maximum tokens to generate per request")
    parser.add_argument("--max-concurrency", type=int, default=256,
                        help="Maximum number of concurrent requests (default: 128)")
    parser.add_argument("--output", default=None,
                        help="Output pickle file for responses (optional)")
    parser.add_argument("--temperature", type=float, default=0.001,
                        help="Temperature for sampling (default: 0.001)")
    parser.add_argument("--top-k", type=int, default=1,
                        help="Top-k for sampling (default: 1)")

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

    # Load pre-tokenized data
    tokenized_df = load_tokenized_data(args.input_tokens)

    # Process requests and get result DataFrame
    result_df = process_requests(tokenized_df, args.server_url,
                                 max_samples=args.max_samples,
                                 max_tokens=args.max_tokens,
                                 max_concurrency=args.max_concurrency,
                                 output_file=args.output,
                                 temperature=args.temperature,
                                 top_k=args.top_k)

    # Print summary
    logger.info(f"\nProcessing completed:")
    logger.info(f"  - Total samples processed: {len(result_df)}")
    logger.info(
        f"  - Average input token length: {result_df['tok_input_len'].mean():.1f}")
    logger.info(
        f"  - Average output text length: {result_df['tok_model_output_len'].mean():.1f}")
    if args.output:
        logger.info(f"  - Results saved to: {args.output}")
    else:
        logger.info("  - Results returned as DataFrame (not saved to file)")


if __name__ == "__main__":
    main()
