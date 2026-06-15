#!/usr/bin/env python3
"""
Script to send pre-tokenized requests to SGLang server.

Usage:
    python run_infer.py --input-tokens tokenized_data.pkl [options]

Arguments:
    --input-tokens     Path to pickle file containing pre-tokenized data from harmony-tokens.py
    --server-url       SGLang server URL (default: http://localhost:30000)
    --max-samples      Maximum number of samples to process (default: all)
    --max-tokens       Maximum tokens to generate per request (default: 100)
    --max-concurrency  Maximum number of concurrent requests (default: 256)
    --output           Output pickle file for responses (optional)
    --pass-k           Number of inference passes per sample for pass@k strategy (default: 1)
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
    def __init__(self,
                 server_url: str = "http://localhost:30000",
                 temperature: float = 0.001,
                 top_k: int = 1,
                 top_p: float = 1.0,
                 timeout: int = 1200
                 ):
        self.base_url = server_url
        self.session = requests.Session()
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.timeout = timeout

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
                "top_p": self.top_p,
            }
        }

        try:
            response = self.session.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=self.timeout,
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
    input_ids, max_tokens, server_url, sample_id, pass_num, temperature, top_k, top_p, timeout = args_tuple

    # Create a new client for this process
    client = SGLangClient(
        server_url=server_url,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        timeout=timeout)

    try:
        # Track latency: time from request sent to response received
        start_time = time.time()
        response = client.send_request(input_ids, max_tokens=max_tokens)
        end_time = time.time()
        latency = end_time - start_time
        return sample_id, pass_num, response, latency
    except Exception as e:
        logger.error(f"Request {sample_id} (pass {pass_num}) failed: {e}")
        # Return None for latency on error
        return sample_id, pass_num, {"error": str(e)}, None


def send_requests_parallel(tokenized_df: pd.DataFrame, server_url: str,
                           max_tokens: int = 100, max_concurrency: int = 128, temperature: float = 0.001, top_k: int = 1, top_p: float = 1.0, timeout: int = 1200,
                           pass_k: int = 1):
    """Send all requests to SGLang server in parallel using multiprocessing.

    Args:
        pass_k: Number of inference passes per sample for pass@k strategy

    Returns:
        tuple: (responses_by_pass, latencies_by_pass) - Dict mapping (sample_id, pass_num) to response/latency
    """
    num_samples = len(tokenized_df)
    total_requests = num_samples * pass_k
    logger.info(
        f"Sending {total_requests} requests ({num_samples} samples × {pass_k} passes) to server with {max_concurrency} concurrent workers...")

    # Prepare arguments for multiprocessing - create pass_k requests per sample
    args_list = []
    for idx, row in tokenized_df.iterrows():
        for pass_num in range(pass_k):
            args_list.append((
                row['tok_input'], max_tokens, server_url,
                idx, pass_num, temperature, top_k, top_p, timeout
            ))

    start_time = time.time()

    with Pool(processes=min(max_concurrency, total_requests)) as pool:
        results = list(tqdm(
            pool.imap_unordered(send_single_request, args_list),
            total=len(args_list),
            desc="Sending requests",
            unit="request"
        ))

    # Group results by sample_id and pass_num
    responses_by_pass = {}
    latencies_by_pass = {}
    for sample_id, pass_num, response, latency in results:
        responses_by_pass[(sample_id, pass_num)] = response
        latencies_by_pass[(sample_id, pass_num)] = latency

    total_time = time.time() - start_time
    logger.info(
        f"Completed {total_requests} requests in {total_time:.2f} seconds")
    logger.info(f"Average rate: {total_requests/total_time:.2f} requests/sec")

    # Log latency statistics
    valid_latencies = [
        lat for lat in latencies_by_pass.values() if lat is not None]
    if valid_latencies:
        avg_latency = sum(valid_latencies) / len(valid_latencies)
        min_latency = min(valid_latencies)
        max_latency = max(valid_latencies)
        logger.info(
            f"Latency stats - Avg: {avg_latency:.3f}s, Min: {min_latency:.3f}s, Max: {max_latency:.3f}s")

    return responses_by_pass, latencies_by_pass


def extract_response_ids(
        responses_by_pass: Dict[tuple, Dict[str, Any]], tokenized_df: pd.DataFrame, pass_k: int) -> Dict[tuple, List[int]]:
    """Extract response output_ids from SGLang responses for all passes.

    Args:
        responses_by_pass: Dict mapping (sample_id, pass_num) to response
        tokenized_df: DataFrame with samples
        pass_k: Number of passes per sample

    Returns:
        Dict mapping (sample_id, pass_num) to output_ids list
    """
    logger.info("Extracting response output_ids...")

    response_ids_by_pass = {}
    total_responses = len(tokenized_df) * pass_k

    with tqdm(total=total_responses, desc="Extracting responses", unit="response") as pbar:
        for idx, row in tokenized_df.iterrows():
            for pass_num in range(pass_k):
                response = responses_by_pass.get((idx, pass_num), {})
                response_id = []
                if "error" not in response and "output_ids" in response:
                    try:
                        # SGLang returns the generated token IDs in the
                        # 'output_ids' field
                        response_id = response["output_ids"]
                    except Exception as e:
                        logger.warning(
                            f"Failed to extract response for sample {idx}, pass {pass_num}: {e}")
                response_ids_by_pass[(idx, pass_num)] = response_id
                pbar.update(1)

    logger.info("Response output_ids extraction complete")
    return response_ids_by_pass


def detokenize_output_ids(
        response_ids_by_pass: Dict[tuple, List[int]], pass_k: int) -> Dict[tuple, str]:
    """Detokenize output_ids back to text using AutoTokenizer for all passes.

    Args:
        response_ids_by_pass: Dict mapping (sample_id, pass_num) to output_ids
        pass_k: Number of passes per sample

    Returns:
        Dict mapping (sample_id, pass_num) to detokenized text
    """
    logger.info("Detokenizing output_ids to text...")

    tokenizer = get_tokenizer()
    detokenized_texts_by_pass = {}

    for (sample_id, pass_num), token_ids in tqdm(
            response_ids_by_pass.items(), desc="Detokenizing outputs", unit="output"):
        try:
            # Detokenize the token IDs back to text
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            detokenized_texts_by_pass[(sample_id, pass_num)] = text
        except Exception as e:
            logger.warning(
                f"Failed to detokenize output for sample {sample_id}, pass {pass_num}: {e}")
            detokenized_texts_by_pass[(sample_id, pass_num)] = ""

    logger.info("Output detokenization complete")
    return detokenized_texts_by_pass


def save_responses(responses_by_pass: Dict[tuple, Dict[str, Any]],
                   response_ids_by_pass: Dict[tuple, List[int]],
                   detokenized_texts_by_pass: Dict[tuple, str],
                   latencies_by_pass: Dict[tuple, float],
                   tokenized_df: pd.DataFrame, pass_k: int, output_file: str = None) -> pd.DataFrame:
    """Save all responses to DataFrame and optionally to pickle file.

    Args:
        responses_by_pass: Dict mapping (sample_id, pass_num) to response
        response_ids_by_pass: Dict mapping (sample_id, pass_num) to output_ids
        detokenized_texts_by_pass: Dict mapping (sample_id, pass_num) to text
        latencies_by_pass: Dict mapping (sample_id, pass_num) to latency
        tokenized_df: Original DataFrame with samples
        pass_k: Number of passes per sample
        output_file: Optional output pickle file

    Returns:
        DataFrame with columns for each pass (e.g., model_output_0, model_output_1, ...)
    """
    logger.info("Processing responses and updating DataFrame...")

    # Work with the original DataFrame
    result_df = tokenized_df.copy()

    # Create columns for each pass with _0, _1, _2, ... suffixes
    for pass_num in range(pass_k):
        # Lists to store data for this pass
        model_outputs = []
        tok_model_outputs = []
        tok_model_output_lens = []
        infer_times = []

        for idx in tokenized_df.index:
            key = (idx, pass_num)
            detokenized_text = detokenized_texts_by_pass.get(key, "")
            response_ids = response_ids_by_pass.get(key, [])
            latency = latencies_by_pass.get(key, None)

            model_outputs.append(detokenized_text)
            tok_model_outputs.append(response_ids)
            tok_model_output_lens.append(len(response_ids))
            infer_times.append(latency)

        # Add columns with suffixes
        result_df[f'model_output_{pass_num}'] = model_outputs
        result_df[f'tok_model_output_{pass_num}'] = tok_model_outputs
        result_df[f'tok_model_output_len_{pass_num}'] = tok_model_output_lens
        result_df[f'infer_time_{pass_num}'] = infer_times

    # Calculate output token lengths for logging
    all_output_token_lengths = []
    for idx in tokenized_df.index:
        for pass_num in range(pass_k):
            key = (idx, pass_num)
            response = responses_by_pass.get(key, {})
            response_ids = response_ids_by_pass.get(key, [])
            try:
                output_token_length = response.get(
                    "meta_info", {}).get(
                    "completion_tokens", len(response_ids))
                all_output_token_lengths.append(output_token_length)
            except Exception as e:
                logger.warning(
                    f"Failed to calculate output tokens for sample {idx}, pass {pass_num}: {e}")
                all_output_token_lengths.append(len(response_ids))

    logger.info(f"Updated DataFrame with shape: {result_df.shape}")
    new_columns = [
        f'model_output_{i}, tok_model_output_{i}, tok_model_output_len_{i}, infer_time_{i}' for i in range(pass_k)]
    logger.info(f"Added columns for {pass_k} passes: {', '.join(new_columns)}")
    if all_output_token_lengths:
        logger.info(
            f"Average output token length: {sum(all_output_token_lengths)/len(all_output_token_lengths):.1f}")

    # Save to pickle file if output_file is provided
    if output_file:
        logger.info(f"Saving responses to {output_file}...")
        result_df.to_pickle(output_file)
        logger.info(f"Responses saved to {output_file}")

    return result_df


def process_requests(tokenized_df: pd.DataFrame, server_url: str,
                     max_samples: int = None, max_tokens: int = 100,
                     max_concurrency: int = 128, output_file: str = None, temperature: float = 0.001, top_k: int = 1, top_p: float = 1.0,
                     timeout: int = 1200, pass_k: int = 1) -> pd.DataFrame:
    """Main processing function that handles requests and response extraction.

    Args:
        pass_k: Number of inference passes per sample for pass@k strategy
    """

    # Step 1: Limit samples if specified
    if max_samples is not None:
        tokenized_df = tokenized_df.head(max_samples)
        logger.info(f"Limited to first {max_samples} samples")

    # Step 2: Send all requests in parallel (k passes per sample)
    responses_by_pass, latencies_by_pass = send_requests_parallel(
        tokenized_df,
        server_url,
        max_tokens,
        max_concurrency,
        temperature,
        top_k,
        top_p,
        timeout,
        pass_k)

    # Step 3: Extract response output_ids for all passes
    response_ids_by_pass = extract_response_ids(
        responses_by_pass, tokenized_df, pass_k)

    # Step 4: Detokenize output_ids to text for model_output for all passes
    detokenized_texts_by_pass = detokenize_output_ids(
        response_ids_by_pass, pass_k)

    # Step 5: Save all results and return DataFrame
    result_df = save_responses(
        responses_by_pass,
        response_ids_by_pass,
        detokenized_texts_by_pass,
        latencies_by_pass,
        tokenized_df,
        pass_k,
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
                        help="Maximum number of concurrent requests (default: 256)")
    parser.add_argument("--output", default=None,
                        help="Output pickle file for responses (optional)")
    parser.add_argument("--pass-k", type=int, default=1,
                        help="Number of inference passes per sample for pass@k strategy (default: 1)")
    parser.add_argument("--temperature", type=float, default=0.001,
                        help="Temperature for sampling (default: 0.001)")
    parser.add_argument("--top-k", type=int, default=1,
                        help="Top-k for sampling (default: 1)")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Top-p for sampling (default: 1.0)")
    parser.add_argument("--timeout", type=int, default=1200,
                        help="Timeout for requests (default: 1200)")

    args = parser.parse_args()

    # Test connection
    logger.info(f"Testing server connection to {args.server_url}...")
    test_client = SGLangClient(
        server_url=args.server_url,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        timeout=args.timeout)

    test_response = test_client.send_request(input_ids=[1, 2, 3], max_tokens=5)
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
                                 top_k=args.top_k,
                                 top_p=args.top_p,
                                 timeout=args.timeout,
                                 pass_k=args.pass_k)

    # Print summary
    logger.info(f"\nProcessing completed:")
    logger.info(f"  - Total samples processed: {len(result_df)}")
    logger.info(f"  - Number of passes per sample: {args.pass_k}")
    logger.info(
        f"  - Average input token length: {result_df['tok_input_len'].mean():.1f}")

    # Calculate average output length across all passes
    if args.pass_k == 1:
        avg_output_len = result_df['tok_model_output_len_0'].mean()
        logger.info(f"  - Average output token length: {avg_output_len:.1f}")
    else:
        all_output_lens = []
        for i in range(args.pass_k):
            all_output_lens.extend(
                result_df[f'tok_model_output_len_{i}'].tolist())
        avg_output_len = sum(all_output_lens) / \
            len(all_output_lens) if all_output_lens else 0
        logger.info(
            f"  - Average output token length (across all passes): {avg_output_len:.1f}")

    if args.output:
        logger.info(f"  - Results saved to: {args.output}")
    else:
        logger.info("  - Results returned as DataFrame (not saved to file)")


if __name__ == "__main__":
    main()
