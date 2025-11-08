#!/usr/bin/env python3
"""
Script to send text prompts to TensorRT-LLM server via OpenAI completions endpoint.
Supports round-robin load balancing across multiple server endpoints.

Usage:
    python run_infer_trtllm.py --input-tokens tokenized_data.pkl [options]

Arguments:
    --input-tokens     Path to pickle file containing data with text_input column from harmony-tokens.py
    --server-url       TensorRT-LLM server URL(s) - comma-separated for round-robin (e.g., "localhost:8000,localhost:8001")
    --max-samples      Maximum number of samples to process (default: all)
    --max-tokens       Maximum tokens to generate per request (default: 100)
    --max-concurrency  Maximum number of concurrent requests (default: 256)
    --output           Output pickle file for responses (optional)
    --pass-k           Number of inference passes per sample for pass@k strategy (default: 1)

Examples:
    # Single server
    python run_infer_trtllm.py --input-tokens data.pkl --server-url localhost:8000
    
    # Multiple servers with round-robin
    python run_infer_trtllm.py --input-tokens data.pkl --server-url localhost:8000,localhost:8001,localhost:8002
"""

import asyncio
import argparse
import time
import logging
from typing import List, Dict, Any, Tuple
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
import httpx
from openai import AsyncOpenAI

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress verbose HTTP logs from httpx and openai
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

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


class TRTLLMClient:
    """Client for TensorRT-LLM server using OpenAI-compatible endpoint with round-robin support."""
    
    def __init__(self,
                 server_urls: List[str] = None,
                 temperature: float = 0.001,
                 top_k: int = 1,
                 top_p: float = 1.0,
                 max_concurrency: int = 256,
                 timeout: int = 1200):
        # Support multiple server URLs for round-robin load balancing
        if server_urls is None:
            server_urls = ["localhost:8000"]
        self.server_urls = server_urls
        self.num_servers = len(server_urls)
        self.current_server_index = 0
        
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_concurrency = max_concurrency
        self.timeout = timeout
        self.model_name = MODEL_NAME
        
        # Initialize async OpenAI clients (one per server)
        self.http_clients = []
        self.async_clients = []
        self.concurrency_semaphore = None
        
        logger.info(f"Initialized client with {self.num_servers} server(s): {', '.join(self.server_urls)}")
        
    async def initialize(self):
        """Initialize OpenAI clients for all servers."""
        # Create semaphore for concurrency control
        self.concurrency_semaphore = asyncio.Semaphore(self.max_concurrency)
        
        # Create HTTP and OpenAI clients for each server
        for server_url in self.server_urls:
            # Setup HTTP client with proper connection limits for high concurrency
            http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(
                    max_keepalive_connections=self.max_concurrency * 2,
                    max_connections=self.max_concurrency * 2,
                ),
                http2=True
            )
            
            # Setup OpenAI client with the configured HTTP client
            async_client = AsyncOpenAI(
                api_key='dummy',  # TensorRT-LLM server doesn't require real API key
                base_url=f"http://{server_url}/v1/",
                timeout=self.timeout,
                max_retries=10,
                http_client=http_client,
            )
            
            self.http_clients.append(http_client)
            self.async_clients.append(async_client)
        
        logger.info(f"Initialized {len(self.async_clients)} OpenAI client(s)")
        
    def _get_next_client(self) -> AsyncOpenAI:
        """Get the next client using round-robin selection."""
        client = self.async_clients[self.current_server_index]
        self.current_server_index = (self.current_server_index + 1) % self.num_servers
        return client
    
    async def send_request(
            self, prompt: str, max_tokens: int = 100,
            sample_id: int = 0, pass_num: int = 0) -> Tuple[int, int, Dict[str, Any], float]:
        """Send a single request to the TensorRT-LLM server using round-robin.
        
        Args:
            prompt: Text prompt to send
            max_tokens: Maximum tokens to generate
            sample_id: Sample identifier
            pass_num: Pass number for pass@k strategy
            
        Returns:
            Tuple of (sample_id, pass_num, response, latency)
        """
        # Prepare generation parameters using OpenAI completions format (as per TensorRT-LLM docs)
        gen_params = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": False,
            "extra_body": {
                # TensorRT-LLM specific parameters
                "min_tokens": 1,
                "top_k": self.top_k,
            },
        }
        
        try:
            # Track latency: time from request sent to response received
            start_time = time.time()
            
            # Select client using round-robin
            client = self._get_next_client()
            
            # Use semaphore for concurrency control
            async with self.concurrency_semaphore:
                completion = await client.completions.create(**gen_params)
            
            end_time = time.time()
            latency = end_time - start_time
            
            # Extract response text from completions format
            response_text = completion.choices[0].text
            
            # Tokenize the response to get output_ids (similar to SGLang format)
            tokenizer = get_tokenizer()
            output_ids = tokenizer.encode(response_text, add_special_tokens=False)
            
            # Format response similar to SGLang format for compatibility
            response = {
                "output_ids": output_ids,
                "text": response_text,
                "meta_info": {
                    "completion_tokens": len(output_ids),
                }
            }
            
            return sample_id, pass_num, response, latency
            
        except Exception as e:
            logger.error(f"Request {sample_id} (pass {pass_num}) failed: {e}")
            return sample_id, pass_num, {"error": str(e)}, None
    
    async def shutdown(self):
        """Clean up resources for all clients."""
        for http_client in self.http_clients:
            if http_client:
                await http_client.aclose()


def load_tokenized_data(data_file: str) -> pd.DataFrame:
    """Load data from pickle file produced by harmony-tokens.py."""
    logger.info(f"Loading data from {data_file}")
    
    # Load DataFrame from pickle
    df = pd.read_pickle(data_file)
    logger.info(f"Loaded DataFrame with shape: {df.shape}")
    
    # Check if text_input column exists and has valid data
    if 'text_input' in df.columns:
        # Check for any None values in text_input
        failed_mask = df['text_input'].isna()
        failed_count = failed_mask.sum()
        
        if failed_count > 0:
            failed_indices = df[failed_mask].index.unique()
            error_msg = f"Found {failed_count} samples with missing text_input at indices: {failed_indices.tolist()}"
            logger.error(error_msg)
            raise AssertionError(error_msg)
        
        # Check first sample
        first_text = df.iloc[0]['text_input']
        if isinstance(first_text, str):
            logger.info(f"First sample text length: {len(first_text)} characters")
        else:
            logger.warning("text_input column exists but first sample is not a string")
        
        logger.info(f"All {len(df)} samples have valid text_input")
    else:
        logger.error("No 'text_input' column found in DataFrame")
        raise ValueError("DataFrame must contain 'text_input' column")
    
    return df


async def send_requests_async(
        tokenized_df: pd.DataFrame, server_urls: List[str],
        max_tokens: int = 100, max_concurrency: int = 256,
        temperature: float = 0.001, top_k: int = 1, top_p: float = 1.0,
        timeout: int = 1200, pass_k: int = 1):
    """Send all requests to TensorRT-LLM server(s) asynchronously with round-robin load balancing.
    
    Args:
        server_urls: List of server URLs for round-robin load balancing
        pass_k: Number of inference passes per sample for pass@k strategy
        
    Returns:
        tuple: (responses_by_pass, latencies_by_pass) - Dict mapping (sample_id, pass_num) to response/latency
    """
    num_samples = len(tokenized_df)
    total_requests = num_samples * pass_k
    logger.info(
        f"Sending {total_requests} requests ({num_samples} samples × {pass_k} passes) with {max_concurrency} concurrent workers...")
    
    # Initialize client with multiple servers for round-robin
    client = TRTLLMClient(
        server_urls=server_urls,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_concurrency=max_concurrency,
        timeout=timeout
    )
    await client.initialize()
    
    # Prepare all tasks - create pass_k requests per sample
    tasks = []
    for idx, row in tokenized_df.iterrows():
        for pass_num in range(pass_k):
            task = client.send_request(
                row['text_input'],
                max_tokens=max_tokens,
                sample_id=idx,
                pass_num=pass_num
            )
            tasks.append(task)
    
    start_time = time.time()
    
    # Execute all tasks concurrently with progress bar
    results = []
    for coro in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Sending requests",
            unit="request"):
        result = await coro
        results.append(result)
    
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
    valid_latencies = [lat for lat in latencies_by_pass.values() if lat is not None]
    if valid_latencies:
        avg_latency = sum(valid_latencies) / len(valid_latencies)
        min_latency = min(valid_latencies)
        max_latency = max(valid_latencies)
        logger.info(
            f"Latency stats - Avg: {avg_latency:.3f}s, Min: {min_latency:.3f}s, Max: {max_latency:.3f}s")
    
    # Shutdown client
    await client.shutdown()
    
    return responses_by_pass, latencies_by_pass


def extract_response_ids(
        responses_by_pass: Dict[tuple, Dict[str, Any]], tokenized_df: pd.DataFrame, pass_k: int) -> Dict[tuple, List[int]]:
    """Extract response output_ids from TensorRT-LLM responses for all passes.
    
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
                        # TensorRT-LLM returns the generated token IDs in the 'output_ids' field
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


async def process_requests_async(tokenized_df: pd.DataFrame, server_urls: List[str],
                                 max_samples: int = None, max_tokens: int = 100,
                                 max_concurrency: int = 256, output_file: str = None,
                                 temperature: float = 0.001, top_k: int = 1, top_p: float = 1.0,
                                 timeout: int = 1200, pass_k: int = 1) -> pd.DataFrame:
    """Main processing function that handles requests and response extraction.
    
    Args:
        server_urls: List of server URLs for round-robin load balancing
        pass_k: Number of inference passes per sample for pass@k strategy
    """
    
    # Step 1: Limit samples if specified
    if max_samples is not None:
        tokenized_df = tokenized_df.head(max_samples)
        logger.info(f"Limited to first {max_samples} samples")
    
    # Step 2: Send all requests asynchronously (k passes per sample)
    responses_by_pass, latencies_by_pass = await send_requests_async(
        tokenized_df,
        server_urls,
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
        description="Send text prompts to TensorRT-LLM server via OpenAI completions endpoint")
    parser.add_argument("--input-tokens", required=True,
                        help="Path to pickle file containing data with text_input column from harmony-tokens.py")
    parser.add_argument("--server-url", default="localhost:8000",
                        help="TensorRT-LLM server URL(s) - comma-separated for round-robin load balancing (default: localhost:8000)")
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
    
    # Parse comma-separated server URLs
    server_urls = [url.strip() for url in args.server_url.split(',')]
    logger.info(f"Configured {len(server_urls)} server(s) for round-robin load balancing")
    
    # Test connection
    async def test_connection():
        logger.info(f"Testing server connection(s)...")
        client = TRTLLMClient(
            server_urls=server_urls,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_concurrency=1,
            timeout=args.timeout
        )
        await client.initialize()
        
        try:
            _, _, test_response, _ = await client.send_request(
                prompt="Test", max_tokens=5, sample_id=0, pass_num=0)
            if "error" in test_response:
                logger.error(f"Server connection failed: {test_response['error']}")
                logger.error("Make sure your TensorRT-LLM server(s) are running with OpenAI endpoint enabled.")
                return False
            logger.info("Server connection successful")
            return True
        finally:
            await client.shutdown()
    
    # Run connection test
    if not asyncio.run(test_connection()):
        return
    
    # Load pre-tokenized data
    tokenized_df = load_tokenized_data(args.input_tokens)
    
    # Process requests and get result DataFrame
    result_df = asyncio.run(process_requests_async(
        tokenized_df, server_urls,
        max_samples=args.max_samples,
        max_tokens=args.max_tokens,
        max_concurrency=args.max_concurrency,
        output_file=args.output,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        timeout=args.timeout,
        pass_k=args.pass_k))
    
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



