# Copyright 2025 The MLPerf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================


#!/usr/bin/env python3
"""
Oracle Single Shot - Generate LLM answers using ground truth Wikipedia articles.

This script reads queries from frames_dataset.tsv, loads the corresponding
Wikipedia articles from wiki_articles folder, and generates answers using
an LLM service with the oracle documents as context.

Features:
- Batch processing of LLM requests (default batch size: 16)
- Checkpointing: saves progress after each batch to pickle file
- Resume capability: skips already processed queries
- Retry failed requests when all queries are processed
- Handles missing documents gracefully
"""

import argparse
import ast
import json
import os
import pickle
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import requests


DEFAULT_CHECKPOINT_FILE = "oracle_checkpoint.pkl"
DEFAULT_SERVICE_URL = "http://localhost:8123/v1/chat/completions"
DEFAULT_MODEL_NAME = "gpt-oss-120b-mxfp4"
DEFAULT_BATCH_SIZE = 1
DEFAULT_TIMEOUT = 2400
# For reasoning model, it should be large enough
DEFAULT_MAX_TOKENS = 10*1024
MAX_TOTAL_CHARS = 400000  # total char budget split across all docs per query (~100K tokens @ 4 chars/token)

# Global cache for URL to filename mapping
_url_to_file_cache: Optional[Dict[str, Path]] = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Oracle Single Shot: Generate LLM answers using ground truth documents",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/frames_dataset.tsv",
        help="Path to frames_dataset.tsv (default: data/frames_dataset.tsv)"
    )
    parser.add_argument(
        "--wiki-articles-dir",
        type=str,
        default="wiki_articles",
        help="Path to wiki_articles directory (default: wiki_articles)"
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default=DEFAULT_CHECKPOINT_FILE,
        help=f"Path to checkpoint pickle file (default: {DEFAULT_CHECKPOINT_FILE})"
    )
    parser.add_argument(
        "--service-url",
        type=str,
        default=DEFAULT_SERVICE_URL,
        help=f"LLM service URL (default: {DEFAULT_SERVICE_URL})"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Model name (default: {DEFAULT_MODEL_NAME})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for LLM requests (default: {DEFAULT_BATCH_SIZE})"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum tokens for LLM response (default: {DEFAULT_MAX_TOKENS})"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout in seconds for LLM generation requests (default: {DEFAULT_TIMEOUT})"
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=False,
        help="Enable thinking/reasoning in the model via chat_template_kwargs (default: False)"
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=None,
        choices=["low", "medium", "high"],
        help="Reasoning effort level to pass to the model (default: not set)"
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry failed requests (automatically enabled when all queries processed)"
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Maximum number of queries to process (default: all)"
    )
    return parser.parse_args()


def build_url_to_file_cache(wiki_dir: Path) -> Dict[str, Path]:
    """
    Build a cache mapping Wikipedia URLs to their corresponding .txt files.
    Uses JSON metadata files to get accurate URL mapping.
    """
    cache = {}
    
    print(f"Building URL cache from {wiki_dir}...")
    json_files = list(wiki_dir.glob("*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
                # Get URL without fragment
                url = data.get('url', '')
                source_url = data.get('source_url', '')
                
                # Remove fragment from source_url if present
                if '#' in source_url:
                    source_url = source_url.split('#')[0]
                
                # Get corresponding .txt file
                txt_file = json_file.with_suffix('.txt')
                if txt_file.exists():
                    # Map both url and source_url (without fragment) to the file
                    if url:
                        cache[url] = txt_file
                    if source_url and source_url != url:
                        cache[source_url] = txt_file
        except Exception as e:
            # Skip files with errors
            continue
    
    print(f"Cached {len(cache)} URL mappings from {len(json_files)} JSON files")
    return cache


def find_wiki_article(url: str, wiki_dir: Path) -> Optional[Path]:
    """
    Find the .txt file in wiki_articles that matches the Wikipedia URL.
    Uses JSON metadata for accurate matching.
    """
    global _url_to_file_cache
    
    if not url or "wikipedia.org/wiki/" not in url:
        return None
    
    # Build cache on first call
    if _url_to_file_cache is None:
        _url_to_file_cache = build_url_to_file_cache(wiki_dir)
    
    # Remove fragment from URL if present
    url_no_fragment = url.split('#')[0]
    
    # Look up in cache
    if url in _url_to_file_cache:
        return _url_to_file_cache[url]
    elif url_no_fragment in _url_to_file_cache:
        return _url_to_file_cache[url_no_fragment]
    
    return None


def load_wiki_articles(wiki_urls: List[str], wiki_dir: Path, max_chars: int = MAX_TOTAL_CHARS) -> Tuple[List[str], List[str], List[int]]:
    """
    Load Wikipedia articles from wiki_articles folder.
    max_chars is the TOTAL character budget shared across all docs for this query.

    Returns:
        Tuple of (documents, file_paths, doc_lengths)
    """
    documents = []
    file_paths = []
    doc_lengths = []

    valid_urls = [u for u in wiki_urls if u]
    n = max(1, len(valid_urls))
    per_doc_limit = max(500, max_chars // n)  # distribute budget evenly
    for url in wiki_urls:
        # Find the actual file using URL
        file_path = find_wiki_article(url, wiki_dir)

        if file_path and file_path.exists():
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                truncated = content[:per_doc_limit]
                documents.append(truncated)
                file_paths.append(str(file_path))
                doc_lengths.append(len(truncated))
            except Exception as e:
                print(f"Warning: Failed to read {file_path}: {e}")
                documents.append("")
                file_paths.append("")
                doc_lengths.append(0)
        else:
            documents.append("")
            file_paths.append("")
            doc_lengths.append(0)
    
    return documents, file_paths, doc_lengths


def generate_llm_answer(query: str, documents: List[str], urls: List[str], llm_config: Dict) -> str:
    """
    Generate LLM answer using the provided documents as context.
    
    This function is adapted from single_shot_retrieval.py _generate_llm_answer
    """
    context_parts = []
    for idx, (doc, url) in enumerate(zip(documents, urls), 1):
        if doc.strip():
            source = url or "Unknown source"
            snippet = doc.strip()
            context_parts.append(f"[{idx}] Source: {source}\n{snippet}")
    
    evidence_block = "\n\n".join(context_parts) if context_parts else "No supporting documents were retrieved."
    
    user_prompt = (
        "Answer the question using only the provided evidence."
        " Respond with a single word or short phrase, or 'Unknown' if the evidence is insufficient.\n\n"
        f"Question:\n{query}\n\nEvidence:\n{evidence_block}"
    )
    
    payload = {
        "model": llm_config["model_name"],
        "messages": [
            {
                "role": "system",
                "content": "You are a concise retrieval QA assistant who trusts the supplied context."
            },
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "max_tokens": llm_config["max_tokens"],
    }

    if llm_config.get("enable_thinking"):
        payload["chat_template_kwargs"] = {"enable_thinking": True}

    if llm_config.get("reasoning_effort"):
        payload["reasoning_effort"] = llm_config["reasoning_effort"]

    response = requests.post(llm_config["service_url"], json=payload, timeout=llm_config["timeout"])
    response.raise_for_status()
    data = response.json()
    #from pprint import pprint
    #pprint(data, indent=4)
    return data["choices"][0]["message"]["content"].strip()


def load_checkpoint(checkpoint_file: Path) -> pd.DataFrame:
    """Load checkpoint from pickle file."""
    if checkpoint_file.exists():
        with open(checkpoint_file, "rb") as f:
            df = pickle.load(f)
            # Ensure it's a DataFrame
            if isinstance(df, dict):
                # Handle old format for backwards compatibility
                df = pd.DataFrame(df.get("results", []))
            return df
    return pd.DataFrame()


def save_checkpoint(checkpoint_file: Path, df: pd.DataFrame):
    """Save checkpoint DataFrame to pickle file."""
    with open(checkpoint_file, "wb") as f:
        pickle.dump(df, f)


def parse_wiki_links(wiki_links_str: str) -> List[str]:
    """Parse the wiki_links column which is stored as a string representation of a list."""
    try:
        # Use ast.literal_eval to safely parse the string as a Python literal
        return ast.literal_eval(wiki_links_str)
    except:
        return []


def process_single(
    idx: int,
    query: str,
    ground_truth: str,
    wiki_urls: List[str],
    wiki_dir: Path,
    llm_config: Dict
) -> Dict:
    """Process a single query and return the result dict."""
    result = {
        "index": idx,
        "query": query,
        "ground_truth": ground_truth,
        "wiki_urls": str(wiki_urls),
        "wiki_file_paths": "",
        "doc_lengths": "",
        "total_doc_length": 0,
        "num_docs": len(wiki_urls),
        "num_missing_docs": 0,
        "llm_answer": "",
        "success": False,
        "failure_reason": ""
    }

    try:
        documents, file_paths, doc_lengths = load_wiki_articles(wiki_urls, wiki_dir)
        result["wiki_file_paths"] = str(file_paths)
        result["doc_lengths"] = str(doc_lengths)
        result["total_doc_length"] = sum(doc_lengths)

        valid_docs = [d for d in documents if d.strip()]
        missing_count = len(wiki_urls) - len(valid_docs)
        result["num_missing_docs"] = missing_count

        if missing_count > 0:
            print(f"  Query {idx}: {missing_count}/{len(wiki_urls)} documents missing")

        llm_answer = generate_llm_answer(query, documents, wiki_urls, llm_config)
        result["llm_answer"] = llm_answer
        result["success"] = True

    except Exception as e:
        result["failure_reason"] = str(e)
        print(f"  Query {idx}: Failed - {e}")

    return result


def process_batch(
    batch_data: List[Tuple[int, str, str, List[str]]],
    wiki_dir: Path,
    llm_config: Dict
) -> List[Dict]:
    """
    Process a batch of queries in parallel.

    Args:
        batch_data: List of (index, query, answer, wiki_urls)
        wiki_dir: Path to wiki_articles directory
        llm_config: LLM configuration dict

    Returns:
        List of result dictionaries sorted by original order
    """
    futures_map = {}
    with ThreadPoolExecutor(max_workers=len(batch_data)) as executor:
        for idx, query, ground_truth, wiki_urls in batch_data:
            future = executor.submit(process_single, idx, query, ground_truth, wiki_urls, wiki_dir, llm_config)
            futures_map[future] = idx

        results_map = {}
        for future in as_completed(futures_map):
            result = future.result()
            results_map[result["index"]] = result

    # Return in original batch order
    return [results_map[idx] for idx, _, _, _ in batch_data]


def main():
    args = parse_args()
    
    # Setup paths
    dataset_path = Path(args.dataset)
    wiki_dir = Path(args.wiki_articles_dir)
    checkpoint_file = Path(args.checkpoint_file)
    
    # Validate paths
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not wiki_dir.exists():
        raise FileNotFoundError(f"Wiki articles directory not found: {wiki_dir}")

    # Pre-build URL cache once before threads start
    global _url_to_file_cache
    _url_to_file_cache = build_url_to_file_cache(wiki_dir)
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path, sep="\t")
    
    # Apply max_queries limit if specified
    if args.max_queries:
        df = df.head(args.max_queries)
    
    print(f"Total queries in dataset: {len(df)}")
    
    # Load checkpoint
    checkpoint_df = load_checkpoint(checkpoint_file)
    processed_indices = set(checkpoint_df["index"].tolist()) if not checkpoint_df.empty else set()
    
    print(f"Already processed: {len(processed_indices)} queries")
    
    # LLM configuration
    llm_config = {
        "service_url": args.service_url,
        "model_name": args.model_name,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
        "enable_thinking": args.enable_thinking,
        "reasoning_effort": args.reasoning_effort,
    }
    
    # Determine which queries to process
    if args.retry_failed:
        # Retry only failed queries
        if not checkpoint_df.empty:
            failed_df = checkpoint_df[checkpoint_df["success"] == False]
            queries_to_process = [(int(row["index"]), df.iloc[int(row["index"])]["Prompt"], 
                                  df.iloc[int(row["index"])]["Answer"],
                                  parse_wiki_links(df.iloc[int(row["index"])]["wiki_links"]))
                                 for _, row in failed_df.iterrows()]
            print(f"Retrying {len(queries_to_process)} failed queries...")
        else:
            queries_to_process = []
    else:
        # Process unprocessed queries
        queries_to_process = []
        for idx, row in df.iterrows():
            if idx not in processed_indices:
                wiki_urls = parse_wiki_links(row["wiki_links"])
                queries_to_process.append((idx, row["Prompt"], row["Answer"], wiki_urls))
        
        print(f"Processing {len(queries_to_process)} new queries...")
    
    if not queries_to_process:
        # If no new queries and all done, check for failed ones
        if not args.retry_failed:
            failed_count = len(checkpoint_df[checkpoint_df["success"] == False]) if not checkpoint_df.empty else 0
            if failed_count > 0:
                print(f"\nAll new queries processed. {failed_count} queries failed.")
                print("Run with --retry-failed to retry failed queries.")
            else:
                print("\nAll queries successfully processed!")
        else:
            print("No failed queries to retry!")
        return
    
    # Process in batches
    batch_size = args.batch_size
    total_batches = (len(queries_to_process) + batch_size - 1) // batch_size
    
    print(f"Batch size: {batch_size}")
    print(f"Total batches: {total_batches}")
    print(f"Service URL: {llm_config['service_url']}")
    print(f"Model: {llm_config['model_name']}\n")
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(queries_to_process))
        batch = queries_to_process[start_idx:end_idx]
        
        print(f"Processing batch {batch_idx + 1}/{total_batches} "
              f"(queries {start_idx + 1}-{end_idx})...")
        
        # Process batch
        batch_results = process_batch(batch, wiki_dir, llm_config)
        
        # Convert batch results to DataFrame
        batch_df = pd.DataFrame(batch_results)
        
        # Update checkpoint
        if args.retry_failed:
            # For retry, update existing results
            # Remove old entries for these indices
            indices_to_update = batch_df["index"].tolist()
            checkpoint_df = checkpoint_df[~checkpoint_df["index"].isin(indices_to_update)]
            # Append new results
            checkpoint_df = pd.concat([checkpoint_df, batch_df], ignore_index=True)
        else:
            # For new queries, append results
            checkpoint_df = pd.concat([checkpoint_df, batch_df], ignore_index=True)
        
        # Sort by index for consistency
        checkpoint_df = checkpoint_df.sort_values("index").reset_index(drop=True)
        
        # Save checkpoint after each batch
        save_checkpoint(checkpoint_file, checkpoint_df)
        print(f"  Checkpoint saved to {checkpoint_file}")
        
        # Show batch statistics
        success_count = batch_df["success"].sum()
        print(f"  Batch success rate: {success_count}/{len(batch_results)}\n")
    
    # Final statistics
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)
    
    total_processed = len(checkpoint_df)
    total_success = checkpoint_df["success"].sum()
    total_failed = total_processed - total_success
    
    print(f"Total queries processed: {total_processed}")
    print(f"Successful: {total_success}")
    print(f"Failed: {total_failed}")
    
    if total_failed > 0:
        print(f"\nRun with --retry-failed to retry {total_failed} failed queries.")
    
    print(f"\nResults saved to: {checkpoint_file}")


if __name__ == "__main__":
    main()
