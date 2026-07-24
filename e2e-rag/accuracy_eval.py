#!/usr/bin/env python3
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

"""
Accuracy evaluation script for RAG-QnA loadgen results.
Evaluates both retrieval accuracy and answer quality using LLM judge.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests


# OpenRouter configuration
DEFAULT_JUDGE_URL = "http://127.0.0.1:8125/v1/chat/completions"
DEFAULT_JUDGE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
# Masked API key (set OPENROUTER_API_KEY environment variable to use OpenRouter)
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY',
    'sk-or-v1-****')


JUDGE_PROMPT = """You are grading whether an LLM answer is correct against a ground truth answer.

QUESTION: {question}

GROUND TRUTH ANSWER: {ground_truth}

LLM ANSWER: {llm_answer}

Grade in two steps.

STEP 1 - If the LLM answer is empty, "Unknown", "I don't know", "cannot be determined", or otherwise does not commit to an answer, then it is WRONG: output correct=false immediately and do not go to step 2.

STEP 2 - Otherwise compare it to the ground truth by meaning, not wording. correct=true only if it supplies every fact the ground truth requires and each clearly matches; if you are unsure or the match is only partial, output correct=false. Rules:
- If the ground truth is a list or has multiple parts, an answer missing any of them is correct=false.
- Every number, date, and name must match the ground truth; a different or differently-rounded value is correct=false, a different name is correct=false.
- Do NOT penalize harmless extras or omissions when the required facts match: a missing suffix like "Inc.", an added state/country, a full middle name, missing units when the number is right, or a briefer/longer phrasing.

Return your evaluation in JSON format:
{{
    "correct": true/false,
    "reasoning": "brief explanation"
}}
"""


def call_judge(question: str, ground_truth: str, llm_answer: str,
               service_url: str = DEFAULT_JUDGE_URL,
               model_name: str = DEFAULT_JUDGE_MODEL,
               api_key: str = OPENROUTER_API_KEY) -> Dict:
    """Call LLM judge to evaluate answer correctness."""

    prompt = JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        llm_answer=llm_answer
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 1024
    }

    try:
        response = requests.post(service_url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        result = response.json()

        content = result['choices'][0]['message']['content']

        # Parse JSON response
        content = content.strip()

        # Extract JSON from markdown code blocks
        if "```" in content:
            json_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
            if json_block_match:
                content = json_block_match.group(1).strip()

        # Try to extract JSON object
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        else:
            return {"correct": False, "reasoning": "No JSON found in judge response"}

        judge_result = json.loads(content)
        return judge_result

    except Exception as e:
        print(f"Error calling judge: {e}")
        return {"correct": False, "reasoning": f"Judge error: {e}"}


def calculate_retrieval_metrics(retrieved_urls: List[str], expected_urls: List[str]) -> Dict:
    """Calculate precision, recall, F1 for retrieval."""

    retrieved_set = set(retrieved_urls)
    expected_set = set(expected_urls)

    if not expected_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    correct = retrieved_set & expected_set

    precision = len(correct) / len(retrieved_set) if retrieved_set else 0.0
    recall = len(correct) / len(expected_set) if expected_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def evaluate_results(results: Dict, dataset_path: str, num_workers: int = 4,
                    judge_service_url: str = DEFAULT_JUDGE_URL,
                    judge_model: str = DEFAULT_JUDGE_MODEL) -> Dict:
    """
    Evaluate loadgen results.

    Args:
        results: Dict mapping query_id -> result_dict
        dataset_path: Path to frames_dataset.tsv
        num_workers: Number of parallel judge workers
        judge_service_url: Judge LLM service URL
        judge_model: Judge LLM model name

    Returns:
        Dict with aggregate metrics
    """

    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path, sep='\t')

    # Build query -> ground truth mapping
    query_to_gt = {}
    for _, row in df.iterrows():
        query = row['Prompt']
        query_to_gt[query] = {
            'answer': row['Answer'],
            'expected_urls': []
        }
        # Extract expected URLs
        for col in df.columns:
            if col.startswith('wikipedia_link_'):
                url = row[col]
                if pd.notna(url) and url != '':
                    query_to_gt[query]['expected_urls'].append(url)

    print(f"Evaluating {len(results)} queries...")
    print(f"Using judge: {judge_model} at {judge_service_url}")

    # Metrics accumulators
    total_retrieval_precision = 0.0
    total_retrieval_recall = 0.0
    total_retrieval_f1 = 0.0
    total_answer_correct = 0
    total_queries = 0

    detailed_results = []

    def evaluate_single_query(query_id, result):
        """Evaluate a single query result."""
        query = result.get('query', '')
        llm_answer = result.get('answer', '')
        retrieved_urls = result.get('retrieved_urls', [])

        # Get ground truth
        gt_data = query_to_gt.get(query)
        if not gt_data:
            print(f"Warning: No ground truth for query: {query[:50]}...")
            return None

        ground_truth = gt_data['answer']
        expected_urls = gt_data['expected_urls']

        # Calculate retrieval metrics
        retrieval_metrics = calculate_retrieval_metrics(retrieved_urls, expected_urls)

        # Judge answer correctness
        judge_result = call_judge(query, ground_truth, llm_answer,
                                 service_url=judge_service_url,
                                 model_name=judge_model)
        answer_correct = judge_result.get('correct', False)

        return {
            'query_id': query_id,
            'query': query,
            'retrieval_precision': retrieval_metrics['precision'],
            'retrieval_recall': retrieval_metrics['recall'],
            'retrieval_f1': retrieval_metrics['f1'],
            'answer_correct': 1 if answer_correct else 0,
            'judge_reasoning': judge_result.get('reasoning', ''),
            'llm_answer': llm_answer,
            'ground_truth': ground_truth
        }

    # Parallel evaluation
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for query_id, result in results.items():
            future = executor.submit(evaluate_single_query, query_id, result)
            futures.append(future)

        for future in as_completed(futures):
            try:
                eval_result = future.result()
                if eval_result:
                    detailed_results.append(eval_result)
                    total_retrieval_precision += eval_result['retrieval_precision']
                    total_retrieval_recall += eval_result['retrieval_recall']
                    total_retrieval_f1 += eval_result['retrieval_f1']
                    total_answer_correct += eval_result['answer_correct']
                    total_queries += 1

                    if total_queries % 10 == 0:
                        print(f"  Evaluated {total_queries}/{len(results)} queries...")
            except Exception as e:
                print(f"Error evaluating query: {e}")

    # Calculate averages
    if total_queries > 0:
        avg_metrics = {
            'total_queries': total_queries,
            'retrieval_precision': total_retrieval_precision / total_queries,
            'retrieval_recall': total_retrieval_recall / total_queries,
            'retrieval_f1': total_retrieval_f1 / total_queries,
            'answer_accuracy': total_answer_correct / total_queries,
            'detailed_results': detailed_results
        }
    else:
        avg_metrics = {
            'total_queries': 0,
            'retrieval_precision': 0.0,
            'retrieval_recall': 0.0,
            'retrieval_f1': 0.0,
            'answer_accuracy': 0.0,
            'detailed_results': []
        }

    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG-QnA loadgen accuracy")
    parser.add_argument('--log_dir', required=True, help='Loadgen log directory')
    parser.add_argument('--results_file', required=True, help='SUT results JSON file')
    parser.add_argument('--dataset_path', required=True, help='Path to frames_dataset.tsv')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel judge workers')
    parser.add_argument('--output', default='accuracy_results.json', help='Output file for detailed results')
    parser.add_argument('--judge_service_url', default=DEFAULT_JUDGE_URL, help='Judge LLM service URL')
    parser.add_argument('--judge_model', default=DEFAULT_JUDGE_MODEL, help='Judge LLM model name')
    args = parser.parse_args()

    # Load results
    print(f"Loading results from {args.results_file}...")
    with open(args.results_file, 'r') as f:
        results = json.load(f)

    print(f"Loaded {len(results)} results")

    # Evaluate
    metrics = evaluate_results(results, args.dataset_path, args.num_workers,
                               judge_service_url=args.judge_service_url,
                               judge_model=args.judge_model)

    # Print summary
    print("\n" + "="*80)
    print("ACCURACY EVALUATION RESULTS")
    print("="*80)
    print(f"Total Queries:        {metrics['total_queries']}")
    print(f"\nRetrieval Metrics:")
    print(f"  Precision@N:        {metrics['retrieval_precision']:.3f}")
    print(f"  Recall@N:           {metrics['retrieval_recall']:.3f}")
    print(f"  F1@N:               {metrics['retrieval_f1']:.3f}")
    print(f"\nAnswer Quality:")
    print(f"  LLM Judge Accuracy: {metrics['answer_accuracy']:.3f}")
    print("="*80 + "\n")

    # Save detailed results
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Detailed results saved to {args.output}")

    # Write accuracy.txt into the loadgen log dir in MLPerf format. The
    # submission checker parses the LLM judge answer accuracy (as a percentage)
    # from the "Accuracy:" line. The hash= line and log truncation are added
    # later by tools/submission/truncate_accuracy_log.py during submission prep.
    accuracy_txt_path = os.path.join(args.log_dir, "accuracy.txt")
    with open(accuracy_txt_path, 'w') as f:
        f.write(f"Accuracy: {metrics['answer_accuracy'] * 100:.4f}\n")
    print(f"Accuracy report saved to {accuracy_txt_path}")


if __name__ == "__main__":
    main()
