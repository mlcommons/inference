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
Evaluate single-shot or oracle results using an LLM judge.

Supports loading results from:
- JSON files (legacy format): result_single_shot.json
- Pickle files (oracle format): oracle_checkpoint.pkl (pandas DataFrame)

Usage:
    OPENROUTER_API_KEY="sk-or-v1-..." python evaluate.py result_single_shot.json
    OPENROUTER_API_KEY="sk-or-v1-..." python evaluate.py oracle_checkpoint.pkl
    OPENROUTER_API_KEY="sk-or-v1-..." python evaluate.py oracle_checkpoint.pkl --dataset data/frames_dataset.tsv
"""

import argparse
import json
import os
import pickle
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# LLM judge configuration (defaults to local vLLM)
DEFAULT_JUDGE_URL = "http://127.0.0.1:8123/v1/chat/completions"
DEFAULT_JUDGE_MODEL = "gpt-oss-20b"
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')


def load_results(path: Path):
    """Load results from either JSON or pickle checkpoint."""
    if path.suffix == '.pkl':
        # Load pandas DataFrame checkpoint
        with open(path, 'rb') as f:
            df = pickle.load(f)
        
        # Convert DataFrame to dict: query -> llm_answer
        # Only include successfully completed queries
        successful = df[df['success'] == True]
        return {row['query']: row['llm_answer'] for _, row in successful.iterrows()}
    else:
        # Legacy JSON format
        data = json.loads(path.read_text(encoding="utf-8"))
        results = data.get("results", [])
        return {entry.get("prompt"): entry.get("llm_answer", "") for entry in results if entry.get("prompt")}


def _parse_score_value(value) -> int:
    """Normalize judge score values to 0 or 1."""
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if float(value) >= 0.5 else 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "correct", "yes"}:
            return 1
        if lowered in {"0", "false", "incorrect", "no"}:
            return 0
        try:
            return 1 if float(lowered) >= 0.5 else 0
        except ValueError:
            return 0
    return 0


def _extract_json_dict(content: str) -> Optional[dict]:
    """Attempt to recover a JSON object from the judge response."""
    if not content:
        return None

    stripped = content.strip()

    candidates = []

    # Remove optional fenced code blocks (``` or ```json)
    if stripped.startswith("```"):
        fence_stripped = stripped.split("```", 1)[1]
        fence_stripped = fence_stripped.strip()
        if fence_stripped.lower().startswith("json"):
            fence_stripped = fence_stripped[4:].strip()
        closing_idx = fence_stripped.find("```")
        if closing_idx != -1:
            fence_stripped = fence_stripped[:closing_idx]
        candidates.append(fence_stripped.strip())

    # Look for a JSON object substring.
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if match:
        candidates.append(match.group(0))

    candidates.append(stripped)

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def call_judge(session: requests.Session, service_url: str, model: str, question: str, gold: str, pred: str):
    prompt = (
        "You judge whether the model answer correctly answers the question based on semantic equivalence to the gold answer.\n\n"
        "GRADING RULES:\n"
        "1. If model answer and gold answer are EXACTLY the same (ignoring case/punctuation), score 1\n"
        "2. Focus on whether the model answer contains the KEY INFORMATION that answers the question\n"
        "3. Minor differences are acceptable:\n"
        "   - Missing articles (a, an, the)\n"
        "   - Missing units when the number is correct (e.g., '50' vs '50 years')\n"
        "   - Additional correct details not in gold answer\n"
        "   - Different word order for lists (e.g., 'A and B' vs 'B, A')\n"
        "   - Missing location qualifiers when answer is already specific (e.g., 'Las Vegas' vs 'Las Vegas, Nevada')\n"
        "   - Different formatting (e.g., 'Dwight D. Eisenhower' vs 'Dwight D Eisenhower')\n"
        "   - Brief answers that directly answer the question vs verbose gold answers\n"
        "4. Score 0 ONLY if:\n"
        "   - Model answer is factually wrong\n"
        "   - Model answer is missing ESSENTIAL information that changes the meaning\n"
        "   - Model answer does NOT answer what the question asked\n"
        "5. Do NOT penalize for:\n"
        "   - Lack of context/explanation when question doesn't require it\n"
        "   - Different level of detail if core answer is correct\n"
        "   - Missing information the question didn't ask for\n\n"
        f"Question: {question}\n"
        f"Gold Answer: {gold}\n"
        f"Model Answer: {pred}\n\n"
        "Return JSON with keys score (1 or 0) and explanation."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a lenient grading assistant focused on semantic correctness, not exact string matching."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 256
    }

    # Add OpenRouter authentication headers if using OpenRouter
    headers = {}
    if "openrouter.ai" in service_url and OPENROUTER_API_KEY:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://github.com/anthropics/e2e-docgrader",
            "X-Title": "E2E DocGrader Evaluation"
        }

    response = session.post(service_url, json=payload, headers=headers, timeout=120)
    response.raise_for_status()
    data = response.json()
    # Defensive: handle missing or malformed 'choices' in response
    choices = data.get("choices")
    if not choices or not isinstance(choices, list) or not choices[0] or "message" not in choices[0] or "content" not in choices[0]["message"]:
        print("[ERROR] Judge response missing 'choices' or 'content':", data)
        # Return score 0, explanation with raw response, and raw data
        return 0, f"Malformed judge response: {data}", str(data)
    content = choices[0]["message"]["content"]
    if content is None:
        print("[ERROR] Judge response 'content' is None:", data)
        return 0, f"Judge response content is None: {data}", str(data)
    content = content.strip()

    parsed = _extract_json_dict(content)

    if parsed is not None:
        score_value = parsed.get("score")
        score = _parse_score_value(score_value)
        explanation_value = parsed.get("explanation")
        if isinstance(explanation_value, str):
            explanation = explanation_value.strip()
        elif explanation_value is None:
            explanation = ""
        else:
            # Convert non-string explanations (e.g., dict) into compact JSON
            explanation = json.dumps(explanation_value, ensure_ascii=False)
    else:
        normalized = content.lower()
        score = 1 if "correct" in normalized and "incorrect" not in normalized else 0
        explanation = content

    return score, explanation, content


def _judge_row(idx, prompt, gold, pred, service_url, model):
    """Call judge for a single row, returning (idx, prompt, gold, pred, score, explanation, raw)."""
    session = requests.Session()
    score, explanation, raw = call_judge(session, service_url, model, prompt, gold, pred)
    return idx, prompt, gold, pred, score, explanation, raw


def evaluate(results_path: Path, dataset_path: Path, service_url: str, model: str, batch_size: int = 16):
    # Check for OpenRouter API key if using OpenRouter
    if "openrouter.ai" in service_url and not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY environment variable not set")
        print("Usage: OPENROUTER_API_KEY=\"sk-or-v1-YOUR_KEY_HERE\" python evaluate.py ...")
        exit(1)

    predictions = load_results(results_path)

    # Show checkpoint stats if loading from pickle
    if results_path.suffix == '.pkl':
        with open(results_path, 'rb') as f:
            checkpoint_df = pickle.load(f)
        print(f"CHECKPOINT STATISTICS")
        print("=" * 80)
        print(f"Total queries in checkpoint: {len(checkpoint_df)}")
        print(f"Successful queries: {(checkpoint_df['success'] == True).sum()}")
        print(f"Failed queries: {(checkpoint_df['success'] == False).sum()}")
        if 'num_docs' in checkpoint_df.columns:
            total_docs = checkpoint_df['num_docs'].sum()
            total_missing = checkpoint_df['num_missing_docs'].sum()
            print(f"Total documents referenced: {total_docs}")
            print(f"Missing documents: {total_missing} ({100*total_missing/total_docs:.2f}%)")
        print("=" * 80)
        print()

    df = pd.read_csv(dataset_path, sep="\t")

    # Build list of items to judge
    items = []
    for idx, row in df.iterrows():
        prompt = row.get("Prompt")
        gold = str(row.get("Answer", "")).strip()
        if prompt not in predictions:
            continue
        pred = str(predictions[prompt]).strip()
        items.append((idx, prompt, gold, pred))

    if not items:
        print("No matching predictions found in results file.")
        return

    total = len(items)
    unknown = sum(1 for _, _, _, pred in items if pred.lower() == "unknown")

    # Submit all judge calls in parallel (batch_size workers), print as they complete
    score_sum = 0
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {
            executor.submit(_judge_row, idx, prompt, gold, pred, service_url, model): idx
            for idx, prompt, gold, pred in items
        }
        for future in as_completed(futures):
            idx, prompt, gold, pred, score, explanation, raw = future.result()
            score_sum += score
            print("=" * 80)
            print(f"Prompt {idx}: {prompt}")
            print(f"Gold: {gold}")
            print(f"Answer: {pred}")
            print(f"Judge Score: {score}")
            print(f"Judge Explanation: {explanation if explanation else raw}")

    judged = len(items)
    accuracy = score_sum / judged
    unknown_ratio = unknown / total
    print("\nSUMMARY")
    print("-" * 80)
    print(f"Evaluated Samples: {judged}")
    print(f"Unknown Ratio: {unknown_ratio:.3f}")
    print(f"Accuracy: {accuracy:.3f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate single-shot results using an LLM judge.")
    parser.add_argument("results", type=Path, help="Path to results (result_single_shot.json or oracle_checkpoint.pkl)")
    parser.add_argument("--dataset", type=Path, default=Path("data/frames_dataset.tsv"), help="Evaluation dataset TSV")
    parser.add_argument("--judge-url", default=DEFAULT_JUDGE_URL, help="Judge service endpoint")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Judge model identifier")
    parser.add_argument("--batch-size", type=int, default=16, help="Number of concurrent judge requests (default: 16)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.results, args.dataset, args.judge_url, args.judge_model, args.batch_size)
