import argparse
import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

DEFAULT_JUDGE_URL = "http://127.0.0.1:8124/v1/chat/completions"
DEFAULT_JUDGE_MODEL = "/mnt/weka/data/pytorch/llama3.1/Meta-Llama-3.1-8B-Instruct/"


def load_results(path: Path):
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
    response = session.post(service_url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"].strip()

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


def evaluate(results_path: Path, dataset_path: Path, service_url: str, model: str):
    predictions = load_results(results_path)
    df = pd.read_csv(dataset_path, sep="\t")
    session = requests.Session()
    total = 0
    judged = 0
    unknown = 0
    score_sum = 0

    for _, row in df.iterrows():
        prompt = row.get("Prompt")
        gold = str(row.get("Answer", "")).strip()
        if prompt not in predictions:
            continue
        pred = str(predictions[prompt]).strip()
        total += 1
        if pred.lower() == "unknown":
            unknown += 1
        score, explanation, raw = call_judge(session, service_url, model, prompt, gold, pred)
        judged += 1
        score_sum += score
        print("=" * 80)
        print(f"Prompt: {prompt}")
        print(f"Gold: {gold}")
        print(f"Answer: {pred}")
        print(f"Judge Score: {score}")
        print(f"Judge Explanation: {explanation if explanation else raw}")

    if judged == 0:
        print("No matching predictions found in results file.")
        return

    accuracy = score_sum / judged if judged else 0.0
    unknown_ratio = unknown / total if total else 0.0
    print("\nSUMMARY")
    print("-" * 80)
    print(f"Evaluated Samples: {judged}")
    print(f"Unknown Ratio: {unknown_ratio:.3f}")
    print(f"Accuracy: {accuracy:.3f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate single-shot results using an LLM judge.")
    parser.add_argument("results", type=Path, help="Path to result_single_shot.json")
    parser.add_argument("--dataset", type=Path, default=Path("data/frames_dataset.tsv"), help="Evaluation dataset TSV")
    parser.add_argument("--judge-url", default=DEFAULT_JUDGE_URL, help="Judge service endpoint")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Judge model identifier")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.results, args.dataset, args.judge_url, args.judge_model)
