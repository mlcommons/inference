#!/usr/bin/env python3
"""
Standalone evaluation script for mlperf-inference deepseek-r1 dataset.

Expected input format (pickle file with DataFrame):
- model_output: The model's response text
- tok_model_output_len: The length of the model's response tokens
- ground_truth: The expected answer (not required for healthbench)
- dataset: Dataset name (e.g., 'gpqa', 'mmlu_pro', 'math500', 'livecodebench', 'aime', 'healthbench')
- question: The question text
- rubrics: List of rubric items (required for healthbench)
- prompt: Conversation history (required for healthbench)

Output adds columns:
- extracted_answer: Parsed answer from model output
- prompt_accuracy: 100.0 if correct, 0.0 if incorrect
- evaluation_details: Detailed evaluation explanation (for healthbench)

For HealthBench evaluation, set OPENAI_API_KEY environment variable for LLM-as-a-judge grading.
"""

import sys
import os
import argparse
import logging
import pickle
import json
import re
import shutil
import time
from functools import lru_cache
from typing import Dict, Any, Optional, Tuple, Union
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path

# MLPerf log processing imports
import numpy as np
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global configuration for HealthBench LLM judge
LLM_JUDGE_BACKEND = "openai"  # "openai" or "nvidia"
LLM_JUDGE_MODEL = None  # None = auto-select based on backend

# =============================================================================
# Input Validation
# =============================================================================


def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate input DataFrame has required columns."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    required_cols = [
        'model_output',
        'dataset',
        'tok_model_output_len']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for ground_truth or rubrics depending on dataset
    has_ground_truth = 'ground_truth' in df.columns
    has_rubrics = 'rubrics' in df.columns

    if not has_ground_truth and not has_rubrics:
        raise ValueError(
            "DataFrame must have either 'ground_truth' or 'rubrics' column")


def validate_text_input(text: Any) -> str:
    """Validate and convert text input to string."""
    if pd.isna(text) or text is None:
        return ""
    return str(text).strip()


def validate_dataset_name(dataset: Any) -> str:
    """Validate dataset name."""
    if pd.isna(dataset) or not dataset:
        raise ValueError("Dataset name cannot be empty")
    return str(dataset).lower()


# =============================================================================
# Answer Parsing Functions
# =============================================================================

def parse_multiple_choice(text: str, max_option: str = 'D') -> Optional[str]:
    """Parse multiple choice answer (A-D or A-J)."""
    text = validate_text_input(text)
    if not text:
        return None

    # Clean artifacts
    if text.startswith(("['", '["')) and text.endswith(("']", '"]')):
        text = text[2:-2].strip()

    text = text.replace(r'\n', '\n').replace(r'\'', "'")

    # Find ANSWER/FINAL ANSWER pattern
    pattern = rf"\b(?:ANSWER|FINAL\s*ANSWER)\b\s*[:=]?\s*(?:\(?\s*([A-{max_option}])\s*\)?)(?:\s*$|[^A-Za-z])"
    matches = list(re.finditer(pattern, text, re.IGNORECASE))

    if matches:
        return matches[-1].group(1).upper()

    # MMLU-Pro fallback: standalone letter
    if max_option == 'J':
        fallback_matches = list(re.finditer(
            r"\b([A-J])\b", text, re.IGNORECASE))
        if fallback_matches:
            return fallback_matches[-1].group(1).upper()

    return None


def parse_boxed_math(text: str) -> Optional[str]:
    """Parse \\boxed{answer} format."""
    text = validate_text_input(text)
    if not text:
        return None

    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None

    # Find matching brace
    depth, i = 0, idx + 7
    content_start = i
    while i < len(text):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            if depth == 0:
                return text[content_start:i].strip()
            depth -= 1
        i += 1
    return None


def parse_aime_answer(text: str) -> Optional[int]:
    """Parse AIME integer answer (0-999)."""
    text = validate_text_input(text)
    if not text:
        return None

    # Priority 1: \boxed{digits}
    boxed_matches = list(re.finditer(r"\\boxed{\s*(\d+)\s*}", text))
    if boxed_matches:
        extracted_str = boxed_matches[-1].group(1)
    else:
        # Priority 2: Answer: <digits>
        answer_matches = list(re.finditer(
            r"Answer:\s*(\d+)(?!\.)\b", text, re.IGNORECASE | re.MULTILINE))
        if not answer_matches:
            return None
        extracted_str = answer_matches[-1].group(1)

    try:
        val = int(extracted_str)
        if 0 <= val <= 999:
            return val
    except ValueError:
        pass

    return None


def parse_code(text: str) -> Optional[str]:
    """Parse code from ```python code block."""
    text = validate_text_input(text)
    if not text:
        return None

    match = re.search(r"```python(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else None


# =============================================================================
# Answer Evaluation Functions
# =============================================================================

def evaluate_multiple_choice(
        parsed: Optional[str], ground_truth: str, valid_options: str) -> bool:
    """Evaluate multiple choice answer."""
    if not parsed or not ground_truth:
        return False

    parsed = parsed.upper()
    ground_truth = ground_truth.upper()

    return parsed in valid_options and parsed == ground_truth


def evaluate_math500(parsed: Optional[str], ground_truth: str) -> bool:
    """Evaluate MATH-500 using PRM800K grader."""
    if not parsed or not ground_truth:
        return False

    parsed = str(parsed).strip()
    ground_truth = str(ground_truth)

    if not parsed:
        return False

    # Use sys.path approach for proper module importing
    workspace_path = os.path.dirname(os.path.abspath(__file__))
    prm800k_module_path = os.path.join(
        workspace_path, "submodules", "prm800k", "prm800k")

    if not os.path.exists(prm800k_module_path):
        raise FileNotFoundError(
            f"PRM800K module not found at: {prm800k_module_path}")

    # Save current directory and sys.path
    original_cwd = os.getcwd()
    original_syspath = sys.path.copy()

    try:
        # Add prm800k module path to sys.path
        if prm800k_module_path not in sys.path:
            sys.path.insert(0, prm800k_module_path)

        # Change directory as some imports might use relative paths
        os.chdir(prm800k_module_path)

        # Now import should work
        from grading.grader import grade_answer
        result = grade_answer(given_answer=parsed, ground_truth=ground_truth)
    except ImportError as e:
        raise ImportError(f"Failed to import PRM800K grader: {e}")
    finally:
        # Always restore original directory and sys.path
        os.chdir(original_cwd)
        sys.path[:] = original_syspath

    return result


def evaluate_aime(parsed: Optional[int], ground_truth: Any) -> bool:
    """Evaluate AIME integer answer."""
    if parsed is None:
        return False

    try:
        gt_int = int(ground_truth)
        return int(parsed) == gt_int
    except (ValueError, TypeError):
        return False


@lru_cache(maxsize=1)
def load_lcb_benchmark() -> Dict[str, Any]:
    """Load LiveCodeBench benchmark with caching."""
    lcb_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "submodules", "LiveCodeBench"))

    if not os.path.isdir(lcb_dir):
        raise FileNotFoundError(
            f"LiveCodeBench submodule required at: {lcb_dir}")

    original_cwd = os.getcwd()
    os.chdir(lcb_dir)

    if lcb_dir not in sys.path:
        sys.path.insert(0, lcb_dir)

    try:
        os.environ['TQDM_DISABLE'] = '1'

        from lcb_runner.utils.scenarios import Scenario
        from lcb_runner.runner.scenario_router import build_prompt_benchmark

        mock_args = argparse.Namespace(
            scenario=Scenario.codegeneration, release_version="release_v1",
            subset="code_generation", language="python", not_fast=False,
            start_date=None, end_date=None, k=[1], num_samples=1,
            timeout=60, num_workers=1, num_process_evaluate=1,
            model_name="standalone_eval", output_dir="/tmp",
            prompt_type="custom", continue_existing=False, evaluate=True
        )

        full_benchmark, _ = build_prompt_benchmark(mock_args)
        return {inst.question_id: inst for inst in full_benchmark}

    finally:
        os.chdir(original_cwd)
        os.environ.pop('TQDM_DISABLE', None)


def evaluate_livecodebench(code: Optional[str], question_id: str) -> bool:
    """Evaluate LiveCodeBench code generation."""
    if not code or not question_id:
        return False

    lcb_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "submodules", "LiveCodeBench"))
    benchmark_map = load_lcb_benchmark()

    instance = benchmark_map.get(question_id)
    if not instance:
        return False

    original_cwd = os.getcwd()
    temp_dir = f"/tmp/temp_lcb_eval_{question_id}_{int(time.time())}"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        os.chdir(lcb_dir)
        os.environ['TQDM_DISABLE'] = '1'

        from lcb_runner.utils.scenarios import Scenario
        from lcb_runner.evaluation import extract_instance_results
        from lcb_runner.runner.scenario_router import sort_and_extract_save_results, get_metrics

        mock_args = argparse.Namespace(
            scenario=Scenario.codegeneration, release_version="release_v1",
            subset="code_generation", language="python", not_fast=False,
            start_date=None, end_date=None, k=[1], num_samples=1,
            timeout=60, num_workers=1, num_process_evaluate=1,
            model_name="inline_handler_eval", output_dir=temp_dir,
            prompt_type="custom", continue_existing=False, evaluate=True,
        )

        batch_benchmark = [instance]
        batch_custom_outputs = [[code]]

        save_results = [inst.insert_output(output, output)
                        for inst, output in zip(batch_benchmark, batch_custom_outputs)]

        _, combined_results = sort_and_extract_save_results(
            mock_args.scenario, save_results)
        _, instance_results, _ = get_metrics(
            mock_args.scenario, mock_args, batch_benchmark, combined_results
        )

        graded = extract_instance_results(instance_results)
        return graded and graded[0] and graded[0][0]

    finally:
        os.chdir(original_cwd)
        shutil.rmtree(temp_dir, ignore_errors=True)
        os.environ.pop('TQDM_DISABLE', None)


def evaluate_livecodebench_worker(args: Tuple[str, str]) -> Tuple[str, bool]:
    """Worker function for parallel LiveCodeBench evaluation."""
    code, question_id = args

    try:
        return question_id, evaluate_livecodebench(code, question_id)
    except Exception:
        return question_id, False


# =============================================================================
# HealthBench Evaluation Functions
# =============================================================================

HEALTHBENCH_GRADER_TEMPLATE = """Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<CONVERSATION>

# Rubric item
<RUBRIC_ITEM>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".

- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item.

If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true if all of the criteria are met.

- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:
```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.

For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:
```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the criteria says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:
```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()


class RubricItem:
    """Represents a single rubric criterion for HealthBench evaluation."""

    def __init__(self, criterion: str, points: float, tags: list):
        self.criterion = criterion
        self.points = points
        self.tags = tags

    def __str__(self):
        return f"[{self.points}] {self.criterion}"

    def to_dict(self):
        return {
            "criterion": self.criterion,
            "points": self.points,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            criterion=d["criterion"],
            points=d["points"],
            tags=d.get("tags", []),
        )


def parse_healthbench_json(json_string: str) -> dict:
    """Parse JSON response from grader, handling markdown code blocks."""
    json_cleaned = re.sub(
        r"^```json\s*|\s*```$",
        "",
        json_string.strip(),
        flags=re.MULTILINE)
    try:
        return json.loads(json_cleaned)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON decoding failed: {e}")
        return {"explanation": "Failed to parse response", "criteria_met": False}


def calculate_healthbench_score(
    rubric_items: list, grading_responses: list
) -> float:
    """Calculate HealthBench score based on rubric items and grading responses.

    Args:
        rubric_items: List of RubricItem objects
        grading_responses: List of dicts with 'criteria_met' and 'explanation'

    Returns:
        Score between 0 and 1, or 0 if no positive points available
    """
    total_possible_points = sum(
        item.points for item in rubric_items if item.points > 0
    )

    if total_possible_points == 0:
        return 0.0

    achieved_points = sum(
        item.points
        for item, response in zip(rubric_items, grading_responses)
        if response.get("criteria_met", False)
    )

    overall_score = achieved_points / total_possible_points
    return max(0.0, min(1.0, overall_score))  # Clamp to [0, 1]


def grade_healthbench_with_llm(
    prompt_messages: list,
    model_output: str,
    rubric_items: list,
    grader_api_key: Optional[str] = None,
    grader_model: str = "gpt-4o-mini",
    grader_base_url: str = "https://api.openai.com/v1",
    grader_backend: str = "openai"
) -> Tuple[float, str]:
    """Grade a HealthBench response using LLM-as-a-judge.

    Args:
        prompt_messages: List of conversation messages
        model_output: The model's response to grade
        rubric_items: List of RubricItem objects
        grader_api_key: API key for grader (OpenAI or NVIDIA NIM)
        grader_model: Model to use for grading
        grader_base_url: Base URL for API
        grader_backend: Backend to use - "openai" or "nvidia" (default: "openai")

    Returns:
        Tuple of (score, detailed_explanation)
    """
    # Use API key from environment if not provided
    if grader_api_key is None:
        if grader_backend == "nvidia":
            grader_api_key = os.environ.get("NVIDIA_NIM_API_KEY")
            if not grader_api_key:
                logger.warning(
                    "No NVIDIA NIM API key found. Set NVIDIA_NIM_API_KEY environment variable.")
                return 0.0, "Error: No NVIDIA NIM API key provided"
        else:
            grader_api_key = os.environ.get("OPENAI_API_KEY")
            if not grader_api_key:
                logger.warning(
                    "No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
                return 0.0, "Error: No OpenAI API key provided"

    # Format conversation
    conversation_text = ""
    for msg in prompt_messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        conversation_text += f"{role}: {content}\n\n"
    conversation_text += f"assistant: {model_output}"

    grading_responses = []

    for rubric_item in rubric_items:
        # Create grading prompt
        grading_prompt = HEALTHBENCH_GRADER_TEMPLATE.replace(
            "<CONVERSATION>", conversation_text
        ).replace("<RUBRIC_ITEM>", str(rubric_item))

        try:
            if grader_backend == "nvidia":
                # Use NVIDIA NIM endpoint
                response_text = _call_nvidia_nim_api(
                    api_key=grader_api_key,
                    model=grader_model,
                    messages=[{"role": "user", "content": grading_prompt}],
                    base_url=grader_base_url,
                    temperature=0.0,
                    max_tokens=1024
                )
            else:
                # Use OpenAI endpoint
                response_text = _call_openai_api(
                    api_key=grader_api_key,
                    model=grader_model,
                    messages=[{"role": "user", "content": grading_prompt}],
                    base_url=grader_base_url,
                    temperature=0.0,
                    max_tokens=1024
                )

            grading_result = parse_healthbench_json(response_text)
            grading_responses.append(grading_result)

        except Exception as e:
            logger.warning(f"Error grading rubric item: {e}")
            grading_responses.append({
                "explanation": f"Error during grading: {e}",
                "criteria_met": False
            })

    # Calculate overall score
    score = calculate_healthbench_score(rubric_items, grading_responses)

    # Create detailed explanation
    explanations = []
    for rubric_item, response in zip(rubric_items, grading_responses):
        met = response.get("criteria_met", False)
        explanation = response.get("explanation", "No explanation")
        explanations.append(
            f"[{'✓' if met else '✗'}] {rubric_item}\n    Explanation: {explanation}"
        )

    detailed_explanation = "\n\n".join(explanations)

    return score, detailed_explanation


def _call_openai_api(
    api_key: str,
    model: str,
    messages: list,
    base_url: str,
    temperature: float = 0.0,
    max_tokens: int = 1024
) -> str:
    """Call OpenAI API for grading.

    Args:
        api_key: OpenAI API key
        model: Model name
        messages: List of messages
        base_url: Base URL for API
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response

    Returns:
        Response text from the model
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai package required. Install with: pip install openai")

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def _call_nvidia_nim_api(
    api_key: str,
    model: str,
    messages: list,
    base_url: str = "https://integrate.api.nvidia.com/v1/chat/completions",
    temperature: float = 0.0,
    max_tokens: int = 1024
) -> str:
    """Call NVIDIA NIM API for grading.

    Args:
        api_key: NVIDIA NIM API key
        model: Model name (e.g., 'deepseek-ai/deepseek-v3.1-terminus')
        messages: List of messages
        base_url: Base URL for NVIDIA NIM API
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response

    Returns:
        Response text from the model
    """
    try:
        import requests
    except ImportError:
        raise ImportError(
            "requests package required. Install with: pip install requests")

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    }

    payload = {
        'model': model,
        'messages': messages,
        'temperature': temperature,
        'max_tokens': max_tokens
    }

    response = requests.post(
        base_url,
        headers=headers,
        json=payload,
        timeout=200)
    response.raise_for_status()

    response_data = response.json()
    return response_data['choices'][0]['message']['content']


def parse_healthbench(text: str) -> Optional[str]:
    """Parse HealthBench response - returns the full text as-is."""
    return validate_text_input(text) or None


def evaluate_healthbench(
    parsed_output: Optional[str],
    row_data: pd.Series,
    grader_api_key: Optional[str] = None,
    grader_backend: str = "openai",
    grader_model: Optional[str] = None
) -> Tuple[bool, Optional[str]]:
    """Evaluate HealthBench response using LLM grading.

    Args:
        parsed_output: The model output text
        row_data: Full row data containing 'rubrics' and 'prompt'
        grader_api_key: Optional API key for grader
        grader_backend: Backend to use - "openai" or "nvidia" (default: "openai")
        grader_model: Optional model name override

    Returns:
        Tuple of (is_correct, detailed_explanation)
    """
    if not parsed_output:
        return False, "Empty output"

    # Extract rubrics from row
    rubrics = row_data.get('rubrics', [])
    if not rubrics:
        logger.warning("No rubrics found in row data")
        return False, "No rubrics available"

    # Convert to RubricItem objects
    rubric_items = [RubricItem.from_dict(r) for r in rubrics]

    # Extract prompt/conversation
    prompt = row_data.get('prompt', [])
    if isinstance(prompt, str):
        # If prompt is a string, convert to message format
        prompt = [{"role": "user", "content": prompt}]

    # Set default model based on backend
    if grader_model is None:
        if grader_backend == "nvidia":
            grader_model = "deepseek-ai/deepseek-v3.1-terminus"
        else:
            grader_model = "gpt-4o-mini"

    # Set base URL based on backend
    if grader_backend == "nvidia":
        grader_base_url = "https://integrate.api.nvidia.com/v1/chat/completions"
    else:
        grader_base_url = "https://api.openai.com/v1"

    # Grade using LLM
    score, explanation = grade_healthbench_with_llm(
        prompt_messages=prompt,
        model_output=parsed_output,
        rubric_items=rubric_items,
        grader_api_key=grader_api_key,
        grader_model=grader_model,
        grader_base_url=grader_base_url,
        grader_backend=grader_backend
    )

    # Consider "correct" if score >= 0.7 (70%)
    is_correct = score >= 0.7

    return is_correct, f"Score: {score:.2%}\n\n{explanation}"


# =============================================================================
# Dataset Configuration
# =============================================================================

DATASET_EVALUATORS = {
    'gpqa': {
        'parse': lambda text: parse_multiple_choice(text, 'D'),
        'evaluate': lambda parsed, gt: evaluate_multiple_choice(parsed, gt, 'ABCD')
    },
    'mmlu_pro': {
        'parse': lambda text: parse_multiple_choice(text, 'J'),
        'evaluate': lambda parsed, gt: evaluate_multiple_choice(parsed, gt, 'ABCDEFGHIJ')
    },
    'math500': {
        'parse': parse_boxed_math,
        'evaluate': evaluate_math500
    },
    'aime': {
        'parse': parse_aime_answer,
        'evaluate': evaluate_aime
    },
    'livecodebench': {
        'parse': parse_code,
        'evaluate': evaluate_livecodebench
    },
    'mmlu': {
        'parse': lambda text: parse_multiple_choice(text, 'J'),
        'evaluate': lambda parsed, gt: evaluate_multiple_choice(parsed, gt, 'ABCDEFGHIJ')
    },
    'healthbench': {
        'parse': parse_healthbench,
        'evaluate': evaluate_healthbench,
        'requires_row_data': True  # Special flag for HealthBench
    },

}


def get_evaluator(dataset_name: str) -> Dict[str, Any]:
    """Get evaluator functions for dataset."""
    dataset_lower = validate_dataset_name(dataset_name)

    for key, evaluator in DATASET_EVALUATORS.items():
        if key in dataset_lower:
            return evaluator

    raise ValueError(f"No evaluator found for dataset: {dataset_name}")


# =============================================================================
# Main Processing Functions
# =============================================================================

def process_row(row: pd.Series) -> Dict[str, Any]:
    """Process a single row and return extracted answer and accuracy."""
    dataset_name = validate_dataset_name(row['dataset'])
    raw_output = validate_text_input(row['model_output'])
    ground_truth = row['ground_truth']

    evaluator = get_evaluator(dataset_name)
    extracted = evaluator['parse'](raw_output)

    is_correct = False
    if extracted is not None and not pd.isna(ground_truth):
        is_correct = evaluator['evaluate'](extracted, ground_truth)

    return {
        'extracted_answer': extracted,
        'prompt_accuracy': 100.0 if is_correct else 0.0
    }


def process_livecodebench_parallel(
        df: pd.DataFrame, group_indices: pd.Index) -> Tuple[int, int]:
    """Process LiveCodeBench items in parallel."""
    # Prepare work items
    work_items = []
    for idx in group_indices:
        row = df.loc[idx]
        extracted = row.get('extracted_answer')
        ground_truth = row.get('ground_truth')

        if extracted is not None and not pd.isna(ground_truth):
            work_items.append((idx, extracted, ground_truth))

    if not work_items:
        return 0, 0

    # Process in parallel
    max_workers = min(multiprocessing.cpu_count(), len(work_items))
    logger.info(
        f"Evaluating {len(work_items)} LiveCodeBench items with {max_workers} workers")

    correct_count = 0
    total_evaluated = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(evaluate_livecodebench_worker, (code, question_id)): idx
            for idx, code, question_id in work_items
        }

        for future in tqdm(as_completed(future_to_idx, timeout=1200),
                           total=len(future_to_idx), desc="Evaluating LiveCodeBench"):
            idx = future_to_idx[future]

            try:
                question_id, is_correct = future.result(timeout=30)
                df.at[idx, 'prompt_accuracy'] = 100.0 if is_correct else 0.0
                total_evaluated += 1
                if is_correct:
                    correct_count += 1
            except Exception as e:
                logger.error(f"Error evaluating row {idx}: {e}")
                df.at[idx, 'prompt_accuracy'] = 0.0
                total_evaluated += 1

    return correct_count, total_evaluated


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Process entire dataframe with optimized batch processing."""
    validate_dataframe(df)

    df_output = df.copy()
    df_output['extracted_answer'] = None
    df_output['prompt_accuracy'] = 0.0

    # Process by dataset
    for dataset_name, group_indices in tqdm(df_output.groupby('dataset').groups.items(),
                                            desc="Processing datasets"):
        evaluator = get_evaluator(dataset_name)

        # Parse answers for all rows in this dataset
        logger.info(
            f"Processing {len(group_indices)} rows for dataset '{dataset_name}'")
        for idx in group_indices:
            row = df_output.loc[idx]
            raw_output = validate_text_input(row['model_output'])
            df_output.at[idx, 'extracted_answer'] = evaluator['parse'](
                raw_output)

        # Evaluate answers
        if 'livecodebench' in dataset_name.lower():
            correct_count, total_evaluated = process_livecodebench_parallel(
                df_output, group_indices)
        elif 'healthbench' in dataset_name.lower():
            # HealthBench evaluation with LLM grading
            correct_count = 0
            total_evaluated = 0

            for idx in tqdm(group_indices, desc=f"Evaluating {dataset_name}"):
                row = df_output.loc[idx]
                extracted = row['extracted_answer']

                if extracted is not None:
                    try:
                        # HealthBench needs full row data for rubrics and
                        # prompts
                        is_correct, explanation = evaluator['evaluate'](
                            extracted,
                            row,
                            grader_backend=LLM_JUDGE_BACKEND,
                            grader_model=LLM_JUDGE_MODEL
                        )
                        df_output.at[idx,
                                     'prompt_accuracy'] = 100.0 if is_correct else 0.0
                        # Store explanation in a new column if needed
                        if 'evaluation_details' not in df_output.columns:
                            df_output['evaluation_details'] = None
                        df_output.at[idx, 'evaluation_details'] = explanation
                        total_evaluated += 1
                        if is_correct:
                            correct_count += 1
                    except Exception as e:
                        logger.error(
                            f"Error evaluating HealthBench row {idx}: {e}")
                        df_output.at[idx, 'prompt_accuracy'] = 0.0
                        total_evaluated += 1
        else:
            # Sequential evaluation for other datasets
            correct_count = 0
            total_evaluated = 0

            for idx in group_indices:
                row = df_output.loc[idx]
                extracted = row['extracted_answer']
                ground_truth = row.get('ground_truth')

                if extracted is not None and not pd.isna(ground_truth):
                    is_correct = evaluator['evaluate'](extracted, ground_truth)
                    df_output.at[idx,
                                 'prompt_accuracy'] = 100.0 if is_correct else 0.0
                    total_evaluated += 1
                    if is_correct:
                        correct_count += 1

        # Log results
        if total_evaluated > 0:
            accuracy = correct_count / total_evaluated * 100
            logger.info(
                f"{dataset_name} results: {correct_count}/{total_evaluated} correct ({accuracy:.1f}% accuracy)")

    return df_output


# =============================================================================
# Unified Evaluation Utilities
# =============================================================================

def print_evaluation_results(df_evaluated: pd.DataFrame,
                             logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Print evaluation results in a unified format.

    Args:
        df_evaluated: DataFrame with evaluated results
        logger: Optional logger instance (uses module logger if not provided)

    Returns:
        Dictionary with evaluation statistics
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Calculate statistics
    evaluated = df_evaluated['extracted_answer'].notna().sum()
    correct = (df_evaluated['prompt_accuracy'] > 0).sum()
    accuracy = df_evaluated['prompt_accuracy'].mean()

    # tok_model_output_len is now a required column
    mean_output_len = float(df_evaluated['tok_model_output_len'].mean())

    results = {
        # 'evaluated': int(evaluated),
        # 'correct': int(correct),
        'exact_match': float(accuracy),
        'tokens_per_sample': mean_output_len,
        'num-samples': len(df_evaluated),
    }

    print("\nResults\n")
    print(results)


def process_and_save_dataframe(df: pd.DataFrame,
                               output_dir: Optional[Union[str, Path]] = None,
                               base_filename: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    """Process dataframe for evaluation and save the results.

    Args:
        df: Input DataFrame to evaluate
        output_dir: Directory to save the evaluated pickle file (defaults to same dir as source)
        base_filename: Base filename for output (defaults to auto-generated)

    Returns:
        Tuple of (evaluated_dataframe, saved_file_path)
    """
    # Process the dataframe
    df_evaluated = process_dataframe(df)

    # Determine output path
    if output_dir is None:
        # Try to infer from existing path info in the dataframe or use current
        # directory
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename if not provided
    if base_filename is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_filename = f"results_evaluated_{timestamp}.pkl"
    elif not base_filename.endswith('_evaluated.pkl'):
        # Ensure it ends with _evaluated.pkl
        if base_filename.endswith('.pkl'):
            base_filename = base_filename[:-4] + '_evaluated.pkl'
        else:
            base_filename = base_filename + '_evaluated.pkl'

    output_path = output_dir / base_filename

    # Save the evaluated dataframe
    with open(output_path, 'wb') as f:
        pickle.dump(df_evaluated, f)

    logger.info(f"Evaluated results saved to: {output_path}")

    return df_evaluated, str(output_path)


# =============================================================================
# Main Function
# =============================================================================

def detect_file_type(file_path: Union[str, Path]) -> str:
    """Detect whether file is MLPerf JSON or pickle format.

    Returns:
        "mlperf_json" or "pickle"
    """
    file_path = Path(file_path)

    # Check by extension first
    if file_path.suffix.lower() == '.json':
        return "mlperf_json"
    elif file_path.suffix.lower() in ['.pkl', '.pickle']:
        return "pickle"

    # Try to detect by content
    try:
        # Try reading as JSON first
        with open(file_path, 'r') as f:
            first_char = f.read(1)
            if first_char in ['[', '{']:
                # Likely JSON
                return "mlperf_json"
    except BaseException:
        pass

    # Default to pickle
    return "pickle"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model outputs - supports both pickle DataFrames and MLPerf JSON logs")
    parser.add_argument("--input-file", required=True,
                        help="Input file (pickle DataFrame or MLPerf JSON log)")
    parser.add_argument(
        "--output-file", help="Output pickle file (defaults to <input-file>_evaluated.pkl)")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose logging")
    parser.add_argument("--llm-judge-backend",
                        choices=["openai", "nvidia"],
                        default="openai",
                        help="Backend for HealthBench LLM judge (default: openai)")
    parser.add_argument("--llm-judge",
                        help="Model for HealthBench LLM judge (default: gpt-4o-mini for openai, deepseek-ai/deepseek-v3.1-terminus for nvidia)")

    args = parser.parse_args()

    # Set global configuration for HealthBench LLM judge
    global LLM_JUDGE_BACKEND, LLM_JUDGE_MODEL
    LLM_JUDGE_BACKEND = args.llm_judge_backend
    LLM_JUDGE_MODEL = args.llm_judge

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    input_path = Path(args.input_file)

    # Detect file type
    file_type = detect_file_type(input_path)
    logger.info(f"Detected input file type: {file_type}")

    # Determine output file path
    if args.output_file:
        output_path = Path(args.output_file)
        output_dir = output_path.parent
        output_filename = output_path.name
    else:
        output_dir = input_path.parent
        output_filename = input_path.stem + "_evaluated.pkl"

    logger.info(f"Processing: {args.input_file}")

    # Handle pickle DataFrame format
    logger.info("Processing pickle DataFrame file")

    # Load and process data
    with open(args.input_file, 'rb') as f:
        df = pickle.load(f)

    logger.info(f"Loaded {len(df)} rows")

    # Process and save with unified function
    df_evaluated, saved_file_path = process_and_save_dataframe(
        df,
        output_dir=output_dir,
        base_filename=output_filename
    )

    # Print evaluation results with unified function
    print_evaluation_results(df_evaluated, logger)

    logger.info("Done!")


if __name__ == "__main__":
    main()
