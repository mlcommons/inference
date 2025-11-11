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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, TimeoutError
import multiprocessing
from pathlib import Path

# MLPerf log processing imports
import numpy as np
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress verbose HTTP logs from OpenAI/httpx client
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Global configuration for HealthBench LLM judge
LLM_JUDGE_BASE_URL = None  # None = default to OpenAI API
LLM_JUDGE_MODEL = None  # None = auto-select based on base URL
LLM_JUDGE_API_KEY = None  # None = auto-select from environment
LLM_JUDGE_MAX_WORKERS = None  # None = auto-select based on rubric count

# =============================================================================
# Input Validation
# =============================================================================


def detect_pass_k(df: pd.DataFrame) -> int:
    """Detect if DataFrame has pass@k format and return k.

    Returns:
        Number of passes (k) if pass@k format detected, otherwise 1
    """
    # Check for model_output_0, model_output_1, etc.
    pass_k = 0
    while f'model_output_{pass_k}' in df.columns:
        pass_k += 1

    # If no _0 suffix found, check for single model_output column
    if pass_k == 0 and 'model_output' in df.columns:
        return 1

    return pass_k


def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate input DataFrame has required columns."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    # Detect pass@k format
    pass_k = detect_pass_k(df)

    if pass_k == 0:
        raise ValueError(
            "No model_output columns found (expected 'model_output' or 'model_output_0', 'model_output_1', etc.)")

    # Check for dataset column
    if 'dataset' not in df.columns:
        raise ValueError("Missing required column: 'dataset'")

    # Check for tok_model_output_len (either single or with suffixes)
    has_tok_len = False
    if pass_k == 1:
        has_tok_len = 'tok_model_output_len' in df.columns
    else:
        has_tok_len = all(
            f'tok_model_output_len_{i}' in df.columns for i in range(pass_k))

    if not has_tok_len:
        raise ValueError("Missing required tok_model_output_len column(s)")

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
    """Parse code from ```python or plain ``` code block.

    Priority:
    1. Last ```python block
    2. Last plain ``` block (if it looks like Python code)
    """
    text = validate_text_input(text)
    if not text:
        return None

    # Try ```python blocks first (most specific)
    python_matches = list(re.finditer(r"```python(.*?)```", text, re.DOTALL))
    if python_matches:
        return python_matches[-1].group(1).strip()

    # Fall back to plain ``` blocks
    plain_matches = list(re.finditer(r"```(.*?)```", text, re.DOTALL))
    if plain_matches:
        # Get the last match
        code = plain_matches[-1].group(1).strip()
        # Remove language tag if present (e.g., ```python\n or ```py\n)
        code = re.sub(r'^(?:python|py)\s*\n', '', code, flags=re.IGNORECASE)
        return code

    return None


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
            scenario=Scenario.codegeneration, release_version="release_v6",
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
    """Evaluate LiveCodeBench code generation.

    Returns:
        bool: True if all tests passed, False otherwise
    """
    result, _ = evaluate_livecodebench_detailed(code, question_id)
    return result


def evaluate_livecodebench_detailed(
        code: Optional[str], question_id: str) -> Tuple[bool, str]:
    """Evaluate LiveCodeBench code generation with detailed results.

    Returns:
        Tuple[bool, str]: (passed, detailed_reason)
            - passed: True if all tests passed, False otherwise
            - detailed_reason: Description of test results or error
    """
    if not code or not question_id:
        return False, "No code or question_id provided"

    lcb_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "submodules", "LiveCodeBench"))

    try:
        benchmark_map = load_lcb_benchmark()
    except Exception as e:
        return False, f"Failed to load benchmark: {type(e).__name__}: {e}"

    instance = benchmark_map.get(question_id)
    if not instance:
        return False, f"Question ID '{question_id}' not found in benchmark"

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
            scenario=Scenario.codegeneration, release_version="release_v6",
            subset="code_generation", language="python", not_fast=False,
            start_date=None, end_date=None, k=[1], num_samples=1,
            timeout=20, num_workers=1, num_process_evaluate=1,
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
        passed = graded and graded[0] and graded[0][0]

        # Try to extract detailed results
        detailed_reason = ""
        try:
            if combined_results and len(combined_results) > 0:
                result_info = combined_results[0]
                if hasattr(result_info, 'result') and result_info.result:
                    # Extract test results
                    test_results = result_info.result
                    if isinstance(test_results, dict):
                        detailed_reason = f"Test results: {test_results}"
                    elif isinstance(test_results, list):
                        num_passed = sum(1 for r in test_results if r)
                        num_total = len(test_results)
                        detailed_reason = f"Passed {num_passed}/{num_total} test cases"
                    else:
                        detailed_reason = f"Result: {test_results}"
                elif hasattr(result_info, 'status'):
                    detailed_reason = f"Status: {result_info.status}"
        except Exception:
            pass

        if not detailed_reason:
            if passed:
                detailed_reason = "All tests passed"
            else:
                detailed_reason = "Failed one or more test cases"

        return passed, detailed_reason

    except Exception as e:
        return False, f"Evaluation error: {type(e).__name__}: {str(e)[:200]}"
    finally:
        os.chdir(original_cwd)
        shutil.rmtree(temp_dir, ignore_errors=True)
        os.environ.pop('TQDM_DISABLE', None)


def evaluate_livecodebench_worker(
        args: Tuple[str, str]) -> Tuple[str, bool, str]:
    """Worker function for parallel LiveCodeBench evaluation.

    Returns:
        Tuple[str, bool, str]: (question_id, passed, detailed_reason)
    """
    code, question_id = args

    try:
        passed, reason = evaluate_livecodebench_detailed(code, question_id)
        return question_id, passed, reason
    except Exception as e:
        error_msg = f"Error evaluating {question_id}: {type(e).__name__}: {e}"
        logger.warning(error_msg)
        return question_id, False, error_msg


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
        logger.warning(
            f"Raw LLM response (first 500 chars): {json_string[:500]}")
        logger.warning(
            f"Cleaned response (first 500 chars): {json_cleaned[:500]}")
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
    grader_backend: str = "openai",
    max_workers: Optional[int] = None
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
        max_workers: Max concurrent requests for rubric grading (default: all rubrics in parallel)

    Returns:
        Tuple of (score, detailed_explanation)
    """
    # Use API key from environment if not provided
    if grader_api_key is None:
        if grader_backend == "nvidia":
            grader_api_key = os.environ.get("NVIDIA_NIM_API_KEY")
            # Check if it's an official NVIDIA URL that requires a key
            if not grader_api_key and "nvidia.com" in grader_base_url.lower():
                logger.warning(
                    "No NVIDIA NIM API key found. Set NVIDIA_NIM_API_KEY environment variable.")
                return 0.0, "Error: No NVIDIA NIM API key provided"
        else:
            grader_api_key = os.environ.get("OPENAI_API_KEY")
            # Check if it's an official OpenAI URL that requires a key
            if not grader_api_key and "api.openai.com" in grader_base_url.lower():
                logger.warning(
                    "No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
                return 0.0, "Error: No OpenAI API key provided"

        # For local servers, use a dummy key if none provided
        if grader_api_key is None:
            grader_api_key = "dummy-key-for-local-server"
            logger.info(
                f"Using local server at {grader_base_url}, no API key required")

    # Format conversation
    conversation_text = ""
    for msg in prompt_messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        conversation_text += f"{role}: {content}\n\n"
    conversation_text += f"assistant: {model_output}"

    # Prepare all grading prompts
    grading_tasks = []
    for rubric_item in rubric_items:
        grading_prompt = HEALTHBENCH_GRADER_TEMPLATE.replace(
            "<CONVERSATION>", conversation_text
        ).replace("<RUBRIC_ITEM>", str(rubric_item))
        grading_tasks.append((rubric_item, grading_prompt))

    # Submit all requests concurrently for server-side batching
    grading_responses = []

    def _grade_single_rubric(task_data):
        """Helper to grade a single rubric item."""
        rubric_item, grading_prompt = task_data
        try:
            if grader_backend == "nvidia":
                response_text = _call_nvidia_nim_api(
                    api_key=grader_api_key,
                    model=grader_model,
                    messages=[{"role": "user", "content": grading_prompt}],
                    base_url=grader_base_url,
                    temperature=0.0,
                    max_tokens=1024
                )
            else:
                response_text = _call_openai_api(
                    api_key=grader_api_key,
                    model=grader_model,
                    messages=[{"role": "user", "content": grading_prompt}],
                    base_url=grader_base_url,
                    temperature=0.0,
                    max_tokens=1024
                )
            return parse_healthbench_json(response_text)
        except Exception as e:
            logger.warning(f"Error grading rubric item: {e}")
            return {
                "explanation": f"Error during grading: {e}",
                "criteria_met": False
            }

    # Use ThreadPoolExecutor to send all requests concurrently
    # The server can batch these together for efficient processing
    # Default to sending all rubric items in parallel if max_workers not
    # specified
    num_workers = max_workers if max_workers is not None else len(
        grading_tasks)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        grading_responses = list(
            executor.map(
                _grade_single_rubric,
                grading_tasks))

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
    grader_base_url: Optional[str] = None,
    grader_model: Optional[str] = None,
    max_workers: Optional[int] = None
) -> Tuple[float, Optional[str]]:
    """Evaluate HealthBench response using LLM grading.

    Args:
        parsed_output: The model output text
        row_data: Full row data containing 'rubrics' and 'prompt'
        grader_api_key: Optional API key for grader
        grader_base_url: Base URL for API (default: OpenAI API)
        grader_model: Optional model name override
        max_workers: Max concurrent requests for rubric grading

    Returns:
        Tuple of (score, detailed_explanation) where score is 0.0-1.0
    """
    if not parsed_output:
        return 0.0, "Empty output"

    # Extract rubrics from row
    rubrics = row_data.get('rubrics', [])
    if not rubrics:
        logger.warning("No rubrics found in row data")
        return 0.0, "No rubrics available"

    # Convert to RubricItem objects
    rubric_items = [RubricItem.from_dict(r) for r in rubrics]

    # Extract prompt/conversation
    prompt = row_data.get('prompt', [])
    if isinstance(prompt, str):
        # If prompt is a string, convert to message format
        prompt = [{"role": "user", "content": prompt}]

    # Set default base URL if not provided
    if grader_base_url is None:
        grader_base_url = "https://api.openai.com/v1"

    # Auto-detect backend based on URL
    if "nvidia.com" in grader_base_url.lower():
        grader_backend = "nvidia"
        # Set default model for NVIDIA if not specified
        if grader_model is None:
            grader_model = "deepseek-ai/deepseek-v3.1-terminus"
    else:
        grader_backend = "openai"
        # Set default model for OpenAI if not specified
        if grader_model is None:
            grader_model = "gpt-4o-mini"

    # Grade using LLM
    score, explanation = grade_healthbench_with_llm(
        prompt_messages=prompt,
        model_output=parsed_output,
        rubric_items=rubric_items,
        grader_api_key=grader_api_key,
        grader_model=grader_model,
        grader_base_url=grader_base_url,
        grader_backend=grader_backend,
        max_workers=max_workers
    )

    # Return the score (0.0 to 1.0) and detailed explanation
    # Note: score is returned as-is, not converted to binary pass/fail
    return score, f"Score: {score:.2%}\n\n{explanation}"


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
        df: pd.DataFrame,
        group_indices: pd.Index,
        extracted_answer_col: str = 'extracted_answer',
        prompt_accuracy_col: str = 'prompt_accuracy',
        evaluation_details_col: str = 'evaluation_details',
        pass_label: str = '',
        executor: Optional[ProcessPoolExecutor] = None) -> Tuple[int, int]:
    """Process LiveCodeBench items in parallel.

    Args:
        df: DataFrame with data
        group_indices: Indices to process
        extracted_answer_col: Column name for extracted answers
        prompt_accuracy_col: Column name for accuracy results
        evaluation_details_col: Column name for evaluation details
        pass_label: Label for logging (e.g., 'pass 0', 'pass 1')
        executor: Optional ProcessPoolExecutor to reuse (for performance)

    Returns:
        Tuple of (correct_count, total_evaluated)
    """
    # Prepare work items
    work_items = []
    for idx in group_indices:
        row = df.loc[idx]
        extracted = row.get(extracted_answer_col)
        ground_truth = row.get('ground_truth')

        if extracted is not None and not pd.isna(ground_truth):
            work_items.append((idx, extracted, ground_truth))

    if not work_items:
        return 0, 0

    # Ensure evaluation_details column exists
    if evaluation_details_col not in df.columns:
        df[evaluation_details_col] = None

    # Process in parallel
    max_workers = min(multiprocessing.cpu_count(), len(work_items), 64)
    desc = f"Evaluating LiveCodeBench{' ' + pass_label if pass_label else ''}"
    logger.info(
        f"Evaluating {len(work_items)} LiveCodeBench items{' ' + pass_label if pass_label else ''} with {max_workers} workers")

    correct_count = 0
    total_evaluated = 0

    # Determine whether to create new executor or reuse existing one
    should_close_executor = False
    if executor is None:
        executor = ProcessPoolExecutor(max_workers=max_workers)
        should_close_executor = True

    try:
        future_to_idx = {
            executor.submit(evaluate_livecodebench_worker, (code, question_id)): idx
            for idx, code, question_id in work_items
        }

        for future in tqdm(as_completed(future_to_idx, timeout=1200),
                           total=len(future_to_idx), desc=desc):
            idx = future_to_idx[future]

            try:
                question_id, is_correct, detailed_reason = future.result(
                    timeout=25)
                df.at[idx, prompt_accuracy_col] = 100.0 if is_correct else 0.0
                df.at[idx, evaluation_details_col] = detailed_reason
                total_evaluated += 1
                if is_correct:
                    correct_count += 1
            except TimeoutError as e:
                logger.warning(
                    f"Timeout evaluating row {idx} (question_id: {df.at[idx, 'ground_truth'] if 'ground_truth' in df.columns else 'unknown'}){' ' + pass_label if pass_label else ''}: Test execution exceeded 25s timeout")
                df.at[idx, prompt_accuracy_col] = 0.0
                df.at[idx, evaluation_details_col] = "Timeout: Test execution exceeded time limit"
                total_evaluated += 1
            except Exception as e:
                logger.error(
                    f"Error evaluating row {idx}{' ' + pass_label if pass_label else ''}: {e}")
                df.at[idx, prompt_accuracy_col] = 0.0
                df.at[idx, evaluation_details_col] = f"Error: {e}"
                total_evaluated += 1
    finally:
        # Only close if we created it
        if should_close_executor:
            executor.shutdown(wait=True)

    return correct_count, total_evaluated


def evaluate_healthbench_batch(
    df: pd.DataFrame,
    group_indices: pd.Index,
    grader_api_key: Optional[str] = None,
    grader_base_url: Optional[str] = None,
    grader_model: Optional[str] = None,
    max_workers: Optional[int] = None,
    extracted_answer_col: str = 'extracted_answer',
    pass_label: str = ''
) -> Dict[int, Tuple[float, str]]:
    """Evaluate all HealthBench rows with batched rubric grading across all rows.

    Args:
        df: DataFrame containing the data
        group_indices: Indices of rows to evaluate
        grader_api_key: Optional API key for grader
        grader_base_url: Base URL for API
        grader_model: Model name
        max_workers: Max concurrent requests
        extracted_answer_col: Column name for extracted answers (e.g., 'extracted_answer_0')
        pass_label: Label for logging (e.g., 'pass 0')

    Returns:
        Dictionary mapping row index to (score, explanation) tuple
    """
    # Set default base URL if not provided
    if grader_base_url is None:
        grader_base_url = "https://api.openai.com/v1"

    # Auto-detect backend based on URL
    if "nvidia.com" in grader_base_url.lower():
        grader_backend = "nvidia"
        if grader_model is None:
            grader_model = "deepseek-ai/deepseek-v3.1-terminus"
    else:
        grader_backend = "openai"
        if grader_model is None:
            grader_model = "gpt-4o-mini"

    # Handle API key
    if grader_api_key is None:
        if grader_backend == "nvidia":
            grader_api_key = os.environ.get("NVIDIA_NIM_API_KEY")
            if not grader_api_key and "nvidia.com" in grader_base_url.lower():
                logger.warning(
                    "No NVIDIA NIM API key found. Set NVIDIA_NIM_API_KEY environment variable.")
                return {idx: (0.0, "Error: No NVIDIA NIM API key provided")
                        for idx in group_indices}
        else:
            grader_api_key = os.environ.get("OPENAI_API_KEY")
            if not grader_api_key and "api.openai.com" in grader_base_url.lower():
                logger.warning(
                    "No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
                return {idx: (0.0, "Error: No OpenAI API key provided")
                        for idx in group_indices}

        if grader_api_key is None:
            grader_api_key = "dummy-key-for-local-server"
            logger.info(
                f"Using local server at {grader_base_url}, no API key required")

    # Prepare all grading tasks for all rows
    all_tasks = []
    row_rubric_map = {}  # Maps task_id to (row_idx, rubric_idx)
    task_id = 0

    for idx in group_indices:
        row = df.loc[idx]
        extracted = row.get(extracted_answer_col)

        if extracted is None or pd.isna(extracted):
            row_rubric_map[f"row_{idx}_skip"] = (idx, None)
            continue

        # Extract rubrics and prompt
        rubrics = row.get('rubrics', [])
        if not rubrics:
            logger.warning(f"No rubrics found for row {idx}")
            row_rubric_map[f"row_{idx}_skip"] = (idx, None)
            continue

        rubric_items = [RubricItem.from_dict(r) for r in rubrics]
        prompt = row.get('prompt', [])
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]

        # Format conversation
        conversation_text = ""
        for msg in prompt:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            conversation_text += f"{role}: {content}\n\n"
        conversation_text += f"assistant: {extracted}"

        # Create grading tasks for all rubrics in this row
        for rubric_idx, rubric_item in enumerate(rubric_items):
            grading_prompt = HEALTHBENCH_GRADER_TEMPLATE.replace(
                "<CONVERSATION>", conversation_text
            ).replace("<RUBRIC_ITEM>", str(rubric_item))

            all_tasks.append({
                'task_id': task_id,
                'prompt': grading_prompt,
                'backend': grader_backend
            })
            row_rubric_map[task_id] = (idx, rubric_idx, rubric_item)
            task_id += 1

    if not all_tasks:
        logger.warning(
            f"No grading tasks to process{' for ' + pass_label if pass_label else ''}")
        return {}

    logger.info(
        f"Batching {len(all_tasks)} rubric grading requests{' for ' + pass_label if pass_label else ''} across {len(group_indices)} rows")

    # Define grading function
    def _grade_single_task(task):
        """Grade a single rubric item."""
        try:
            if task['backend'] == "nvidia":
                response_text = _call_nvidia_nim_api(
                    api_key=grader_api_key,
                    model=grader_model,
                    messages=[{"role": "user", "content": task['prompt']}],
                    base_url=grader_base_url,
                    temperature=0.0,
                    max_tokens=1024
                )
            else:
                response_text = _call_openai_api(
                    api_key=grader_api_key,
                    model=grader_model,
                    messages=[{"role": "user", "content": task['prompt']}],
                    base_url=grader_base_url,
                    temperature=0.0,
                    max_tokens=1024
                )
            return task['task_id'], parse_healthbench_json(response_text)
        except Exception as e:
            logger.warning(f"Error grading task {task['task_id']}: {e}")
            return task['task_id'], {
                "explanation": f"Error during grading: {e}",
                "criteria_met": False
            }

    # Send all requests concurrently for server-side batching
    num_workers = max_workers if max_workers is not None else len(all_tasks)
    grading_results = {}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                _grade_single_task,
                task): task['task_id'] for task in all_tasks}

        desc = f"Grading HealthBench{' ' + pass_label if pass_label else ''} (batched)"
        for future in tqdm(as_completed(futures), total=len(
                futures), desc=desc):
            try:
                task_id, result = future.result(timeout=60)
                grading_results[task_id] = result
            except Exception as e:
                task_id = futures[future]
                logger.error(f"Error processing task {task_id}: {e}")
                grading_results[task_id] = {
                    "explanation": f"Error during grading: {e}",
                    "criteria_met": False
                }

    # Reconstruct results per row
    row_results = {}
    # Group results by row: {row_idx: {rubric_idx: (rubric_item,
    # grading_result)}}
    rows_rubrics = {}

    for task_id, grading_result in grading_results.items():
        if task_id not in row_rubric_map:
            continue

        row_idx, rubric_idx, rubric_item = row_rubric_map[task_id]

        if row_idx not in rows_rubrics:
            rows_rubrics[row_idx] = {}

        rows_rubrics[row_idx][rubric_idx] = (rubric_item, grading_result)

    # Calculate scores for each row
    for row_idx, rubric_data in rows_rubrics.items():
        # Sort by rubric_idx to maintain correct order
        sorted_rubrics = sorted(rubric_data.items(), key=lambda x: x[0])
        rubric_items = [item for _, (item, _) in sorted_rubrics]
        grading_responses = [response for _, (_, response) in sorted_rubrics]

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

        detailed_explanation = f"Score: {score:.2%}\n\n" + \
            "\n\n".join(explanations)
        row_results[row_idx] = (score, detailed_explanation)

    # Handle skipped rows
    for key, value in row_rubric_map.items():
        if isinstance(key, str) and key.startswith(
                "row_") and key.endswith("_skip"):
            row_idx = value[0]
            if row_idx not in row_results:
                row_results[row_idx] = (0.0, "Empty output or no rubrics")

    return row_results


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Process entire dataframe with optimized batch processing.

    Supports both single-pass and pass@k formats:
    - Single-pass: model_output -> extracted_answer, prompt_accuracy
    - Pass@k: model_output_0, model_output_1, ... -> extracted_answer_0, prompt_accuracy_0, ...
              and aggregated prompt_accuracy = max(prompt_accuracy_0, prompt_accuracy_1, ...)
    """
    validate_dataframe(df)

    df_output = df.copy()

    # Detect pass@k
    pass_k = detect_pass_k(df)
    logger.info(f"Detected pass@k format with k={pass_k}")

    # Initialize columns for each pass
    for pass_num in range(pass_k):
        suffix = f'_{pass_num}' if pass_k > 1 else ''
        df_output[f'extracted_answer{suffix}'] = None
        df_output[f'prompt_accuracy{suffix}'] = 0.0
        df_output[f'evaluation_details{suffix}'] = None

    # Add aggregated columns for pass@k
    if pass_k > 1:
        df_output['prompt_accuracy'] = 0.0  # Will be max of all passes
        df_output['evaluation_details'] = None  # Will aggregate details

    # Check if we have LiveCodeBench datasets to evaluate
    has_livecodebench = any('livecodebench' in str(ds).lower()
                            for ds in df_output['dataset'].unique())

    # Pre-load LiveCodeBench benchmark and create shared process pool for all
    # LCB evaluations
    lcb_executor = None
    if has_livecodebench:
        try:
            logger.info(
                "Pre-loading LiveCodeBench benchmark for worker processes...")
            # Load benchmark in main process before forking - workers will
            # inherit via copy-on-write
            _ = load_lcb_benchmark()
            logger.info("LiveCodeBench benchmark loaded successfully")

            # Create a single process pool for all LCB evaluations
            max_workers = multiprocessing.cpu_count()
            lcb_executor = ProcessPoolExecutor(max_workers=max_workers)
            logger.info(
                f"Created shared ProcessPoolExecutor with {max_workers} workers for LiveCodeBench")
        except Exception as e:
            logger.warning(f"Failed to pre-load LiveCodeBench benchmark: {e}")
            logger.warning("Will fall back to per-evaluation loading")

    try:
        # Process by dataset
        for dataset_name, group_indices in tqdm(df_output.groupby('dataset').groups.items(),
                                                desc="Processing datasets"):
            evaluator = get_evaluator(dataset_name)

            # For LiveCodeBench with multiple passes, collect all work upfront
            # to maximize parallelism
            is_livecodebench = 'livecodebench' in dataset_name.lower()
            if is_livecodebench and pass_k > 1:
                # Parse all passes first
                logger.info(
                    f"Parsing {len(group_indices)} rows for dataset '{dataset_name}' across {pass_k} passes")
                for pass_num in range(pass_k):
                    suffix = f'_{pass_num}'
                    model_output_col = f'model_output{suffix}'
                    extracted_answer_col = f'extracted_answer{suffix}'
                    evaluation_details_col = f'evaluation_details{suffix}'

                    for idx in group_indices:
                        row = df_output.loc[idx]
                        raw_output = validate_text_input(row[model_output_col])
                        extracted = evaluator['parse'](raw_output)
                        df_output.at[idx, extracted_answer_col] = extracted

                        if extracted is None or pd.isna(extracted):
                            df_output.at[idx,
                                         evaluation_details_col] = "No answer extracted from model output"

                # Collect all work items from all passes
                all_work_items = []
                work_item_metadata = []  # (idx, pass_num)
                for pass_num in range(pass_k):
                    extracted_answer_col = f'extracted_answer_{pass_num}'
                    for idx in group_indices:
                        row = df_output.loc[idx]
                        extracted = row.get(extracted_answer_col)
                        ground_truth = row.get('ground_truth')

                        if extracted is not None and not pd.isna(ground_truth):
                            all_work_items.append((extracted, ground_truth))
                            work_item_metadata.append((idx, pass_num))

                if all_work_items:
                    # Submit all work at once for maximum parallelism
                    max_workers = min(multiprocessing.cpu_count(), len(all_work_items), 64)
                    logger.info(
                        f"Evaluating {len(all_work_items)} LiveCodeBench items across {pass_k} passes with {max_workers} workers")

                    future_to_metadata = {
                        lcb_executor.submit(evaluate_livecodebench_worker, work_item): metadata
                        for work_item, metadata in zip(all_work_items, work_item_metadata)
                    }

                    # Collect results and assign to appropriate pass columns
                    pass_results = {i: {'correct': 0, 'total': 0} for i in range(pass_k)}
                    
                    for future in tqdm(as_completed(future_to_metadata, timeout=1200),
                                       total=len(future_to_metadata), 
                                       desc=f"Evaluating LiveCodeBench (all passes)"):
                        idx, pass_num = future_to_metadata[future]
                        prompt_accuracy_col = f'prompt_accuracy_{pass_num}'
                        evaluation_details_col = f'evaluation_details_{pass_num}'

                        try:
                            question_id, is_correct, detailed_reason = future.result(timeout=25)
                            df_output.at[idx, prompt_accuracy_col] = 100.0 if is_correct else 0.0
                            df_output.at[idx, evaluation_details_col] = detailed_reason
                            pass_results[pass_num]['total'] += 1
                            if is_correct:
                                pass_results[pass_num]['correct'] += 1
                        except TimeoutError:
                            logger.warning(
                                f"Timeout evaluating row {idx} pass {pass_num}: Test execution exceeded 25s timeout")
                            df_output.at[idx, prompt_accuracy_col] = 0.0
                            df_output.at[idx, evaluation_details_col] = "Timeout: Test execution exceeded time limit"
                            pass_results[pass_num]['total'] += 1
                        except Exception as e:
                            logger.error(
                                f"Error evaluating row {idx} pass {pass_num}: {e}")
                            df_output.at[idx, prompt_accuracy_col] = 0.0
                            df_output.at[idx, evaluation_details_col] = f"Error: {e}"
                            pass_results[pass_num]['total'] += 1

                    # Log results for each pass
                    for pass_num in range(pass_k):
                        if pass_results[pass_num]['total'] > 0:
                            correct = pass_results[pass_num]['correct']
                            total = pass_results[pass_num]['total']
                            accuracy = correct / total * 100
                            logger.info(
                                f"{dataset_name} pass {pass_num} results: {correct}/{total} correct ({accuracy:.1f}% accuracy)")

            else:
                # Original sequential pass processing for non-LCB or single-pass LCB
                for pass_num in range(pass_k):
                    suffix = f'_{pass_num}' if pass_k > 1 else ''
                    model_output_col = f'model_output{suffix}'
                    extracted_answer_col = f'extracted_answer{suffix}'
                    prompt_accuracy_col = f'prompt_accuracy{suffix}'
                    evaluation_details_col = f'evaluation_details{suffix}'

                    logger.info(
                        f"Processing {len(group_indices)} rows for dataset '{dataset_name}', pass {pass_num}")

                    # Parse answers for all rows in this dataset for this pass
                    for idx in group_indices:
                        row = df_output.loc[idx]
                        raw_output = validate_text_input(row[model_output_col])
                        extracted = evaluator['parse'](raw_output)
                        df_output.at[idx, extracted_answer_col] = extracted

                        # Set initial evaluation details for rows without extracted
                        # answers
                        if extracted is None or pd.isna(extracted):
                            df_output.at[idx,
                                         evaluation_details_col] = "No answer extracted from model output"

                    # Evaluate answers for this pass
                    pass_label_str = f'(pass {pass_num})' if pass_k > 1 else ''

                    if is_livecodebench:
                        # Single-pass LCB evaluation
                        correct_count, total_evaluated = process_livecodebench_parallel(
                            df_output,
                            group_indices,
                            extracted_answer_col=extracted_answer_col,
                            prompt_accuracy_col=prompt_accuracy_col,
                            evaluation_details_col=evaluation_details_col,
                            pass_label=pass_label_str,
                            executor=lcb_executor  # Reuse shared executor
                        )
                    elif 'healthbench' in dataset_name.lower():
                        # HealthBench evaluation with LLM grading - batched across
                        # all rows
                        total_score = 0.0
                        total_evaluated = 0

                        # Process all rows with batched grading for this pass
                        results = evaluate_healthbench_batch(
                            df_output,
                            group_indices,
                            grader_api_key=LLM_JUDGE_API_KEY,
                            grader_base_url=LLM_JUDGE_BASE_URL,
                            grader_model=LLM_JUDGE_MODEL,
                            max_workers=LLM_JUDGE_MAX_WORKERS,
                            extracted_answer_col=extracted_answer_col,
                            pass_label=pass_label_str
                        )

                        # Store results for this pass
                        for idx, (score, explanation) in results.items():
                            # Store score as percentage (0-100)
                            df_output.at[idx, prompt_accuracy_col] = score * 100.0
                            df_output.at[idx, evaluation_details_col] = explanation
                            total_evaluated += 1
                            total_score += score
                    else:
                        # Sequential evaluation for other datasets
                        correct_count = 0
                        total_evaluated = 0

                        for idx in group_indices:
                            row = df_output.loc[idx]
                            extracted = row[extracted_answer_col]
                            ground_truth = row.get('ground_truth')

                            if extracted is not None and not pd.isna(ground_truth):
                                is_correct = evaluator['evaluate'](
                                    extracted, ground_truth)
                                df_output.at[idx,
                                             prompt_accuracy_col] = 100.0 if is_correct else 0.0
                                total_evaluated += 1
                                if is_correct:
                                    correct_count += 1

                    # Log results for this pass
                    if total_evaluated > 0:
                        if 'healthbench' in dataset_name.lower():
                            # For HealthBench, report average score
                            avg_score = total_score / total_evaluated * 100
                            logger.info(
                                f"{dataset_name} pass {pass_num} results: Average score {avg_score:.1f}% ({total_evaluated} samples)")
                        else:
                            # For other datasets, report accuracy
                            accuracy = correct_count / total_evaluated * 100
                            logger.info(
                                f"{dataset_name} pass {pass_num} results: {correct_count}/{total_evaluated} correct ({accuracy:.1f}% accuracy)")

            # Aggregate results across all passes (take max)
            if pass_k > 1:
                logger.info(
                    f"Aggregating results across {pass_k} passes for dataset '{dataset_name}'")
                for idx in group_indices:
                    # Get all accuracy values for this row
                    accuracies = []
                    for pass_num in range(pass_k):
                        acc = df_output.at[idx, f'prompt_accuracy_{pass_num}']
                        accuracies.append(acc if not pd.isna(acc) else 0.0)

                    # Set aggregated accuracy as max
                    max_accuracy = max(accuracies)
                    df_output.at[idx, 'prompt_accuracy'] = max_accuracy

                    # Find which pass achieved max accuracy
                    max_pass = accuracies.index(max_accuracy)
                    df_output.at[idx,
                                 'evaluation_details'] = f"Best pass: {max_pass} (accuracy: {max_accuracy:.1f}%)"

        return df_output
    finally:
        # Clean up shared LiveCodeBench executor
        if lcb_executor is not None:
            logger.info(
                "Shutting down shared LiveCodeBench ProcessPoolExecutor")
            lcb_executor.shutdown(wait=True)


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

    # Detect pass@k
    pass_k = detect_pass_k(df_evaluated)

    # Calculate statistics
    if pass_k > 1:
        # For pass@k, use the aggregated prompt_accuracy (max across passes)
        # Count from first pass
        evaluated = df_evaluated['extracted_answer_0'].notna().sum()
        correct = (df_evaluated['prompt_accuracy'] > 0).sum()
        accuracy = df_evaluated['prompt_accuracy'].mean()

        # Calculate average token length across all passes
        all_output_lens = []
        for i in range(pass_k):
            all_output_lens.extend(
                df_evaluated[f'tok_model_output_len_{i}'].tolist())
        mean_output_len = float(
            sum(all_output_lens) /
            len(all_output_lens)) if all_output_lens else 0.0
    else:
        # Single pass format
        suffix = '' if 'extracted_answer' in df_evaluated.columns else '_0'
        evaluated = df_evaluated[f'extracted_answer{suffix}'].notna().sum()
        correct = (df_evaluated[f'prompt_accuracy{suffix}'] > 0).sum()
        accuracy = df_evaluated[f'prompt_accuracy{suffix}'].mean()

        # tok_model_output_len is now a required column
        tok_len_col = 'tok_model_output_len' if 'tok_model_output_len' in df_evaluated.columns else 'tok_model_output_len_0'
        mean_output_len = float(df_evaluated[tok_len_col].mean())

    # Check if this is HealthBench dataset
    is_healthbench = False
    if 'dataset' in df_evaluated.columns:
        datasets = df_evaluated['dataset'].unique()
        is_healthbench = any('healthbench' in str(ds).lower()
                             for ds in datasets)

    # Use appropriate metric name
    if is_healthbench:
        metric_key = 'healthbench_score'
    else:
        metric_key = 'exact_match'

    results = {
        # 'evaluated': int(evaluated),
        # 'correct': int(correct),
        metric_key: float(accuracy),
        'tokens_per_sample': mean_output_len,
        'num-samples': len(df_evaluated),
    }

    if pass_k > 1:
        results['pass_k'] = pass_k
        # Also report individual pass accuracies
        for i in range(pass_k):
            pass_acc = df_evaluated[f'prompt_accuracy_{i}'].mean()
            results[f'{metric_key}_pass_{i}'] = float(pass_acc)

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
    parser.add_argument("--llm-judge-base-url",
                        help="Base URL for HealthBench LLM judge API (default: https://api.openai.com/v1). "
                             "For local servers like SGLang, use http://localhost:8000/v1")
    parser.add_argument("--llm-judge",
                        help="Model for HealthBench LLM judge (default: gpt-4o-mini for OpenAI-compatible APIs, "
                             "deepseek-ai/deepseek-v3.1-terminus for NVIDIA)")
    parser.add_argument("--llm-judge-api-key",
                        help="API key for HealthBench LLM judge (default: read from OPENAI_API_KEY or NVIDIA_NIM_API_KEY env var). "
                             "Not required for local servers.")
    parser.add_argument("--llm-judge-max-workers", type=int,
                        help="Max concurrent requests per row for HealthBench rubric grading (default: all rubrics in parallel). "
                             "Useful for rate limiting or controlling server load.")

    args = parser.parse_args()

    # Set global configuration for HealthBench LLM judge
    global LLM_JUDGE_BASE_URL, LLM_JUDGE_MODEL, LLM_JUDGE_API_KEY, LLM_JUDGE_MAX_WORKERS
    LLM_JUDGE_BASE_URL = args.llm_judge_base_url
    LLM_JUDGE_MODEL = args.llm_judge
    LLM_JUDGE_API_KEY = args.llm_judge_api_key
    LLM_JUDGE_MAX_WORKERS = args.llm_judge_max_workers

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
