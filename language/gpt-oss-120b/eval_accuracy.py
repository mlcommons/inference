#!/usr/bin/env python3
"""
Standalone evaluation script for mlperf-inference deepseek-r1 dataset.

Expected input format (pickle file with DataFrame):
- model_output: The model's response text
- tok_model_output_len: The length of the model's response tokens
- ground_truth: The expected answer
- dataset: Dataset name (e.g., 'gpqa', 'mmlu_pro', 'math500', 'livecodebench', 'aime')
- question: The question text

Output adds columns:
- extracted_answer: Parsed answer from model output
- prompt_accuracy: 100.0 if correct, 0.0 if incorrect
- evaluation_details: Detailed evaluation explanation
"""

import sys
import os
import argparse
import logging
import pickle
import re
import shutil
import time
from functools import lru_cache
from typing import Dict, Any, Optional, Tuple, Union
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import multiprocessing
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

# MLPerf log processing imports
import numpy as np
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
# Harmony Format Extraction
# =============================================================================


def extract_final_section(text: str) -> str:
    """Extract content from the <|channel|>final<|message|>...<|return|> section.

    The model outputs have two sections:
    - <|channel|>analysis<|message|>... (reasoning, may have draft answers)
    - <|channel|>final<|message|>... (actual final answer)

    This function extracts only the final section to avoid extracting
    wrong answers from the analysis section.

    Uses a flexible regex to handle corrupted markers like:
    - <|channel|>final 明<|message|>
    - <|channel|>final537<|message|>

    Args:
        text: Full model output text

    Returns:
        Content of final section if found, otherwise returns original text
    """
    text = validate_text_input(text)
    if not text:
        return ""

    # Flexible pattern to handle corrupted markers (allows chars between final
    # and <|message|>)
    match = re.search(
        r'<\|channel\|>final[^<]*<\|message\|>(.*?)(?:<\|return\|>|$)',
        text, re.DOTALL
    )
    if match:
        return match.group(1).strip()

    # Fallback: return original text if no final section found
    return text


def strip_markdown_bold(text: str) -> str:
    """Remove markdown bold formatting (**text**) from text.

    Args:
        text: Text that may contain **bold** formatting

    Returns:
        Text with bold markers removed
    """
    return re.sub(r'\*\*([^*]+)\*\*', r'\1', text)


# =============================================================================
# Answer Parsing Functions
# =============================================================================

def parse_multiple_choice(text: str, max_option: str = 'D') -> Optional[str]:
    """Parse multiple choice answer (A-D or A-J).

    First extracts the final section from harmony-formatted outputs,
    then parses the answer from that section only.
    """
    text = validate_text_input(text)
    if not text:
        return None

    # Extract final section first (for harmony format)
    final_section = extract_final_section(text)

    # Strip markdown bold formatting (**A** -> A)
    final_section = strip_markdown_bold(final_section)

    # Clean artifacts
    if final_section.startswith(
            ("['", '["')) and final_section.endswith(("']", '"]')):
        final_section = final_section[2:-2].strip()

    final_section = final_section.replace(r'\n', '\n').replace(r'\'', "'")

    # Try to extract from final section first
    # Priority 1: Single letter answer at start of final section (common in
    # harmony format)
    single_letter_match = re.match(
        rf'^[^a-zA-Z]*([A-{max_option}])(?:[^a-zA-Z]|$)',
        final_section.strip(), re.IGNORECASE
    )
    if single_letter_match:
        return single_letter_match.group(1).upper()

    # Priority 2: "Answer: X" pattern in final section
    answer_pattern = rf'\b(?:Answer|ANSWER)\s*[:.]?\s*([A-{max_option}])\b'
    answer_match = re.search(answer_pattern, final_section, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()

    # Priority 3: Fall back to ANSWER/FINAL ANSWER pattern in full text
    # (for backwards compatibility with non-harmony outputs)
    full_text = text.replace(r'\n', '\n').replace(r'\'', "'")
    pattern = rf"\b(?:ANSWER|FINAL\s*ANSWER)\b\s*[:=]?\s*(?:\(?\s*([A-{max_option}])\s*\)?)(?:\s*$|[^A-Za-z])"
    matches = list(re.finditer(pattern, full_text, re.IGNORECASE))

    if matches:
        return matches[-1].group(1).upper()

    # MMLU-Pro fallback: standalone letter in final section
    if max_option == 'J':
        fallback_matches = list(re.finditer(
            rf"\b([A-{max_option}])\b", final_section, re.IGNORECASE))
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

    First extracts the final section from harmony-formatted outputs,
    then parses code from that section only. This avoids extracting
    malformed code blocks from the analysis section.

    Priority:
    1. Code from final section (if harmony format detected)
    2. Last ```python block from full text (fallback)
    3. Last plain ``` block from full text (fallback)
    """
    text = validate_text_input(text)
    if not text:
        return None

    # First try to extract from final section (for harmony format)
    final_section = extract_final_section(text)

    # Check if we got a different final section (harmony format detected)
    if final_section != text:
        # Parse code from final section only
        python_matches = list(
            re.finditer(
                r"```python(.*?)```",
                final_section,
                re.DOTALL))
        if python_matches:
            return python_matches[-1].group(1).strip()

        plain_matches = list(
            re.finditer(
                r"```(.*?)```",
                final_section,
                re.DOTALL))
        if plain_matches:
            code = plain_matches[-1].group(1).strip()
            code = re.sub(
                r'^(?:python|py)\s*\n',
                '',
                code,
                flags=re.IGNORECASE)
            return code

    # Fallback: search full text (for non-harmony outputs or if final section has no code)
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

    # Suppress all stdout/stderr from worker processes to prevent pollution
    try:
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                # Also set environment variable to disable tqdm
                os.environ['TQDM_DISABLE'] = '1'
                passed, reason = evaluate_livecodebench_detailed(
                    code, question_id)
                return question_id, passed, reason
    except Exception as e:
        error_msg = f"Error evaluating {question_id}: {type(e).__name__}: {e}"
        # Don't use logger here as it might output to stdout in worker process
        return question_id, False, error_msg


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
    raw_output = validate_text_input(row['model_output_0'])
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


def process_dataframe(df: pd.DataFrame,
                      num_lcb_workers: int = 64) -> pd.DataFrame:
    """Process entire dataframe with optimized batch processing.

    Args:
        df: Input DataFrame to evaluate
        num_lcb_workers: Maximum number of parallel workers for LiveCodeBench evaluation

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
        suffix = f'_{pass_num}'
        df_output[f'extracted_answer{suffix}'] = None
        df_output[f'prompt_accuracy{suffix}'] = 0.0
        df_output[f'evaluation_details{suffix}'] = None

    # Add aggregated columns (max across all passes)
    df_output['prompt_accuracy'] = 0.0
    df_output['evaluation_details'] = None

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
            max_workers = min(multiprocessing.cpu_count(), num_lcb_workers)
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

            # For LiveCodeBench, always use batched evaluation across all
            # passes
            is_livecodebench = 'livecodebench' in dataset_name.lower()
            if is_livecodebench:
                # Validate prerequisites for batched LCB evaluation
                if lcb_executor is None:
                    raise RuntimeError(
                        "LiveCodeBench evaluation requires a shared executor, but it was not initialized. "
                        "This may indicate the LiveCodeBench benchmark failed to load.")

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
                    suffix = f'_{pass_num}'
                    extracted_answer_col = f'extracted_answer{suffix}'
                    for idx in group_indices:
                        row = df_output.loc[idx]
                        extracted = row.get(extracted_answer_col)
                        ground_truth = row.get('ground_truth')

                        if extracted is not None and not pd.isna(ground_truth):
                            all_work_items.append((extracted, ground_truth))
                            work_item_metadata.append((idx, pass_num))

                if all_work_items:
                    # Submit all work at once for maximum parallelism
                    max_workers = min(
                        multiprocessing.cpu_count(), len(all_work_items), num_lcb_workers)
                    logger.info(
                        f"Evaluating {len(all_work_items)} LiveCodeBench items across {pass_k} passes with {max_workers} workers")

                    future_to_metadata = {
                        lcb_executor.submit(evaluate_livecodebench_worker, work_item): metadata
                        for work_item, metadata in zip(all_work_items, work_item_metadata)
                    }

                    # Collect results and assign to appropriate pass columns
                    pass_results = {i: {'correct': 0, 'total': 0}
                                    for i in range(pass_k)}

                    for future in tqdm(as_completed(future_to_metadata, timeout=1200),
                                       total=len(future_to_metadata),
                                       desc=f"Evaluating LiveCodeBench (all passes)"):
                        idx, pass_num = future_to_metadata[future]
                        suffix = f'_{pass_num}'
                        prompt_accuracy_col = f'prompt_accuracy{suffix}'
                        evaluation_details_col = f'evaluation_details{suffix}'

                        try:
                            question_id, is_correct, detailed_reason = future.result(
                                timeout=80)
                            df_output.at[idx,
                                         prompt_accuracy_col] = 100.0 if is_correct else 0.0
                            df_output.at[idx,
                                         evaluation_details_col] = detailed_reason
                            pass_results[pass_num]['total'] += 1
                            if is_correct:
                                pass_results[pass_num]['correct'] += 1
                        except TimeoutError:
                            logger.warning(
                                f"Timeout evaluating row {idx} pass {pass_num}: Test execution exceeded 80s timeout")
                            df_output.at[idx, prompt_accuracy_col] = 0.0
                            df_output.at[idx,
                                         evaluation_details_col] = "Timeout: Test execution exceeded time limit"
                            pass_results[pass_num]['total'] += 1
                        except Exception as e:
                            logger.error(
                                f"Error evaluating row {idx} pass {pass_num}: {e}")
                            df_output.at[idx, prompt_accuracy_col] = 0.0
                            df_output.at[idx,
                                         evaluation_details_col] = f"Error: {e}"
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
                # Sequential pass processing for non-LCB datasets
                for pass_num in range(pass_k):
                    suffix = f'_{pass_num}'
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
                    # Sequential evaluation for all non-LCB datasets
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
                        accuracy = correct_count / total_evaluated * 100
                        logger.info(
                            f"{dataset_name} pass {pass_num} results: {correct_count}/{total_evaluated} correct ({accuracy:.1f}% accuracy)")

            # Aggregate results across all passes (take max)
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

    # Calculate statistics - always use aggregated prompt_accuracy (max across
    # passes)
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

    # Use exact_match as the metric key
    metric_key = 'exact_match'

    results = {
        # 'evaluated': int(evaluated),
        # 'correct': int(correct),
        metric_key: float(accuracy),
        'tokens_per_sample': mean_output_len,
        'num-samples': len(df_evaluated),
        'pass_k': pass_k,
    }

    # Report individual pass accuracies
    for i in range(pass_k):
        pass_acc = df_evaluated[f'prompt_accuracy_{i}'].mean()
        results[f'{metric_key}_pass_{i}'] = float(pass_acc)

    print("\nResults\n")
    print(results)


def process_and_save_dataframe(df: pd.DataFrame,
                               output_dir: Optional[Union[str, Path]] = None,
                               base_filename: Optional[str] = None,
                               num_lcb_workers: int = 64) -> Tuple[pd.DataFrame, str]:
    """Process dataframe for evaluation and save the results.

    Args:
        df: Input DataFrame to evaluate
        output_dir: Directory to save the evaluated pickle file (defaults to same dir as source)
        base_filename: Base filename for output (defaults to auto-generated)
        num_lcb_workers: Maximum number of parallel workers for LiveCodeBench evaluation

    Returns:
        Tuple of (evaluated_dataframe, saved_file_path)
    """
    # Process the dataframe
    df_evaluated = process_dataframe(df, num_lcb_workers=num_lcb_workers)

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
    parser.add_argument("--num-lcb-workers", type=int, default=64,
                        help="Maximum number of parallel workers for LiveCodeBench evaluation (default: 64)")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose logging")

    args = parser.parse_args()

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
        base_filename=output_filename,
        num_lcb_workers=args.num_lcb_workers
    )

    # Print evaluation results with unified function
    print_evaluation_results(df_evaluated, logger)

    logger.info("Done!")


if __name__ == "__main__":
    main()
