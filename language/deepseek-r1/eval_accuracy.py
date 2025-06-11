#!/usr/bin/env python3
"""
Standalone evaluation script for mlperf-inference deepseek-r1 dataset.

Expected input format (pickle file with DataFrame):
- model_output: The model's response text
- tok_model_output_len: The length of the model's response tokens
- ground_truth: The expected answer
- dataset: Dataset name (e.g., 'gpqa', 'mmlu_pro', 'math500', 'livecodebench', 'aime')
- question: The question text

Output adds two columns:
- extracted_answer: Parsed answer from model output
- prompt_accuracy: 100.0 if correct, 0.0 if incorrect
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


# =============================================================================
# MLPerf Log Accuracy Processing
# =============================================================================

def process_mlperf_log_accuracy(mlperf_log_file: Union[str, Path],
                                dataset_file: Union[str, Path],
                                checkpoint_path: str,
                                dtype: str = "int32",
                                output_dir: Optional[Union[str, Path]] = None,
                                base_filename: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    """Process MLPerf log accuracy file and evaluate results.

    Args:
        mlperf_log_file: Path to mlperf_log_accuracy.json
        dataset_file: Path to ground truth dataset pickle file
        checkpoint_path: Path to tokenizer checkpoint
        dtype: Data type for numpy conversion ("int32", "int64", "float") - default "int32" matches MLPerf SUT
        output_dir: Directory to save evaluated results
        base_filename: Base filename for output file

    Returns:
        Tuple of (evaluated_dataframe, saved_file_path)
    """
    # Validate input files
    mlperf_log_file = Path(mlperf_log_file)
    dataset_file = Path(dataset_file)

    if not mlperf_log_file.exists():
        raise FileNotFoundError(
            f"MLPerf log file not found: {mlperf_log_file}")
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    logger.info(f"Processing MLPerf log: {mlperf_log_file}")
    logger.info(f"Using dataset: {dataset_file}")
    logger.info(f"Using checkpoint: {checkpoint_path}")

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            model_max_length=22000,
            padding_side="left",
            use_fast=False,
        )
        logger.info("Tokenizer loaded successfully")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load tokenizer from {checkpoint_path}: {e}")

    # Load ground truth dataset
    try:
        with open(dataset_file, 'rb') as f:
            dataset_df = pickle.load(f)

        if isinstance(dataset_df, pd.DataFrame):
            if 'gt_output' in dataset_df.columns:
                ground_truths = dataset_df['gt_output'].tolist()
            elif 'ground_truth' in dataset_df.columns:
                ground_truths = dataset_df['ground_truth'].tolist()
            else:
                raise ValueError(
                    "Dataset must contain 'gt_output' or 'ground_truth' column")

            # Get other required columns with fallbacks
            if 'dataset' in dataset_df.columns:
                datasets = dataset_df['dataset'].tolist()
            elif 'metric' in dataset_df.columns:
                # Infer dataset from metric names
                datasets = [
                    metric.replace(
                        '_em',
                        '').replace(
                        '_',
                        '') for metric in dataset_df['metric'].tolist()]
            else:
                datasets = ['unknown'] * len(ground_truths)

            if 'question' in dataset_df.columns:
                questions = dataset_df['question'].tolist()
            elif 'input' in dataset_df.columns:
                questions = dataset_df['input'].tolist()
            else:
                questions = [''] * len(ground_truths)
        else:
            # Fallback for other formats
            ground_truths = dataset_df if isinstance(dataset_df, list) else []
            datasets = ['unknown'] * len(ground_truths)
            questions = [''] * len(ground_truths)

        logger.info(f"Loaded {len(ground_truths)} ground truth entries")
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset file: {e}")

    # Set numpy dtype - default to int32 to match MLPerf SUT output format
    eval_dtype = np.int32
    if dtype == "int64":
        eval_dtype = np.int64
    elif dtype == "float":
        eval_dtype = np.float32

    # Load MLPerf log
    try:
        # First, check if this is a JSON array format or newline-delimited JSON
        with open(mlperf_log_file, 'r') as f:
            first_line = f.readline().strip()

        if first_line == '[':
            # JSON array format - load the entire file
            logger.info("Detected JSON array format")
            with open(mlperf_log_file, 'r') as f:
                try:
                    mlperf_results = json.load(f)
                except json.JSONDecodeError as e:
                    # If full file parsing fails, try to parse line by line,
                    # skipping brackets
                    logger.warning(
                        f"Failed to parse as complete JSON array: {e}")
                    logger.info("Attempting line-by-line parsing")
                    mlperf_results = []
                    with open(mlperf_log_file, 'r') as f2:
                        for line_num, line in enumerate(f2, 1):
                            line = line.strip()
                            # Skip empty lines, opening/closing brackets
                            if not line or line == '[' or line == ']':
                                continue
                            # Remove trailing comma if present
                            if line.endswith(','):
                                line = line[:-1]
                            try:
                                mlperf_results.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                logger.warning(
                                    f"Failed to parse line {line_num}: {e}")
                                continue
        else:
            # Newline-delimited JSON format
            logger.info("Detected newline-delimited JSON format")
            mlperf_results = []
            with open(mlperf_log_file, 'r') as f:
                # Reset to beginning since we read the first line
                f.seek(0)
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    try:
                        mlperf_results.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num}: {e}")
                        continue

        logger.info(f"Loaded {len(mlperf_results)} MLPerf results")
    except Exception as e:
        raise RuntimeError(f"Failed to load MLPerf log file: {e}")

    # Process MLPerf results
    target_required = []
    datasets_required = []
    questions_required = []
    preds_token_ids = []
    tok_lengths = []

    seen_qsl_idx = set()

    for pred_entry in mlperf_results:
        qsl_idx = pred_entry["qsl_idx"]

        # Skip duplicates
        if qsl_idx in seen_qsl_idx:
            continue
        seen_qsl_idx.add(qsl_idx)

        # Validate qsl_idx is within bounds
        if qsl_idx >= len(ground_truths):
            logger.warning(f"qsl_idx {qsl_idx} out of bounds, skipping")
            continue

        # Extract and convert hex data to token IDs
        try:
            hex_data = pred_entry["data"]
            token_ids = np.frombuffer(bytes.fromhex(hex_data), eval_dtype)
            tok_lengths.append(len(token_ids))
            preds_token_ids.append(token_ids)

            # Map to ground truth
            target_required.append(ground_truths[qsl_idx])
            datasets_required.append(datasets[qsl_idx])
            questions_required.append(questions[qsl_idx])

        except Exception as e:
            logger.warning(
                f"Error processing entry with qsl_idx {qsl_idx}: {e}")
            continue

    if not preds_token_ids:
        raise ValueError("No valid predictions found in MLPerf log")

    logger.info(f"Processing {len(preds_token_ids)} valid predictions")

    # Decode tokens to text
    try:
        preds_decoded_text = tokenizer.batch_decode(
            preds_token_ids, skip_special_tokens=True
        )
        logger.info("Token decoding completed")
    except Exception as e:
        raise RuntimeError(f"Failed to decode tokens: {e}")

    # Create DataFrame for evaluation
    df_data = {
        'model_output': preds_decoded_text,
        'ground_truth': target_required,
        'dataset': datasets_required,
        'question': questions_required,
        'tok_model_output_len': tok_lengths
    }

    df = pd.DataFrame(df_data)
    logger.info(f"Created DataFrame with {len(df)} rows")

    # Process and save using existing unified function
    df_evaluated, saved_file_path = process_and_save_dataframe(
        df,
        output_dir=output_dir,
        base_filename=base_filename
    )

    # Print results using unified function
    print_evaluation_results(df_evaluated, logger)

    return df_evaluated, saved_file_path


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
        'ground_truth',
        'tok_model_output_len']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


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
    }
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
        else:
            # Sequential evaluation for other datasets
            correct_count = 0
            total_evaluated = 0

            for idx in group_indices:
                row = df_output.loc[idx]
                extracted = row['extracted_answer']
                ground_truth = row['ground_truth']

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
        'mean-accuracy': float(accuracy),
        'mean-output-tok-len': mean_output_len,
        'num-samples': len(df_evaluated),
    }

    result_str = json.dumps(results, indent=2)
    logger.info(f"\nEvaluation Results: {result_str}")

    return results


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

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model outputs with parallel LiveCodeBench processing")
    parser.add_argument("--input-file", required=True,
                        help="Input pickle file")
    parser.add_argument(
        "--output-file", help="Output pickle file (defaults to <input-file>_evaluated.pkl)")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    # Determine output file path
    if args.output_file:
        output_path = Path(args.output_file)
        output_dir = output_path.parent
        output_filename = output_path.name
    else:
        input_path = Path(args.input_file)
        output_dir = input_path.parent
        output_filename = input_path.stem + "_evaluated.pkl"

    logger.info(f"Processing: {args.input_file}")

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
