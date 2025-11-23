"""Task definitions for the VL2L benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from datasets import load_dataset
from hiclass.metrics import f1
from loguru import logger
from sklearn.metrics import f1_score
from tabulate import tabulate

if TYPE_CHECKING:
    from pydantic import FilePath

    from .cli import Dataset as DatasetCLI


def get_hierarchical_components(predicted_path: str,
                                true_path: str,
                                separator: str = " > ") -> tuple[int, int, int]:
    """Calculates the components for Hierarchical Precision.

    Args:
        predicted_path: Categories predicted by the VLM.
        true_path: Ground truth categories.
        separator: String used to separate each category.

    Returns:
        Tuple of number of intersections,
        correctly predicted categories and
        ground truth categories.
    """
    # 1. Split the paths into categories (nodes)
    predicted_categories = [c.strip() for c in predicted_path.split(separator)]
    true_categories = [c.strip() for c in true_path.split(separator)]

    # Check for empty paths
    if not predicted_categories or not true_categories:
        return 0, len(predicted_categories), len(true_categories)

    # 2. Count the intersection (longest common prefix)
    intersection_count = 0

    # Iterate through the paths simultaneously
    for pred_cat, true_cat in zip(predicted_categories,
                                  true_categories,
                                  strict=False):
        if pred_cat == true_cat:
            intersection_count += 1
        else:
            # Stop as soon as a mismatch is found (enforces hierarchical match)
            break

    pred_length = len(predicted_categories)
    true_length = len(true_categories)

    return intersection_count, pred_length, true_length


def calculate_hierarchical_f1(data: list[tuple[str, str]]) -> float:
    """Calculates the aggregate hF scores for a list of samples.

    Args:
        data: A list of tuples, where each tuple is
            (predicted_path_str, true_path_str).

    Returns:
        F1 score
    """
    total_intersection = 0
    total_predicted_length = 0
    total_true_length = 0

    # 1. Aggregate the components across all samples
    for pred_path, true_path in data:
        intersection, pred_len, true_len = \
            get_hierarchical_components(pred_path, true_path)

        total_intersection += intersection
        total_predicted_length += pred_len
        total_true_length += true_len

    # 2. Calculate hP and hR
    hp = total_intersection / total_predicted_length \
        if total_predicted_length > 0 else 0.0
    hr = total_intersection / total_true_length \
        if total_true_length > 0 else 0.0

    return 0.0 if hp + hr == 0 else 2 * (hp * hr) / (hp + hr)


def calculate_exact_match(generated_text: str, original_text: str) -> float:
    """Calculates binary Exact Match (EM) score.

    We clean the text (lowercase, strip whitespace) for a fairer comparison.

    Args:
        generated_text: Output from the VLM.
        original_text: Ground truth information from the dataset.

    Returns:
        1 if the values match or 0 otherwise
    """
    gen = generated_text.strip().lower()
    orig = original_text.strip().lower()

    return 1.0 if gen == orig else 0.0

def calculate_secondhand_f1(data: list[tuple[str, str]]) -> float:
    """Calculate F1 score of is_secondhand field.

    Args:
         data: List of tuples of predicted and true values
    Returs:
        f1 score
    """
    y_pred = []
    y_src = []
    for pred, src in data:
        y_pred.append(pred)
        y_src.append(src)

    return f1_score(y_src, y_pred)

def calculate_hiclass_f1(data: list[tuple[str, str]]) -> float:
    """Alt method to calculate hierarchical F1.

    Args:
         data: List of tuples of predicted and true values
    Returs:
        f1 score
    """
    y_pred_raw = []
    y_true_raw = []

    for pred, src in data:
        path1 = pred.split(" > ")
        path2 = src.split(" > ")

        y_pred_raw.append(path1)
        y_true_raw.append(path2)

    # 2. Find the global maximum length across ALL samples
    # We check the longest path in both true and pred lists
    max_len = max(len(p) for p in y_true_raw + y_pred_raw)

    # 3. Pad all lists to the global max_len
    for i in range(len(y_true_raw)):
        # Pad Truth
        pad_len_true = max_len - len(y_true_raw[i])
        y_true_raw[i] += [""] * pad_len_true

        # Pad Prediction
        pad_len_pred = max_len - len(y_pred_raw[i])
        y_pred_raw[i] += [""] * pad_len_pred

    # 4. Convert to numpy arrays
    y_true = np.array(y_true_raw)
    y_pred = np.array(y_pred_raw)

    # 5. Calculate Score
    return f1(y_true, y_pred)


def run_evaluation(filename: FilePath, dataset: DatasetCLI) -> None:
    """Main function to run the evaluation."""
    with Path.open(filename) as f:
        model_output = json.load(f)

    original_data = load_dataset(
        dataset.repo_id,
        dataset.token,
        split="+".join(dataset.split),
    )

    category_dataset_pred_src = []
    is_secondhand_pred_src = []
    for elem in model_output:
        byte_data = bytes.fromhex(elem["data"])
        idx = elem["qsl_idx"]
        pred_text_decode = byte_data.decode("utf-8")
        pred_item = json.loads(pred_text_decode)
        ground_truth_item = original_data[idx]
        category_dataset_pred_src.append((pred_item["category"],
                                          ground_truth_item["ground_truth_category"]))
        is_secondhand_pred_src.append((int(pred_item["is_secondhand"]),
                                      int(ground_truth_item["ground_truth_is_secondhand"])))

    category_f1_score = calculate_hierarchical_f1(
        category_dataset_pred_src)
    hiclass_f1 = calculate_hiclass_f1(category_dataset_pred_src)
    is_secondhand_f1_score = calculate_secondhand_f1(is_secondhand_pred_src)

    data = [
        ["category", category_f1_score, hiclass_f1],
        ["is_secondhand", is_secondhand_f1_score],
    ]

    logger.info("Results:\n{}", tabulate(data,
                                         headers=["Fields", "F1 Score",
                                                  "HiClass F1 Score"],
                                         tablefmt="fancy_grid"))
