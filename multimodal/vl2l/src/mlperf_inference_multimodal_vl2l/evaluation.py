"""Task definitions for the VL2L benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from datasets import load_dataset
from hiclass.metrics import f1  # type: ignore[import-untyped]
from loguru import logger
from pydantic import ValidationError
from rapidfuzz import fuzz  # type: ignore[import-untyped]
from sklearn.metrics import f1_score  # type: ignore[import-untyped]
from tabulate import tabulate

if TYPE_CHECKING:
    from pydantic import FilePath

    from .cli import Dataset as DatasetCLI

from .schema import ProductMetadata

_TRUE_CATEGORY_PAD = "<|__TRUE_CATEGORY_PAD__|>"
_PRED_CATEGORY_PAD = "<|__PRED_CATEGORY_PAD__|>"
_PRED_BRAND_PAD = "<|__PRED_BRAND_PAD__|>"
_CATEGORY_SEPARATOR = " > "


def get_hierarchical_components(
    predicted_path: str,
    true_path: str,
    separator: str = _CATEGORY_SEPARATOR,
) -> tuple[int, int, int]:
    """Calculates the components for Hierarchical Precision.

    Args:
        predicted_path: Categories predicted by the VLM.
        true_path: Ground truth categories.
        separator: The separator used to separate each level of the category.

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
    for pred_cat, true_cat in zip(predicted_categories, true_categories, strict=False):
        if pred_cat == true_cat:
            intersection_count += 1
        else:
            # Stop as soon as a mismatch is found (enforces hierarchical match)
            break

    pred_length = len(predicted_categories)
    true_length = len(true_categories)

    return intersection_count, pred_length, true_length


def calculate_hierarchical_f1(
    data: list[tuple[str, str]],
    separator: str = _CATEGORY_SEPARATOR,
) -> float:
    """Calculates the aggregate hF scores for a list of samples.

    Args:
        data: A list of tuples, where each tuple is (predicted_path_str, true_path_str).
        separator: The separator used to split the paths into levels of the category.

    Returns:
        F1 score
    """
    total_intersection = 0
    total_predicted_length = 0
    total_true_length = 0

    # 1. Aggregate the components across all samples
    for pred_path, true_path in data:
        intersection, pred_len, true_len = get_hierarchical_components(
            predicted_path=pred_path,
            true_path=true_path,
            separator=separator,
        )

        total_intersection += intersection
        total_predicted_length += pred_len
        total_true_length += true_len

    # 2. Calculate hP and hR
    hp = (
        total_intersection / total_predicted_length
        if total_predicted_length > 0
        else 0.0
    )
    hr = total_intersection / total_true_length if total_true_length > 0 else 0.0

    return 0.0 if hp + hr == 0 else 2 * (hp * hr) / (hp + hr)


def calculate_brand_f1_score(data: list[tuple[str, str]]) -> float:
    """Calculate the F1 score of brand field.

    Args:
        data: A list of tuples, where each tuple is
            (predicted_path_str, true_path_str).

    Returns:
        F1 score
    """
    valid_threshold = 90
    matches = []
    for pred, src in data:
        norm_truth = src.strip().lower()
        norm_pred = pred.strip().lower()

        # Fuzzy Match (Handles typos like "Adodas")
        # fuzz.ratio calculates edit distance similarity (0-100)
        score = fuzz.ratio(norm_truth, norm_pred)

        # Threshold: If > 90/100 similarity, count as correct
        if score > valid_threshold:
            matches.append(1)
        else:
            matches.append(0)

    # Calculate the Score
    # For 1-to-1 extraction, Accuracy = Recall = Micro F1
    return sum(matches) / len(matches)


def calculate_secondhand_f1(data: list[tuple[bool, bool]]) -> float:
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


def calculate_hiclass_f1(
    data: list[tuple[str, str]],
    separator: str = _CATEGORY_SEPARATOR,
) -> float:
    """Alt method to calculate hierarchical F1.

    Args:
        data: List of tuples of predicted and true values
        separator: The separator used to split the paths into levels of the category.

    Returs:
        f1 score
    """
    y_pred_raw = []
    y_true_raw = []

    for pred, src in data:
        path1 = pred.split(separator)
        path2 = src.split(separator)

        y_pred_raw.append(path1)
        y_true_raw.append(path2)

    # 2. Find the global maximum length across ALL samples
    # We check the longest path in both true and pred lists
    max_len = max(len(p) for p in y_true_raw + y_pred_raw)

    # 3. Pad all lists to the global max_len
    for i in range(len(y_true_raw)):
        # Pad Truth
        pad_len_true = max_len - len(y_true_raw[i])
        y_true_raw[i] += [_TRUE_CATEGORY_PAD] * pad_len_true

        # Pad Prediction
        pad_len_pred = max_len - len(y_pred_raw[i])
        y_pred_raw[i] += [_PRED_CATEGORY_PAD] * pad_len_pred

    # 4. Convert to numpy arrays
    y_true = np.array(y_true_raw)
    y_pred = np.array(y_pred_raw)

    # 5. Calculate Score
    return f1(y_true, y_pred)


def run_evaluation(random_seed: int, filename: FilePath, dataset: DatasetCLI) -> None:
    """Main function to run the evaluation."""
    rng = np.random.default_rng(seed=random_seed)
    with Path.open(filename) as f:
        model_output = json.load(f)

    original_data = load_dataset(
        dataset.repo_id,
        token=dataset.token,
        split="+".join(dataset.split),
    )

    num_unparsable_responses = 0
    category_dataset_pred_src = []
    category_rand_pred_src = []
    is_secondhand_pred_src = []
    is_secondhand_rand_pred_src = []
    brand_pred_src = []

    all_possible_brands = set()

    for elem in model_output:
        idx = elem["qsl_idx"]
        response = bytes.fromhex(elem["data"]).decode("utf-8")
        ground_truth_item = original_data[idx]
        all_possible_brands.add(ground_truth_item["ground_truth_brand"])
        try:
            pred_item = ProductMetadata.model_validate_json(response)
        except ValidationError:
            num_unparsable_responses += 1
            pred_item = ProductMetadata(
                category=_CATEGORY_SEPARATOR.join(
                    [_PRED_CATEGORY_PAD]
                    * len(
                        ground_truth_item["ground_truth_category"].split(
                            _CATEGORY_SEPARATOR,
                        ),
                    ),
                ),
                brand=_PRED_BRAND_PAD,
                is_secondhand=rng.choice([True, False], size=1).tolist()[0],
            )
            logger.error(
                "Response\n{}\n(for the sample at index {}) cannot be validated against"
                " the expected schema. Overwriting this response into \n{}\n",
                response,
                idx,
                pred_item,
            )
        category_dataset_pred_src.append(
            (pred_item.category, ground_truth_item["ground_truth_category"]),
        )
        is_secondhand_pred_src.append(
            (
                pred_item.is_secondhand,
                ground_truth_item["ground_truth_is_secondhand"],
            ),
        )
        brand_pred_src.append(
            (pred_item.brand, ground_truth_item["ground_truth_brand"]),
        )
        # random category selection
        # Uniform distribution is the default
        rand_cat = rng.choice(ground_truth_item["potential_product_categories"])
        category_rand_pred_src.append(
            (rand_cat, ground_truth_item["ground_truth_category"]),
        )
        # random is_secondhand selection
        rand_is_secondhand = rng.choice([True, False])
        is_secondhand_rand_pred_src.append(
            (rand_is_secondhand, ground_truth_item["ground_truth_is_secondhand"]),
        )

    category_f1_score = calculate_hierarchical_f1(category_dataset_pred_src)
    hiclass_f1_score = calculate_hiclass_f1(category_dataset_pred_src)
    is_secondhand_f1_score = calculate_secondhand_f1(is_secondhand_pred_src)
    brand_score = calculate_brand_f1_score(brand_pred_src)

    rand_cat_f1_score = calculate_hierarchical_f1(category_rand_pred_src)
    rand_hiclass_f1_score = calculate_hiclass_f1(category_rand_pred_src)
    rand_is_seconhand_f1_score = calculate_secondhand_f1(is_secondhand_rand_pred_src)
    rand_brand_score = calculate_brand_f1_score(
        [
            (
                rng.choice(list(all_possible_brands)),
                original_data[elem["qsl_idx"]]["ground_truth_brand"],
            )
            for elem in model_output
        ],
    )

    logger.info(
        "{} responses cannot be parsed against the expected schema. Results:\n{}",
        num_unparsable_responses,
        tabulate(
            [
                [
                    "From accuracy file",
                    category_f1_score,
                    hiclass_f1_score,
                    brand_score,
                    is_secondhand_f1_score,
                ],
                [
                    "Random selection",
                    rand_cat_f1_score,
                    rand_hiclass_f1_score,
                    rand_brand_score,
                    rand_is_seconhand_f1_score,
                ],
            ],
            headers=[
                "Results",
                "Category hierarchical F1 Score",
                "Category HiClass F1 Score",
                "Brand F1 Score",
                "Is_secondhand F1 Score",
            ],
            tablefmt="fancy_grid",
        ),
    )
