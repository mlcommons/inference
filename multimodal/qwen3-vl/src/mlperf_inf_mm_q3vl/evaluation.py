"""Task definitions for the Qwen3-VL (Q3VL) benchmark."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from datasets import load_dataset
from loguru import logger
from pydantic import ValidationError
from rapidfuzz import fuzz  # type: ignore[import-untyped]
from sklearn.metrics import f1_score  # type: ignore[import-untyped]
from tabulate import tabulate
import hashlib

if TYPE_CHECKING:
    from typing import Any

    from pydantic import FilePath

    from .cli import Dataset as DatasetCLI

from .schema import ProductMetadata

_PRED_CATEGORY_PAD = "<|__PRED_CATEGORY_PAD__|>"
_PRED_BRAND_PAD = "<|__PRED_BRAND_PAD__|>"
_CATEGORY_SEPARATOR = " > "

_WORKER_CONTEXT = {}
_MAX_JOBS = 4


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
    for pred_cat, true_cat in zip(
            predicted_categories, true_categories, strict=False):
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


def _process_chunk_rnd_brand(args: tuple[str, dict, dict]) -> tuple[str, str]:
    """Function to process only chunks for random brand predictions.

    Args:
        args: Tuple containing
    """
    pred_brand, elem, data_source = args
    # We pass the specific data row needed, or the whole structure if efficient
    return (pred_brand, data_source[elem["qsl_idx"]]["ground_truth_brand"])


def init_worker(dataset: dict) -> None:
    """Initialize worker data to process each chunk.

    Args:
        dataset: huggingface dataset
    """
    _WORKER_CONTEXT["dataset"] = dataset


def _process_chunk(args: tuple[list[dict], int]) -> dict[str, Any]:
    """Retrieve relevant information from each chunk of data.

    Args:
        args: Tuple that contains chunk of data and seed

    Returns:
        Object with processed information
    """
    chunk_data, seed = args

    # 1. Access the global dataset
    dataset = _WORKER_CONTEXT["dataset"]

    # 2. Create a local, reproducible RNG for this specific chunk
    local_rng = np.random.default_rng(seed)

    num_unparsable_responses = 0
    category_dataset_pred_src = []
    category_rand_pred_src = []
    is_secondhand_pred_src = []
    is_secondhand_rand_pred_src = []
    brand_pred_src = []
    all_possible_brands = set()
    error_messages = []

    for elem in chunk_data:
        idx = elem["qsl_idx"]
        response = bytes.fromhex(elem["data"]).decode("utf-8")
        ground_truth_item = dataset[idx]
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
                is_secondhand=local_rng.choice(
                    [True, False], size=1).tolist()[0],
            )
            error_messages.append(
                (
                    f"Response\n{response}\n(for the sample at index {idx})"
                    f"cannot be validated against the expected schema. "
                    f"Overwriting this response into \n{pred_item}\n",
                ),
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
        rand_cat = local_rng.choice(
            ground_truth_item["potential_product_categories"])
        category_rand_pred_src.append(
            (rand_cat, ground_truth_item["ground_truth_category"]),
        )
        # random is_secondhand selection
        rand_is_secondhand = local_rng.choice([True, False])
        is_secondhand_rand_pred_src.append(
            (rand_is_secondhand,
             ground_truth_item["ground_truth_is_secondhand"]),
        )

    return {
        "num_unparsable_responses": num_unparsable_responses,
        "error_messages": error_messages,
        "category_dataset_pred_src": category_dataset_pred_src,
        "category_rand_pred_src": category_rand_pred_src,
        "is_secondhand_pred_src": is_secondhand_pred_src,
        "is_secondhand_rand_pred_src": is_secondhand_rand_pred_src,
        "brand_pred_src": brand_pred_src,
        "all_possible_brands": list(all_possible_brands),
    }


def run_evaluation(random_seed: int, filename: FilePath,
                   dataset: DatasetCLI) -> None:
    """Main function to run the evaluation."""
    master_rng = np.random.default_rng(seed=random_seed)
    with Path.open(filename) as f:
        model_output = json.load(f)

    original_data = load_dataset(
        dataset.repo_id,
        token=dataset.token,
        split="+".join(dataset.split),
    )

    # get number of available CPU and get chunk size
    cpu_count = min(os.cpu_count() or 1, _MAX_JOBS)
    chunk_size = max(len(model_output) // cpu_count, 1)
    # Create chunks
    output_chunks = [
        model_output[i: i + chunk_size]
        for i in range(0, len(model_output), chunk_size)
    ]

    # Generate Seeds
    # One seed per chunk to ensure reproducibility.
    # The master_rng generates these,
    # so the whole run is deterministic based on `random_seed`.
    chunk_seeds = master_rng.integers(0, 2**32, size=len(output_chunks))

    # Zip them: Each task is ([model_out_1, ...], 12345)
    tasks = zip(output_chunks, chunk_seeds, strict=False)

    num_unparsable_responses = 0
    err_messages = []
    category_dataset_pred_src = []
    category_rand_pred_src = []
    is_secondhand_pred_src = []
    is_secondhand_rand_pred_src = []
    brand_pred_src = []
    all_possible_brands = []

    with ProcessPoolExecutor(
        max_workers=cpu_count,
        initializer=init_worker,
        initargs=(original_data,),
    ) as executor:
        # Execute
        chunk_results = list(executor.map(_process_chunk, tasks))

    for chunk in chunk_results:
        num_unparsable_responses += chunk["num_unparsable_responses"]
        err_messages.extend(chunk["error_messages"])
        category_dataset_pred_src.extend(chunk["category_dataset_pred_src"])
        category_rand_pred_src.extend(chunk["category_rand_pred_src"])
        is_secondhand_pred_src.extend(chunk["is_secondhand_pred_src"])
        is_secondhand_rand_pred_src.extend(
            chunk["is_secondhand_rand_pred_src"])
        brand_pred_src.extend(chunk["brand_pred_src"])
        all_possible_brands.extend(chunk["all_possible_brands"])

    for err in err_messages:
        logger.error("{}", err)

    category_f1_score = calculate_hierarchical_f1(category_dataset_pred_src)
    is_secondhand_f1_score = calculate_secondhand_f1(is_secondhand_pred_src)
    brand_score = calculate_brand_f1_score(brand_pred_src)

    rand_cat_f1_score = calculate_hierarchical_f1(category_rand_pred_src)

    rand_is_seconhand_f1_score = calculate_secondhand_f1(
        is_secondhand_rand_pred_src)

    all_brands_list = list(set(all_possible_brands))
    random_brand_predictions = master_rng.choice(
        all_brands_list,
        size=len(model_output),
    )

    args_list = (
        (pred, elem, original_data)
        for pred, elem in zip(random_brand_predictions, model_output, strict=False)
    )

    with ProcessPoolExecutor() as executor:
        rand_brand_data = list(
            executor.map(
                _process_chunk_rnd_brand,
                args_list,
                chunksize=chunk_size),
        )

    rand_brand_score = calculate_brand_f1_score(
        rand_brand_data,
    )

    logger.info(
        "{} responses cannot be parsed against the expected schema. Results:\n{}",
        num_unparsable_responses,
        tabulate(
            [
                [
                    "From accuracy file",
                    category_f1_score,
                    brand_score,
                    is_secondhand_f1_score,
                ],
                [
                    "Random selection",
                    rand_cat_f1_score,
                    rand_brand_score,
                    rand_is_seconhand_f1_score,
                ],
            ],
            headers=[
                "Results",
                "Category hierarchical F1 Score",
                "Brand F1 Score",
                "Is_secondhand F1 Score",
            ],
            tablefmt="fancy_grid",
        ),
    )

    # Generate accuracy.txt file
    results_dict = {"f1": category_f1_score}
    data_string = json.dumps(results_dict, sort_keys=True)
    file_hash = hashlib.sha256(data_string.encode()).hexdigest()

    with open("accuracy.txt", "w") as f:
        f.write("Results\n\n")
        f.write(f"{data_string}\n\n")
        f.write(f"hash={file_hash}")
