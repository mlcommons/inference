"""Task definitions for the VL2L benchmark."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

from datasets import load_dataset
from loguru import logger
from sklearn.metrics import f1_score
from tabulate import tabulate

if TYPE_CHECKING:
    from .cli import Dataset as DatasetCLI

class Evaluator:
    """Class used to evaluate the accuracy of the VLM."""

    def __init__(self, filename: Path, dataset_cli: "DatasetCLI")-> None:
        """Initialize class.

        Args:
            filename: Location of the accuracy file.
            dataset_cli: The dataset configuration passed in from the CLI.
        """
        self.filename = filename
        self.dataset = dataset_cli

        self.verify_file_exists()


    def verify_file_exists(self) -> None:
        """Verify if the accuracy file exists."""
        if not self.filename.exists():
            error_message = f"File :{self.filename.as_posix()} does not exists."
            raise RuntimeError(error_message)

    def get_hierarchical_components(self, predicted_path: str,
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

    def calculate_hierarchical_metrics(self, data: list) -> float:
        """Calculates the aggregate hP, hR, and hF scores for a list of samples.

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
                self.get_hierarchical_components(pred_path, true_path)

            total_intersection += intersection
            total_predicted_length += pred_len
            total_true_length += true_len

        # 2. Calculate hP and hR
        hp = total_intersection / total_predicted_length \
            if total_predicted_length > 0 else 0.0
        hr = total_intersection / total_true_length \
            if total_true_length > 0 else 0.0

        return 0.0 if hp + hr == 0 else 2 * (hp * hr) / (hp + hr)

    def calculate_exact_match(self, generated_text: str, original_text: str) -> float:
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


    def run_evaluation(self) -> None:
        """Main function to run the evaluation."""
        with Path.open(self.filename) as f:
            model_output = json.load(f)

        original_data = load_dataset(
            self.dataset.repo_id,
            token=self.dataset.token,
        )["train"]

        category_dataset_pred_src = []
        is_secondhand_pred = []
        is_secondhand_src = []
        for elem in model_output:
            byte_data = bytes.fromhex(elem["data"])
            idx = elem["qsl_idx"]
            pred_text_decode = byte_data.decode("utf-8")
            """
            Model response is similar to:
            ```json
             {
             .....
             }
            ```
            Need to find open and close brackets
            """
            start_index = pred_text_decode.find("{")
            end_index = pred_text_decode.rfind("}") + 1
            pred_item = json.loads(pred_text_decode[start_index:end_index])
            ground_truth_item = original_data[idx]
            category_dataset_pred_src.append((pred_item["category"],
                                              ground_truth_item["ground_truth_category"]))
            is_secondhand_pred.append(int(pred_item["is_secondhand"]))
            is_secondhand_src.append(int(ground_truth_item["ground_truth_is_secondhand"]))

        category_f1_score = self.calculate_hierarchical_metrics(
            category_dataset_pred_src)
        is_secondhand_f1_score = f1_score(is_secondhand_src,
                                          is_secondhand_pred)

        data = [
            ["category", category_f1_score],
            ["is_secondhand", is_secondhand_f1_score],
        ]

        logger.info("Results:\n{}",tabulate(data,
                                            headers=["Fields", "F1 Score"],
                                            tablefmt="fancy_grid"))


