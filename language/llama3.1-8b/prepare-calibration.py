import os
import sys
import json
from argparse import ArgumentParser
from transformers import AutoTokenizer
from datasets import load_dataset


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--calibration-list-file", required=True, help="Path to calibration list"
    )
    parser.add_argument(
        "--output-dir", help="Output directory", default="calibration-data"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Model ID to get the tokenizer",
    )

    return parser.parse_args()


dataset_id = "cnn_dailymail"
version = "3.0.0"
split = "train"

# Load tokenizer
args = get_args()
tokenizer = AutoTokenizer.from_pretrained(args.model_id)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 8000

instruction_template = "Summarize the following news article in 128 tokens. Please output the summary only, without any other text.\n\nArticle:\n{input}\n\nSummary:"


def check_path(path):
    return os.path.exists(path)


def prepare_calibration_data(calibration_list_file, output_dir):
    if not check_path(calibration_list_file):
        print("Calibration list file not found: {}".format(calibration_list_file))
        sys.exit(1)

    dataset = load_dataset("cnn_dailymail", name="3.0.0", split="train")
    train = dict((x["id"], x) for x in dataset)

    with open(calibration_list_file, "r") as fid:
        calibration_ids = fid.read().splitlines()

    inputs = []
    for id in calibration_ids:
        calibration_sample = train[id]
        x = dict()
        x["instruction"] = instruction_template
        x["input"] = calibration_sample["article"]
        x["tok_input"] = tokenizer.encode(instruction_template.format_map(x))
        x["output"] = calibration_sample["highlights"]
        inputs.append(x)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, "cnn_dailymail_calibration.json")
    with open(output_path, "w") as write_f:
        json.dump(inputs, write_f, indent=4, ensure_ascii=False)

    print("Calibration data saved at {}".format(output_path))


def main():

    args = get_args()
    prepare_calibration_data(args.calibration_list_file, args.output_dir)


if __name__ == "__main__":
    main()
