# experiment config
import sys
from argparse import ArgumentParser
import simplejson as json
import os
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets

def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model-id",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="Model ID to get the tokenizer",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="cnn_dailymail",
        help="Model ID to get the tokenizer",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="3.0.0",
        help="Model ID to get the tokenizer",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="article",
        help="Model ID to get the tokenizer",
    )
    parser.add_argument(
        "--summary-column",
        type=str,
        default="highlights",
        help="Model ID to get the tokenizer",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Model ID to get the tokenizer",
    )

    return parser.parse_args()

args = get_args()
model_id = args.model_id
dataset_id = args.dataset_id
dataset_config = args.dataset_config
text_column = args.text_column
summary_column = args.summary_column
n_samples = args.n_samples


save_dataset_path = os.environ.get("DATASET_CNNDM_PATH", "data")

# Check whether the specified path exists or not
isExist = os.path.exists(save_dataset_path)
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(save_dataset_path)

# Load dataset from the hub
dataset = load_dataset(dataset_id, name=dataset_config)
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 2048


instruction_template = "In very brief sentences, summarize the following news article. Only return the summary.\nArticle: {}\nSummary: "

prompt_length = len(tokenizer(instruction_template)["input_ids"])
max_sample_length = tokenizer.model_max_length - prompt_length


def preprocess_function(sample, padding="max_length"):
    # create list of samples
    inputs = []

    for i in range(0, len(sample[text_column])):
        x = dict()
        x["instruction"] = instruction_template
        x["input"] = sample[text_column][i]
        x["tok_input"] = tokenizer.encode(instruction_template.format(x["input"]))
        x["output"] = sample[summary_column][i]
        inputs.append(x)
    model_inputs = dict()
    model_inputs["text"] = inputs

    return model_inputs


# process dataset
tokenized_dataset = dataset.map(
    preprocess_function, batched=True, remove_columns=list(dataset["train"].features)
)

# save dataset to disk
if n_samples is None:
    with open(os.path.join(save_dataset_path, "cnn_eval.json"), "w") as write_f:
        json.dump(
            tokenized_dataset["validation"]["text"], write_f, indent=4, ensure_ascii=False
        )
else:
    with open(os.path.join(save_dataset_path, f"sample_cnn_eval_{n_samples}.json"), "w") as write_f:
        json.dump(
            tokenized_dataset["validation"]["text"][:n_samples], write_f, indent=4, ensure_ascii=False
        )


print("Dataset saved in ", save_dataset_path)
