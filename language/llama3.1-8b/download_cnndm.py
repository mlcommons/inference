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
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Model ID to get the tokenizer",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="cnn_dailymail",
        help="Dataset ID",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="3.0.0",
        help="Dataset version",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="article",
        help="Column containing the full text",
    )
    parser.add_argument(
        "--summary-column",
        type=str,
        default="highlights",
        help="Column containing the summarized text",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of samples ",
    )

    return parser.parse_args()


args = get_args()
model_id = args.model_id
dataset_id = args.dataset_id
dataset_config = args.dataset_config
text_column = args.text_column
summary_column = args.summary_column
n_samples = args.n_samples
instruction = "llama"


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
tokenizer.model_max_length = 8000


instruction_template = {
    "llama": (
        "Summarize the following news article in 128 tokens. Please output the summary only, without any other text.\n\nArticle:\n{input}\n\nSummary:")
}

prompt_length = len(tokenizer(instruction_template[instruction])["input_ids"])
max_sample_length = tokenizer.model_max_length - prompt_length


def preprocess_function(sample, padding="max_length"):
    # create list of samples
    inputs = []

    if n_samples:
        import random
        random.seed(42)
        ind = random.sample(range(0, 13368), n_samples)
    else:
        ind = list(range(0, len(sample[text_column])))

    for i in range(0, len(sample[text_column])):
        if i in ind:
            x = dict()
            x["instruction"] = instruction_template
            x["input"] = sample[text_column][i]
            x["tok_input"] = tokenizer.encode(
                instruction_template[instruction].format_map(x))
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
    file = "cnn_eval.json"
else:
    file = f"sample_cnn_eval_{n_samples}.json"

with open(os.path.join(save_dataset_path, file), "w") as write_f:
    json.dump(
        tokenized_dataset["validation"]["text"], write_f, indent=4, ensure_ascii=False
    )
print("Dataset saved in ", save_dataset_path)
