# experiment config
dataset_id = "cnn_dailymail"
dataset_config = "3.0.0"
text_column = "article"
summary_column = "highlights"

from datasets import load_dataset

import numpy as np
import os
import simplejson as json
import sys

save_dataset_path = os.environ.get("DATASET_CNNDM_PATH", "data")

# Check whether the specified path exists or not
isExist = os.path.exists(save_dataset_path)
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(save_dataset_path)

# Load dataset from the hub
dataset = load_dataset(dataset_id, name=dataset_config)

instruction_template = "summarize: "


def preprocess_function(sample, padding="max_length"):
    # create list of samples
    inputs = []

    for i in range(0, len(sample[text_column])):
        x = dict()
        x["instruction"] = instruction_template
        x["input"] = sample[text_column][i]
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

with open(os.path.join(save_dataset_path, "cnn_eval.json"), "w") as write_f:
    json.dump(
        tokenized_dataset["validation"]["text"], write_f, indent=4, ensure_ascii=False
    )


print("Dataset saved in ", save_dataset_path)
