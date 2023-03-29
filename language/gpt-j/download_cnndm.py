# experiment config
model_id = "EleutherAI/gpt-j-6b"
dataset_id = "cnn_dailymail"
dataset_config = "3.0.0"
save_dataset_path = "data"
text_column = "article"
summary_column = "highlights"

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import numpy as np
import os
import simplejson as json
import sys

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



instruction_template = "Summarize the following news article:"

prompt_length = len(tokenizer(instruction_template)["input_ids"])
max_sample_length = tokenizer.model_max_length - prompt_length


# The maximum total input sequence length after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded.
tokenized_inputs = concatenate_datasets([dataset["train"], dataset["validation"]]).map(lambda x: tokenizer(x[text_column], truncation=True), batched=True, remove_columns=[text_column, summary_column])
max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
max_source_length = min(max_source_length, max_sample_length)


# The maximum total sequence length for target text after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded."
tokenized_targets = concatenate_datasets([dataset["train"], dataset["validation"]]).map(lambda x: tokenizer(x[summary_column], truncation=True), batched=True, remove_columns=[text_column, summary_column])
target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
# use 95th percentile as max target length
max_target_length = int(np.percentile(target_lenghts, 95))



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
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=list(dataset["train"].features))

# save dataset to disk

with open(os.path.join(save_dataset_path,"cnn_eval.json"), 'w') as write_f:
    json.dump(tokenized_dataset["validation"]["text"], write_f, indent=4, ensure_ascii=False)


print("Dataset saved in ",save_dataset_path)

