import os
import glob
import json
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_folder", type=str, default="dataset")
parser.add_argument("--healthbench_folder", type=str, default="healthbench")
args = parser.parse_args()

# Find all jsonl files (update the path and pattern to match actual data)
files = glob.glob(os.path.join(args.healthbench_folder, "*.jsonl"))

all_rows = []
all_columns = set()

# First pass: gather all columns
for file in files:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            all_columns.update(row.keys())

all_columns = list(all_columns)

# Second pass: load rows, filling missing keys with None
for file in files:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            row_filled = {col: row.get(col, None) for col in all_columns}
            all_rows.append(row_filled)

# Create DataFrame
df = pd.DataFrame(all_rows, columns=all_columns)
df['dataset'] = 'healthbench'
df.to_pickle(os.path.join(args.dataset_folder, 'healthbench.pkl'))
