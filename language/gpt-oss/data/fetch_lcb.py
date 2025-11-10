from datasets import load_dataset
import pickle
import os
import glob
import json
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_folder", type=str, default="dataset")
parser.add_argument("--lcb_folder", type=str, default="lcb")
args = parser.parse_args()

files = [
    "test.jsonl",
    "test2.jsonl",
    "test3.jsonl",
    "test4.jsonl",
    "test5.jsonl"]
files = [os.path.join(args.lcb_folder, file) for file in files]

all_rows = []
all_columns = set()

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            all_columns.update(row.keys())
            all_rows.append(row)

all_columns = list(all_columns)

df = pd.DataFrame(all_rows, columns=all_columns)
df['dataset'] = 'livecodebench'
df.drop(
    columns=[
        'private_test_cases',
        'metadata',
        'public_test_cases',
        'contest_id',
        'platform',
        'difficulty',
        'contest_date',
        'question_title'],
    inplace=True)

starter_prompt = """
### Format: You will use the following starter code to write the solution to the problem and enclose your code within delimiters.
```python
<<starter_code>>
```
"""

for idx, row in df.iterrows():
    starter_code = row['starter_code']
    starter_prompt_filled = starter_prompt.replace(
        '<<starter_code>>', starter_code)
    df.loc[idx, 'question'] = df.loc[idx,
                                     'question_content'] + starter_prompt_filled

df.rename(columns={'question_id': 'ground_truth'}, inplace=True)
df.to_pickle(os.path.join(args.dataset_folder, 'lcb.pkl'))
