import pandas as pd
from datasets import load_dataset

ds = load_dataset("cais/mmlu", "all")
df = ds['auxiliary_train'].to_pandas()

breakpoint()

for col in ['subject']:
    df.drop(col, axis=1, inplace=True)

df.rename(columns={'question': 'base_question'}, inplace=True)
df['dataset'] = 'mmlu'

for row in df.itertuples():
    base_question = row.base_question
    options = row.choices
    question = f"{base_question}"
    for idx, option in enumerate(options):
        question += f"\n{chr(65+idx)}) {option}"
    df.loc[row.Index, 'question'] = question
    df.loc[row.Index, 'ground_truth'] = f"{chr(65+row.answer)}"

breakpoint()
df.to_pickle('mmlu.pkl')
