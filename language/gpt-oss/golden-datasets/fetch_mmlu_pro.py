import pandas as pd
from datasets import load_dataset

ds = load_dataset("TIGER-Lab/MMLU-Pro")
df_test = ds['test'].to_pandas()
df_validation = ds['validation'].to_pandas()

df = pd.concat([df_validation, df_test], ignore_index=True)

for col in ['answer_index', 'cot_content', 'category', 'src']:
    df.drop(col, axis=1, inplace=True)

df.rename(
    columns={
        'question': 'base_question',
        'answer': 'ground_truth'},
    inplace=True)
df['dataset'] = 'mmlu_pro'

for row in df.itertuples():
    base_question = row.base_question
    options = row.options
    question = f"{base_question}"
    for idx, option in enumerate(options):
        question += f"\n{chr(65+idx)}) {option}"
    df.loc[row.Index, 'question'] = question

breakpoint()
df.to_pickle('mmlu_pro.pkl')
