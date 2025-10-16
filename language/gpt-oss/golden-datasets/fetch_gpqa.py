from datasets import load_dataset
import pandas as pd
import random
random.seed(42)

# Login using e.g. `huggingface-cli login` to access this dataset
ds_diamond = load_dataset("Idavidrein/gpqa", "gpqa_diamond")
ds_experts = load_dataset("Idavidrein/gpqa", "gpqa_experts")
ds_main = load_dataset("Idavidrein/gpqa", "gpqa_main")
ds_extended = load_dataset("Idavidrein/gpqa", "gpqa_extended")

df_diamond = ds_diamond['train'].to_pandas()
# df_experts = ds_experts['train'].to_pandas()
# df_main = ds_main['train'].to_pandas()
# df_extended = ds_extended['train'].to_pandas()


# df = pd.concat([df_diamond, df_main, df_extended], ignore_index=True)
df = df_diamond
df = df[['Question', 'Correct Answer', 'High-level domain', 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3']]

for idx, row in df.iterrows():
    options = [str(row[col]) for col in ['Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3']]
    options.append(str(row['Correct Answer']))
    random.shuffle(options)
    answer_idx = options.index(str(row['Correct Answer']))

    options = [option.strip() for option in options]
    answer = chr(65 + answer_idx)

    question = f"{row['Question']}\nA) {options[0]}\nB) {options[1]}\nC) {options[2]}\nD) {options[3]}"
    df.loc[idx, 'Question'] = question
    df.loc[idx, 'ground_truth'] = answer

df.rename(columns={'High-level domain': 'domain', 'Question': 'question'}, inplace=True)
df['dataset'] = 'gpqa'

breakpoint()
df.to_pickle('gpqa.pkl')