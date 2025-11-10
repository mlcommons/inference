from datasets import load_dataset
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_folder", type=str, default="dataset")
args = parser.parse_args()

df = load_dataset("di-zhang-fdu/AIME_1983_2024")['train'].to_pandas()

df_1_aime2025 = load_dataset(
    "opencompass/AIME2025",
    "AIME2025-I")['test'].to_pandas()
df_2_aime2025 = load_dataset(
    "opencompass/AIME2025",
    "AIME2025-II")['test'].to_pandas()
# df_aime2025 = pd.concat([df_1_aime2025, df_2_aime2025], ignore_index=True)
# df_aime2025.rename(columns={'answer': 'ground_truth'}, inplace=True)

df.rename(
    columns={
        'Answer': 'ground_truth',
        'Question': 'question'},
    inplace=True)
df.drop(columns=['Year', 'ID', 'Problem Number', 'Part'], inplace=True)
df['dataset'] = 'aime1983'

df.to_pickle(os.path.join(args.dataset_folder, 'aime1983-2024.pkl'))
