from datasets import load_dataset
import pandas as pd

df_1 = load_dataset("opencompass/AIME2025", "AIME2025-I")['test'].to_pandas()
df_2 = load_dataset("opencompass/AIME2025", "AIME2025-II")['test'].to_pandas()

df = pd.concat([df_1, df_2], ignore_index=True)

df.rename(columns={'answer': 'ground_truth'}, inplace=True)
df['dataset'] = 'aime2025'

breakpoint()
df.to_pickle('aime2025.pkl')