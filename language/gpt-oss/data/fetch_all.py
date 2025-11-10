from datasets import load_dataset
import pandas as pd
import random
import argparse
import os
import json

parser = argparse.ArgumentParser(description="Fetch and combine AIME, GPQA, and LiveCodeBench datasets")
parser.add_argument("--output_path", type=str, default="./combined_dataset.pkl", help="Full path to output pickle file")
parser.add_argument("--lcb_folder", type=str, default="lcb", help="Folder containing LiveCodeBench repo cloned from HuggingFace")
args = parser.parse_args()

# Ensure output folder exists
output_dir = os.path.dirname(args.output_path)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

random.seed(42)

print("=" * 80)
print("Fetching datasets...")
print("=" * 80)

# ============================================================================
# 1. FETCH AIME DATASET
# ============================================================================
print("\n[1/3] Fetching AIME dataset...")
df_aime = load_dataset("di-zhang-fdu/AIME_1983_2024")['train'].to_pandas()

# Optional: AIME 2025 datasets
# df_1_aime2025 = load_dataset("opencompass/AIME2025", "AIME2025-I")['test'].to_pandas()
# df_2_aime2025 = load_dataset("opencompass/AIME2025", "AIME2025-II")['test'].to_pandas()
# df_aime2025 = pd.concat([df_1_aime2025, df_2_aime2025], ignore_index=True)
# df_aime2025.rename(columns={'answer': 'ground_truth'}, inplace=True)

df_aime.rename(
    columns={
        'Answer': 'ground_truth',
        'Question': 'question'},
    inplace=True)
df_aime.drop(columns=['Year', 'ID', 'Problem Number', 'Part'], inplace=True)
df_aime['dataset'] = 'aime1983'

print(f"   ✓ AIME dataset loaded: {len(df_aime)} samples")

# ============================================================================
# 2. FETCH GPQA DATASET
# ============================================================================
print("\n[2/3] Fetching GPQA dataset...")

# Note: Login using `huggingface-cli login` to access this dataset if needed
ds_diamond = load_dataset("Idavidrein/gpqa", "gpqa_diamond")

# Optional: Other GPQA variants
# ds_experts = load_dataset("Idavidrein/gpqa", "gpqa_experts")
# ds_main = load_dataset("Idavidrein/gpqa", "gpqa_main")
# ds_extended = load_dataset("Idavidrein/gpqa", "gpqa_extended")

df_gpqa = ds_diamond['train'].to_pandas()
# df_experts = ds_experts['train'].to_pandas()
# df_main = ds_main['train'].to_pandas()
# df_extended = ds_extended['train'].to_pandas()

# df = pd.concat([df_diamond, df_main, df_extended], ignore_index=True)
df_gpqa = df_gpqa[['Question',
                    'Correct Answer',
                    'High-level domain',
                    'Incorrect Answer 1',
                    'Incorrect Answer 2',
                    'Incorrect Answer 3']].copy()

# Format questions with multiple choice options
for idx, row in df_gpqa.iterrows():
    options = [str(row[col]) for col in ['Incorrect Answer 1',
                                          'Incorrect Answer 2', 'Incorrect Answer 3']]
    options.append(str(row['Correct Answer']))
    random.shuffle(options)
    answer_idx = options.index(str(row['Correct Answer']))

    options = [option.strip() for option in options]
    answer = chr(65 + answer_idx)

    question = f"{row['Question']}\nA) {options[0]}\nB) {options[1]}\nC) {options[2]}\nD) {options[3]}"
    df_gpqa.loc[idx, 'Question'] = question
    df_gpqa.loc[idx, 'ground_truth'] = answer

df_gpqa.rename(
    columns={
        'High-level domain': 'domain',
        'Question': 'question'},
    inplace=True)
df_gpqa['dataset'] = 'gpqa'

print(f"   ✓ GPQA dataset loaded: {len(df_gpqa)} samples")

# ============================================================================
# 3. FETCH LIVECODEBENCH DATASET
# ============================================================================
print("\n[3/3] Fetching LiveCodeBench dataset...")
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
    if not os.path.exists(file):
        raise FileNotFoundError(f"Error: File not found: {file}")

    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            all_columns.update(row.keys())
            all_rows.append(row)

all_columns = list(all_columns)

df_lcb = pd.DataFrame(all_rows, columns=all_columns)
df_lcb['dataset'] = 'livecodebench'
df_lcb.drop(
    columns=[
        'private_test_cases',
        'metadata',
        'public_test_cases',
        'contest_id',
        'platform',
        'difficulty',
        'contest_date',
        'question_title'],
    inplace=True,
    errors='ignore')  # Use errors='ignore' in case some columns don't exist

starter_prompt = """
### Format: You will use the following starter code to write the solution to the problem and enclose your code within delimiters.
```python
<<starter_code>>
```
"""

for idx, row in df_lcb.iterrows():
    starter_code = row['starter_code']
    starter_prompt_filled = starter_prompt.replace(
        '<<starter_code>>', starter_code)
    df_lcb.loc[idx, 'question'] = df_lcb.loc[idx,
                                              'question_content'] + starter_prompt_filled

df_lcb.rename(columns={'question_id': 'ground_truth'}, inplace=True)

print(f"   ✓ LiveCodeBench dataset loaded: {len(df_lcb)} samples")

# ============================================================================
# 4. COMBINE ALL DATASETS
# ============================================================================
print("\n" + "=" * 80)
print("Combining datasets...")
print("=" * 80)

# Combine all dataframes
df_combined = pd.concat([df_aime, df_gpqa, df_lcb], ignore_index=True)

print(f"\nCombined dataset statistics:")
print(f"  • Total samples: {len(df_combined)}")
print(f"  • AIME samples: {len(df_aime)}")
print(f"  • GPQA samples: {len(df_gpqa)}")
print(f"  • LiveCodeBench samples: {len(df_lcb)}")
print(f"\nDataset columns: {list(df_combined.columns)}")

# Save combined dataset
df_combined.to_pickle(args.output_path)

print(f"\n✓ Combined dataset saved to: {args.output_path}")
print("=" * 80)

