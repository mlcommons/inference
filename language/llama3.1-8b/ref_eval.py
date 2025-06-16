from multiprocessing import Pool, cpu_count
from pathlib import Path
import re

import pandas as pd
from rouge_score import rouge_scorer
from tqdm import tqdm

rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


def rouge(label, pred):
    score = rouge_scorer.score(label, pred)
    return {
        'rougeL': 100 * score['rougeL'].fmeasure,
    }


def niah_em(label, pred):
    label_uuids = re.findall(
        r'[\w]{8}-[\w]{4}-[\w]{4}-[\w]{4}-[\w]{12}', label)
    pred_uuids = re.findall(r'[\w]{8}-[\w]{4}-[\w]{4}-[\w]{4}-[\w]{12}', pred)

    # https://github.com/hsiehjackson/RULER/blob/main/scripts/eval/synthetic/constants.py#L28
    score = sum([
        sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)
        for pred, ref in zip(pred_uuids, label_uuids)
    ]) / len(pred_uuids) * 100

    return {'exact_match': round(score, 2)}


def qa_em(label, pred):
    answer_substring = pred

    if 'Answer: ' in pred:
        last_answer_index = pred.rfind("Answer: ")
        if last_answer_index == -1:
            return {'exact_match': 0.0}

        answer_substring = pred[last_answer_index + len("Answer: "):]

    if answer_substring in label:
        return {'exact_match': 100.0}

    normalized_answer = re.sub(r'\s+', '', answer_substring).lower()
    label_entries = [re.sub(r'\s+', '', entry).lower()
                     for entry in label.split('|')]

    match_found = any(entry in normalized_answer for entry in label_entries)
    return {'exact_match': 100.0 if match_found else 0.0}


metrics = {
    fn.__name__: fn
    for fn in [rouge, niah_em, qa_em]
}


def process_row(row):
    metric_fn = metrics[row['metric']]
    metric_eval = metric_fn(row['gt_output'], row['ref_output'])
    return metric_eval


def run_evaluation(df):
    with Pool(cpu_count()) as pool:
        accuracies = list(
            tqdm(
                pool.imap(
                    process_row,
                    df.to_dict('records')),
                total=len(df)))

    df['accuracy'] = accuracies
    return df


if __name__ == '__main__':
    fname = 'dataset/mlperf_llama3_8b_dataset_13368_processed_fp16_eval.pkl'
    df = pd.read_pickle(fname)

    df = run_evaluation(df)
    # df.to_pickle(str(fname).replace(".pkl", "_eval.pkl"))
    print(f"WROTE: {str(fname).replace('.pkl', '_eval.pkl')}")

    accuracy = df.accuracy.apply(pd.Series)
    print(df.dataset.value_counts())
    print(accuracy.describe())
    print(df.describe())
