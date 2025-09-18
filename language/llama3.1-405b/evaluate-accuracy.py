import argparse
from transformers import AutoTokenizer
import nltk
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
import re
from rouge_score import rouge_scorer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-path",
        default="meta-llama/Meta-Llama-3-8B",
        help="Path to Llama3.1-405b-hf-chat checkpoint"
    )
    parser.add_argument(
        "--mlperf-accuracy-file", required=True, help="path to mlperf_log_accuracy.json"
    )
    parser.add_argument(
        "--dataset-file",
        required=True,
        help="path to processed dataset set",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose messages")
    parser.add_argument(
        "--dtype",
        default="int64",
        help="dtype of the accuracy log",
        choices=["int32", "int64", "float"],
    )
    args = parser.parse_args()
    return args


scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


def rouge(label, pred):
    score = scorer.score(label, pred)
    return {
        'rougeL': 100 * score['rougeL'].fmeasure,
    }


def niah_em(label, pred):
    label_uuids = re.findall(
        r'[\w]{8}-[\w]{4}-[\w]{4}-[\w]{4}-[\w]{12}', label)
    pred_uuids = re.findall(r'[\w]{8}-[\w]{4}-[\w]{4}-[\w]{4}-[\w]{12}', pred)

    if len(pred_uuids) == 0:
        return {'exact_match': 0.0}

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


def get_groundtruth(processed_dataset_file, return_metrics=True):
    data = pd.read_pickle(processed_dataset_file)
    ground_truths = data["gt_output"]
    if return_metrics:
        metrics = data["metric"]
        return ground_truths, metrics
    return ground_truths


def postprocess_text(preds, targets):
    preds = [pred.strip() for pred in preds]
    targets = [target.strip() for target in targets]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    targets = ["\n".join(nltk.sent_tokenize(target)) for target in targets]

    return preds, targets


def process_item(item):
    pred, target, metric = item
    metric_fn = metrics[metric]
    metric_eval = metric_fn(target, pred)
    return metric_eval


def run_evaluation(preds, targets, metrics, n_process=None):
    n_process = cpu_count() if n_process is None else n_process
    with Pool(n_process) as pool:
        accuracies = list(
            tqdm(
                pool.imap(
                    process_item, zip(
                        preds, targets, metrics)), total=len(preds)))
    df = pd.DataFrame({"accuracy": accuracies, "metric": metrics})
    return df.accuracy.apply(pd.Series).describe().loc["mean"].to_dict()


def main():

    args = get_args()
    dataset_path = args.dataset_file
    checkpoint_path = args.checkpoint_path
    nltk.download("punkt")
    nltk.download('punkt_tab')

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        model_max_length=22000,
        padding_side="left",
        use_fast=False,
    )

    targets, metrics = get_groundtruth(args.dataset_file)

    target_required = []
    metrics_required = []
    preds_token_ids = []

    eval_dtype = np.int64
    if args.dtype == "int32":
        eval_dtype = np.int32
    elif args.dtype == "float":
        eval_dtype = np.float32

    with open(args.mlperf_accuracy_file, "r") as f:
        results = json.load(f)

    seen = set()
    gen_tok_len = 0
    for pred in results:
        qsl_idx = pred["qsl_idx"]
        if qsl_idx in seen:
            continue

        seen.add(qsl_idx)
        target_required.append(targets[qsl_idx])
        metrics_required.append(metrics[qsl_idx])
        pred = np.frombuffer(bytes.fromhex(pred["data"]), eval_dtype)

        gen_tok_len += len(pred)
        preds_token_ids.append(pred)

    preds_decoded_text = tokenizer.batch_decode(
        preds_token_ids, skip_special_tokens=True
    )

    preds, targets = postprocess_text(preds_decoded_text, target_required)

    result = run_evaluation(preds, targets, metrics_required)
    result = dict(result)
    prediction_lens = [len(pred) for pred in preds]
    gen_num = len(preds)

    result = {
        **result,
        "gen_len": np.sum(prediction_lens),
        "gen_num": gen_num,
        "gen_tok_len": gen_tok_len,
        "tokens_per_sample": round(gen_tok_len / gen_num, 1),
    }

    print("\nResults\n")
    print(result)


if __name__ == "__main__":
    main()
