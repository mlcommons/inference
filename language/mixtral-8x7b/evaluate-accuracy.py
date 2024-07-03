import argparse
from transformers import AutoTokenizer
import nltk
import evaluate
import numpy as np
import pandas as pd
import json
import re


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", required=True,
                        help="Path to Llama2-70b-hf-chat checkpoint")
    parser.add_argument("--mlperf-accuracy-file", required=True,
                        help="path to mlperf_log_accuracy.json")
    parser.add_argument("--dataset-file", required=True,
                        help="path to processed validation dataset")
    parser.add_argument("--n_workers", default=2, type=int,
                        help="Number of workers used for the MBXP evaluation")
    parser.add_argument("--verbose", action="store_true",
                        help="verbose messages")
    parser.add_argument("--dtype", default="int64",
                        help="dtype of the accuracy log", choices=["int32", "int64", "float"])
    args = parser.parse_args()
    return args


def get_groundtruth(processed_dataset_file):
    data = pd.read_pickle(processed_dataset_file)
    return data


# Functions for evaluating GSM8K
def find_numbers(x: str) -> list[str]:
    """Finds all numbers in a string."""
    # Search for number, possibly negative (hyphen), with thousand separators
    # (comma), and with a decimal point (period inbetween digits).
    numbers = re.compile(
        r'-?[\d,]*\.?\d+',
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    ).findall(x)
    return numbers


def find_number(x: str,
                answer_delimiter: str = 'The answer is') -> str:
    """Finds the most relevant number in a string."""
    # If model uses the answer delimiter, then select the first number following
    # that format.
    if answer_delimiter in x:
        answer = x.split(answer_delimiter)[-1]
        numbers = find_numbers(answer)
        if numbers:
            return numbers[0]

    # In general, select the last number in the string.
    numbers = find_numbers(x)
    if numbers:
        return numbers[-1]
    return ''


def maybe_remove_comma(x: str) -> str:
    # Example: 5,600 -> 5600
    return x.replace(',', '')


def try_float(x: str):
    try:
        ret = float(x)
    except BaseException:
        ret = None
    return ret

# Functions for evaluating OpenOrca


def postprocess_text(preds, targets):
    preds = [pred.strip() for pred in preds]
    targets = [target.strip() for target in targets]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    targets = ["\n".join(nltk.sent_tokenize(target)) for target in targets]

    return preds, targets

# Functions for MBXP


def create_mbxp_dict(row, response):
    lang, entry_point = row["id"].split("_", 1)
    return {
        "lang": lang,
        "prompt": row["input"],
        "test_code": row["gt_output"],
        "entry_point": entry_point,
        "response": response
    }


def main():

    args = get_args()
    dataset_path = args.dataset_file
    checkpoint_path = args.checkpoint_path
    metric = evaluate.load("rouge")
    nltk.download('punkt')

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        model_max_length=2048,
        padding_side="left",
        use_fast=False,)

    data = get_groundtruth(args.dataset_file)
    query_types, gt_outputs = data["dataset"], data["gt_output"]

    target_required_GSM8K = []
    target_required_OpenOrca = []
    results_MBXP = []
    preds_token_GSM8K = []
    preds_token_OpenOrca = []
    preds_token_MBXP = []

    eval_dtype = np.int64
    if args.dtype == "int32":
        eval_dtype = np.int32
    elif args.dtype == "float":
        eval_dtype = np.float32

    with open(args.mlperf_accuracy_file, "r") as f:
        results = json.load(f)

    seen = set()
    gen_tok_len = 0
    gen_num = 0
    for pred in results:
        gen_num += 1
        qsl_idx = pred['qsl_idx']
        if qsl_idx in seen:
            continue

        seen.add(qsl_idx)

        query_type = query_types.iloc[qsl_idx]
        if query_type == "GSM8K":
            target = gt_outputs.iloc[qsl_idx]
            target_required_GSM8K.append(target)
            pred = np.frombuffer(bytes.fromhex(pred['data']), eval_dtype)

            gen_tok_len += len(pred)
            preds_token_GSM8K.append(pred)
        elif query_type == "OpenOrca":
            target = gt_outputs.iloc[qsl_idx]
            target_required_OpenOrca.append(target)
            pred = np.frombuffer(bytes.fromhex(pred['data']), eval_dtype)

            gen_tok_len += len(pred)
            preds_token_OpenOrca.append(pred)
        else:
            target = data.iloc[qsl_idx]
            pred = np.frombuffer(bytes.fromhex(pred['data']), eval_dtype)
            pred_str = tokenizer.decode(pred, skip_special_tokens=True)
            results_MBXP.append(create_mbxp_dict(target, pred_str))

            gen_tok_len += len(pred)

    # OpenOrca metric
    preds_decoded_text = tokenizer.batch_decode(
        preds_token_OpenOrca, skip_special_tokens=True)

    preds, targets = postprocess_text(
        preds_decoded_text, target_required_OpenOrca)
    
    if preds:
        result = metric.compute(
            predictions=preds, references=targets, use_stemmer=True, use_aggregator=False)
        result = {k: float(round(np.mean(v) * 100, 4)) for k, v in result.items()}
        prediction_lens = [len(pred) for pred in preds]

    else:
        result = {}
        prediction_lens = []

    # GSM8K metric
    preds_decoded_text = tokenizer.batch_decode(
        preds_token_GSM8K, skip_special_tokens=True)
    pred_nums = [
        maybe_remove_comma(
            find_number(
                pred_text.split("\nQ:")[0])) for pred_text in preds_decoded_text]
    gsm8k_total = len(target_required_GSM8K)
    correct = 0
    for idx in range(len(target_required_GSM8K)):
        ref = try_float(target_required_GSM8K[idx])
        tgt = try_float(pred_nums[idx])
        if tgt is None:
            continue
        correct += (ref == tgt)

    result['gsm8k'] = 100.0 * correct / gsm8k_total

    # MBXP metric
    from evaluate_mbxp import evaluate_mbxp

    if results_MBXP:
        result['mbxp'] = evaluate_mbxp(results_MBXP, args.n_workers)
    else:
        result['mbxp'] = 0

    result = {
        **result,
        'gen_len': int(np.sum(prediction_lens)),
        'gen_num': gen_num,
        'gen_tok_len': gen_tok_len,
        'tokens_per_sample': round(gen_tok_len / gen_num, 1),
    }

    print("\nResults\n")
    print(result)


if __name__ == "__main__":
    main()
