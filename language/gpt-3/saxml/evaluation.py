from dataset import Dataset
import numpy as np
import json
from rouge_score import rouge_scorer
import os
from typing import List
import argparse
import seqio


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--spm-path", required=True, help="The spm path")
    parser.add_argument("--mlperf-accuracy-file", required=True, help="The mlperf_log_accuracy.json path")
    parser.add_argument("--dataset-path", required=True, help="The dataset path")
    parser.add_argument("--log-dir", required=True, help="The evaluation log dir")
    args = parser.parse_args()
    return args


def compute_rogue_scores(targets: List[str], predictions: List[str]):

    assert len(targets) == len(predictions)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    print("Compute rouge for {} samples".format(len(targets)))

    rogue_scores, r1, r2, rl, rlsum = dict(), list(), list(), list(), list()
    for target, prediction in zip(targets, predictions):
        scores = scorer.score(target, prediction)
        r1.append(scores['rouge1'])
        r2.append(scores['rouge2'])
        rl.append(scores['rougeL'])
        rlsum.append(scores['rougeLsum'])

    rogue_scores['r1_mean'] = np.mean(r1)
    rogue_scores['r2_mean'] = np.mean(r2)
    rogue_scores['rl_mean'] = np.mean(rl)
    rogue_scores['rlsum_mean'] = np.mean(rlsum)

    rogue_scores['gen_len'] = sum([len(prediction) for prediction in predictions])
    rogue_scores['gen_num'] = len(predictions)

    return rogue_scores


def main():

    args = get_args()

    vocabulary = seqio.SentencePieceVocabulary(args.spm_path)
    print("Loading Dataset ... ")
    dataset = Dataset(
        dataset_path=args.dataset_path
    )

    with open(args.mlperf_accuracy_file, "r") as f:
        results = json.load(f)

    targets_pretokenized = []
    pred_texts = []

    output_results = list()
    for result in results:

        output_result = dict()

        qsl_idx = result['qsl_idx']

        pred_token_ids = np.frombuffer(bytes.fromhex(result['data']), np.int64).astype(np.int32).tolist()
        pred_token_ids_str = ','.join([str(i) for i in pred_token_ids])
        pred_text = vocabulary.tokenizer.detokenize(pred_token_ids)
        target_str = ','.join([str(i) for i in list(dataset.targets[qsl_idx])])
        target_pretokenized = dataset.targets_pretokenized[qsl_idx]

        targets_pretokenized.append(dataset.targets_pretokenized[qsl_idx])
        pred_texts.append(pred_text)

        output_result['qsl_idx'] = qsl_idx
        output_result['input_pretokenized'] = dataset.inputs_pretokenized[qsl_idx]
        output_result['target_pretokenized'] = target_pretokenized
        output_result['pred_text'] = pred_text
        output_result['input_str'] = dataset.inputs_str[qsl_idx]
        output_result['target_str'] = target_str
        output_result['pred_token_ids_str'] = pred_token_ids_str
        output_results.append(output_result)

    rogue_scores = compute_rogue_scores(targets=targets_pretokenized, predictions=pred_texts)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    results_file_path = os.path.join(args.log_dir, 'results.json')
    output_results_json = json.dumps(output_results, indent=4)
    with open(results_file_path, "w") as f:
        json.dump(output_results_json, f)

    scores_file_path = os.path.join(args.log_dir, 'scores.json')
    with open(scores_file_path, "w") as f:
        json.dump(rogue_scores, f, indent=4)


if __name__ == "__main__":
    main()
