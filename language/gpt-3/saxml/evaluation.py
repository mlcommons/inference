from dataset import Dataset
import numpy as np
import json
from rouge_score import rouge_scorer
import os

import argparse
import seqio


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--spm-path", default="gs://cnn_dailymail_public/mlperf/vocab/c4_en_301_5Mexp2_spm.model", help="spm path")
    parser.add_argument("--mlperf-accuracy-file", required=True, help="path to mlperf_log_accuracy.json")
    parser.add_argument("--dataset-path", default="gs://cnn_dailymail_public/mlperf/tokenized_cnn_dailymail_3.0.0/cnn_dailymail-validation.tfrecord-00000-of-00001", help="")
    parser.add_argument("--log-dir", default="/mlperf_inference/language/gpt-3/saxml/evaluation_logs", help="log path")
    args = parser.parse_args()
    return args


def compute_rogue(targets, predictions):

    assert len(targets) == len(predictions)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    print("Compute rouge for {} samples".format(len(targets)))

    r1, r2, rl, rlsum = [], [], [], []
    for target, prediction in zip(targets, predictions):
        scores = scorer.score(target, prediction)
        r1.append(scores['rouge1'])
        r2.append(scores['rouge2'])
        rl.append(scores['rougeL'])
        rlsum.append(scores['rougeLsum'])

    r1_mean = np.mean(r1)
    r2_mean = np.mean(r2)
    rl_mean = np.mean(rl)
    rlsum_mean = np.mean(rlsum)

    return r1_mean, r2_mean, rl_mean, rlsum_mean


def log_output(targets, predictions, output_file):
    output = dict()
    r1, r2, r, rlsum = compute_rogue(targets, predictions)
    output['rouge1'] = r1
    output['rouge2'] = r2
    output['rouge'] = r
    output['rougeLsum'] = rlsum
    output["gen_num"] = len(predictions)
    json.dump(output, open(output_file, 'w'), indent=4)


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
    preds_token_ids = []
    preds_text = []

    for result in results:
        qsl_idx = result['qsl_idx']
        target_pretokenized = dataset.targets_pretokenized[qsl_idx]
        targets_pretokenized.append(target_pretokenized)
        pred_token_ids = np.frombuffer(bytes.fromhex(result['data']), np.int64)
        print('pred_token_ids: ', pred_token_ids)
        pred_token_ids = pred_token_ids.astype(np.int32).tolist()
        preds_token_ids.append(pred_token_ids)
        pred_text = vocabulary.tokenizer.detokenize(pred_token_ids)
        preds_text.append(pred_text)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    log_file_path = os.path.join(args.log_dir, 'evaluation.json')
    log_output(targets_pretokenized, preds_text, log_file_path)


if __name__ == "__main__":
    main()
