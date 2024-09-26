from dataset import Dataset 
import numpy as np
import json
import nltk

import evaluate
import argparse
from argparse import Namespace


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlperf-accuracy-file", required=True, help="path to mlperf_log_accuracy.json")
    parser.add_argument("--dataset-file", required=True, help="path to cnn_eval.json")
    parser.add_argument(
        "--tokenizer-model",
        default="./data/c4_en_301_5Mexp2_spm.model",
        help="Path to tokenizer model",
    )
    parser.add_argument("--verbose", action="store_true", help="verbose messages")
    args = parser.parse_args()
    return args

def postprocess_text(preds, targets):
    preds = [pred.strip() for pred in preds]
    targets = [target.strip() for target in targets]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    targets = ["\n".join(nltk.sent_tokenize(target)) for target in targets]

    return preds, targets


def main():
    args = get_args()
    dataset_path = args.dataset_file
    metric = evaluate.load("rouge")
    nltk.download('punkt')
    
    dataset_args = Namespace(tokenizer_model = args.tokenizer_model)
    data_object = Dataset(dataset_path, args=dataset_args)

    targets = data_object.targets

    with open(args.mlperf_accuracy_file, "r") as f:
        results = json.load(f)


    target_required = []
    preds_token_ids = []

    for pred in results:
        qsl_idx = pred['qsl_idx']
        target = targets[qsl_idx]
        target_required.append(target)
        preds = np.frombuffer(bytes.fromhex(pred['data']), np.int64).tolist()
        preds = [int(p) for p in preds]
        preds_token_ids.append(preds)
        

    preds_decoded_text = [data_object.tokenizer.detokenize(ids) for ids in preds_token_ids]
    preds, targets = postprocess_text(preds_decoded_text, target_required)


    result = metric.compute(predictions=preds, references=targets, use_stemmer=True,use_aggregator=False)
    result = {k: round(np.mean(v) * 100, 4) for k, v in result.items()}
    prediction_lens = [len(pred) for pred in preds]
    result["gen_len"] = np.sum(prediction_lens)
    result["gen_num"] = len(preds)
    print("\nResults\n")
    print(result)

if __name__ == "__main__":
    main()
