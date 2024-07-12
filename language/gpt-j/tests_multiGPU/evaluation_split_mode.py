from dataset import Dataset
import os
from pathlib import Path
import time
import numpy as np
import json
import nltk
import array
import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader
import evaluate
import argparse
import nltk
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml,pdb

def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--first_log_folder_path", required=True,
                        help="path to mlperf_log_accuracy.json")
    parser.add_argument("--dataset-file", default="./data/cnn_eval.json",
                        help="path to cnn_eval.json")
    parser.add_argument("--verbose", action="store_true",
                        help="verbose messages")
    parser.add_argument("--dtype", default="int64",
                        help="dtype of the accuracy log", choices=["int32", "int64"])
    parser.add_argument("--num_splits", type=int, default=1, 
                        help="")
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
    model_name = "EleutherAI/gpt-j-6B"
    dataset_path = args.dataset_file
    metric = evaluate.load("rouge")
    nltk.download('punkt')

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=2048,
        padding_side="left",
        use_fast=False,)
    tokenizer.pad_token = tokenizer.eos_token

    data_object = Dataset(dataset_path)
    targets = data_object.targets

    default_log_name="mlperf_log_accuracy.json"
    results_path_list = []
    
    first_log_folder_path=Path(args.first_log_folder_path)
    folder_name = first_log_folder_path.stem
    # {TEST_DATE}_{DATASET}_{START_IDX} 로 입력받은 폴더 위치에서 전체 폴더를 추출함
    # 0711_cnn_eval_0 을 입력 받고 log 가 포함된 0711_cnn_eval_0, 0711_cnn_eval_1, 0711_cnn_eval_2... 폴더 path list 생성
    for idx in range(args.num_splits):
        new_folder_name = f"{folder_name[:-2]}_{idx}" 
        results_path_list.append(first_log_folder_path.with_stem(new_folder_name))
        
    

    results = []

    for result_path in results_path_list:
        with open(os.path.join(result_path, default_log_name), "r") as f:
            result = json.load(f)
            results.append(result)
        

    n_splited_data=len(results[0])

    # Deduplicate the results loaded from the json
    dedup_results = []
    seen = set()
    for idx, split_result in enumerate(results):
        for result in split_result:
            item = result['qsl_idx'] + n_splited_data*idx
            if item not in seen:
                seen.add(item)
                result['qsl_idx'] = item
                dedup_results.append(result)



    results = dedup_results      

    target_required = []
    preds_token_ids = []

    eval_dtype = np.int64
    if args.dtype == "int32":
        eval_dtype = np.int32

    for pred in results:
        qsl_idx = pred['qsl_idx']
        target = targets[qsl_idx]
        target_required.append(target)
        preds_token_ids.append(np.frombuffer(
            bytes.fromhex(pred['data']), eval_dtype))

    preds_decoded_text = tokenizer.batch_decode(
        preds_token_ids, skip_special_tokens=True)

    preds, targets = postprocess_text(preds_decoded_text, target_required)

    result = metric.compute(
        predictions=preds, references=targets, use_stemmer=True, use_aggregator=False)
    name = dataset_path.split('.')[1].split('/')[-1]
    # with open(f'result_qlevel_4_{name}.yaml', 'w') as file:
    #     yaml.dump(result, file)
    result = {k: round(np.mean(v) * 100, 4) for k, v in result.items()}
    prediction_lens = [len(pred) for pred in preds]
    result["gen_len"] = np.sum(prediction_lens)
    result["gen_num"] = len(preds)
    print("\nResults\n")
    print(result)


if __name__ == "__main__":
    main()
