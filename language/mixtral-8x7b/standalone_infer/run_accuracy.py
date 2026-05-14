#!/usr/bin/env python3
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import re
import numpy as np
import argparse
import evaluate
import nltk
from tqdm import tqdm

import timeit
import multiprocessing
import json
import pickle
import queue

from mxeval.execution import check_correctness as check_correctness_python
from mxeval.execution import (
    check_correctness_cpp,
    check_correctness_csharp,
    check_correctness_go,
    check_correctness_java,
    check_correctness_javascript,
    check_correctness_kotlin,
    check_correctness_perl,
    check_correctness_php,
    check_correctness_ruby,
    check_correctness_scala,
    check_correctness_swift,
    check_correctness_typescript,
)

nltk.download("punkt")
nltk.download("punkt_tab")
metric = evaluate.load("rouge")


def calculate_rouge_score(model_outputs, ref_outputs):
    metric = evaluate.load("rouge")
    m_preds = [pred.strip() for pred in model_outputs]
    m_targets = [target.strip() for target in ref_outputs]

    # rougeLSum expects newline after each sentence
    m_preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in m_preds]
    m_targets = ["\n".join(nltk.sent_tokenize(target)) for target in m_targets]
    m_result = metric.compute(
        predictions=m_preds, references=m_targets, use_stemmer=True, use_aggregator=False
    )
    m_rouge_result = {k: round(np.mean(v) * 100, 4)
                      for k, v in m_result.items()}

    return m_rouge_result


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


def postprocess_golang(code: str) -> str:
    multi_line_imports = re.compile(
        r"^import \(\n(.+)((?:\n.+)+)\n\)", re.MULTILINE)
    line_imports = re.compile(r"^import \".*\"")
    func_main = re.compile(r"^func main.*^}", re.MULTILINE | re.DOTALL)

    code = code.replace("package main", "")  # Remove package main
    code = multi_line_imports.sub("", code)
    code = line_imports.sub("", code)
    code = func_main.sub("", code)

    return code


def postprocess_scala(code: str) -> str:
    code = code.replace("object Main extends App {", "")
    code = "".join(code.splitlines(True)[:-1])
    return code


def postprocess_python(code: str) -> str:
    return code.lstrip()


def worker(inp_queue, out_queue):
    while True:
        try:
            problem = inp_queue.get(timeout=5)
        except queue.Empty:
            break

        key = f"{problem['lang']}_{problem['entry_point']}"
        checker = eval(f"check_correctness_{problem['lang']}")

        problem["task_id"] = key
        problem["test"] = problem["test_code"]

        solution = problem["response"]

        try:
            solution = solution[:solution.index("```")]
        except ValueError:
            # Happens when a code block isn't closed properly
            pass

        if problem["lang"] == "go":
            solution = postprocess_golang(solution)
        elif problem["lang"] == "python":
            solution = postprocess_python(solution)
        elif problem["lang"] == "scala":
            solution = postprocess_scala(solution)

        # Mixtral likes escaping underscores for some reason, so let's remove
        # these
        solution = solution.replace("\\_", "_")

        # The evaluation script evaluates `code = prompt + solution + tests`
        # But Mixtral regenerates the prompt in its output, so we should remove
        # this
        problem["prompt"] = ""

        result = checker(problem, solution, timeout=20.0)
        out_queue.put(
            (key,
             problem["lang"],
             result["passed"],
                result["result"],
                problem["response"]))


def convert_pickle(df: pd.DataFrame, result_keys: dict):
    problems = []
    for _, row in df.iterrows():
        lang, entry_point = row["id"].split("_", 1)
        problems.append({
            "lang": lang,
            "prompt": row["input"],
            "test_code": row["gt_output"],
            "entry_point": entry_point,
            "response": row[f"{result_keys['result']}"]
        })
    return problems


def evaluate_mbxp(n_works: int, df: pd.DataFrame, result_keys: dict):
    print(f"Evaluating MBXP score...")
    # Convert pickle file into dictionary
    results = convert_pickle(df, result_keys)

    by_lang = {}
    for problem in results:
        by_lang.setdefault(problem["lang"], []).append(problem)

    inp_queue = multiprocessing.Queue()
    out_queue = multiprocessing.Queue()

    n_problems = 0

    for lang, problems in by_lang.items():
        if lang not in ["cpp", "python", "php",
                        "javascript", "ruby", "typescript"]:
            raise RuntimeError(f"{lang} not in supported list.")

        n_problems += len(problems)
        for problem in problems:
            inp_queue.put(problem)

    start = timeit.default_timer()
    workers = []
    for _ in range(args.n_workers):
        w = multiprocessing.Process(target=worker, args=(inp_queue, out_queue))
        w.start()
        workers.append(w)

    passes = {}
    n_passed = 0
    lang_passed = {}
    lang_counts = {}
    for i in tqdm(range(n_problems)):
        key, lang, passed, result, response = out_queue.get()
        passes[key] = {
            "passed": passed,
            "result": result,
            "response": response}
        n_passed += passed

        lang_passed.setdefault(lang, 0)
        lang_passed[lang] += passed

        lang_counts.setdefault(lang, 0)
        lang_counts[lang] += 1

    end = timeit.default_timer()
    print(f"Processed {n_problems} in {end - start}s")
    print(f"{100 * n_passed / n_problems : .02f}% pass@1")
    print(lang_passed, " out of ", lang_counts)

    gen_token_len = df[result_keys['length']].tolist()
    gen_token_per_sample = sum(gen_token_len) / len(gen_token_len)
    print(f"gen_tokens_per_sample: {gen_token_per_sample}")

    # with open("evaluated_test.json", "w") as f:
    #    json.dump(passes, f, indent=2)

    return n_passed / n_problems


def evaluate_openorca(df: pd.DataFrame, result_keys: dict):
    print(f"Evaluating OpenOrca score...")
    gen_output = df[f"{result_keys['result']}"].tolist()
    gt_output = df.gt_output.tolist()
    score = calculate_rouge_score(gen_output, gt_output)
    gen_token_len = df[result_keys['length']].tolist()
    gen_token_per_sample = sum(gen_token_len) / len(gen_token_len)
    print(
        f"OpenOrca score: {score}, gen_token_per_sample: {gen_token_per_sample}")
    return score


def evaluate_gsm8k(df: pd.DataFrame, result_keys: dict):
    print(f"Evaluating GSM8K score...")
    gen_output = df[f"{result_keys['result']}"].tolist()
    gt_numbers = df.gt_output.tolist()
    gen_nums = [maybe_remove_comma(find_number(msg.split("\nQ:")[0]))
                for msg in gen_output]
    correct = 0
    total = len(gt_numbers)
    for idx in range(len(gt_numbers)):
        ref = try_float(gt_numbers[idx])
        tgt = try_float(gen_nums[idx])
        if tgt is None:
            continue
        correct += (ref == tgt)

    em = correct / total
    gen_token_len = df[result_keys['length']].tolist()
    gen_token_per_sample = sum(gen_token_len) / len(gen_token_len)
    print(
        f"EM: {em}, correct: {correct} / {total}, gen_token_per_sample: {gen_token_per_sample}")
    return em


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_workers",
        type=int,
        default=10,
        help="The number of processes to use")
    parser.add_argument("--results_path", type=str, default="mixtral_8x7b_15000_greedy_reference_fp16_mintoken2.pkl",
                        help="The path to the results file pickle file")
    parser.add_argument("--result_key", type=str, default="ref_output",
                        help="ref output dict key")
    parser.add_argument("--length_key", type=str, default="tok_ref_output_len",
                        help="ref output dict key")
    args = parser.parse_args()

    """
    Sample command:
    python3 nv_accuracy.py --results_path=trtllm_fp16_mixtral_8x7b_all15k_15000_BS128_greedy_06102024.pkl --result_key=nv_tllm_ref_output --length_key=nv_tllm_tok_ref_output_length
    """

    result_keys = {
        "result": args.result_key,
        "length": args.length_key
    }

    """
    dataset                                                            MBXP (OpenOrca/GSM8K)
    id                                            typescript_minimum_Length
    question              /**\n * Write a typescript function to minimiz...
    input                 <s> [INST] Complete the following code. Be con...
    ref_output            \nconst minimumLength = (s: string): number =>...
    gt_output             \nimport * as assert from 'assert'\n\nlet actu...
    tok_input             [1, 1, 28705, 733, 16289, 28793, 21929, 272, 2...
    tok_ref_output        [13, 1978, 7968, 4645, 327, 325, 28713, 28747,...
    stop_sequence                                                   \n```\n
    tok_stop_sequence                                [13, 13940, 28832, 13]
    tok_input_len                                                       139
    tok_ref_output_len                                                  123
    """

    df = pd.read_pickle(args.results_path)
    df_gsm8k = df[df['dataset'] == "GSM8K"].copy()
    evaluate_gsm8k(df_gsm8k, result_keys)
    df_openorca = df[df['dataset'] == "OpenOrca"].copy()
    evaluate_openorca(df_openorca, result_keys)
    df_mbxp = df[df['dataset'] == "MBXP"].copy()
    evaluate_mbxp(args.n_workers, df_mbxp, result_keys)
