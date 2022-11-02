# coding=utf-8
# Copyright 2021 Arm Limited and affiliates.
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors.
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

import argparse
import collections
import json
import math
import os
import subprocess
import sys

import numpy as np
import pkg_resources
import six
from transformers import BertTokenizer

# To support feature cache.
import pickle

sys.path.insert(0, os.path.dirname(__file__))

installed = {pkg.key for pkg in pkg_resources.working_set}
if "tensorflow" in installed:
    import tensorflow
    sys.path.insert(
        0, os.path.join(
            os.path.dirname(__file__),
            "DeepLearningExamples", "TensorFlow", "LanguageModeling", "BERT"
        )
    )
elif "torch" in installed:
    import torch
    sys.path.insert(
        0, os.path.join(
            os.path.dirname(__file__),
            "DeepLearningExamples", "PyTorch", "LanguageModeling", "BERT"
        )
    )

import tokenization
from create_squad_data import convert_examples_to_features, read_squad_examples

max_seq_length = 384
max_query_length = 64
doc_stride = 128

RawResult = collections.namedtuple(
    "RawResult", ["unique_id", "start_logits", "end_logits"])

dtype_map = {
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64
}


def get_final_text(pred_text, orig_text, do_lower_case):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #     pred_text = steve smith
    #     orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(
        enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file, max_examples=None):
    """Write final predictions to the json file and log-odds of null if needed."""
    print("Writing predictions to: %s" % (output_prediction_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        if max_examples and example_index == max_examples:
            break

        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min mull score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            # FIX: During compliance/audit runs, we only generate a small subset of
            # all entries from the dataset. As a result, sometimes dict retrieval
            # fails because a key is missing.
            # result = unique_id_to_result[feature.unique_id]
            result = unique_id_to_result.get(feature.unique_id, None)
            if result is None:
                continue
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = " ".join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, do_lower_case)
            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        all_predictions[example.qas_id] = nbest_json[0]["text"]

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")


def load_loadgen_log(log_path, eval_features, dtype=np.float32, output_transposed=False):
    with open(log_path) as f:
        predictions = json.load(f)

    results = []
    for prediction in predictions:
        qsl_idx = prediction["qsl_idx"]
        if output_transposed:
            logits = np.frombuffer(bytes.fromhex(
                prediction["data"]), dtype).reshape(2, -1)
            logits = np.transpose(logits)
        else:
            logits = np.frombuffer(bytes.fromhex(
                prediction["data"]), dtype).reshape(-1, 2)
        # Pad logits to max_seq_length
        seq_length = logits.shape[0]
        start_logits = np.ones(max_seq_length) * -10000.0
        end_logits = np.ones(max_seq_length) * -10000.0
        start_logits[:seq_length] = logits[:, 0]
        end_logits[:seq_length] = logits[:, 1]
        results.append(RawResult(
            unique_id=eval_features[qsl_idx].unique_id,
            start_logits=start_logits.tolist(),
            end_logits=end_logits.tolist()
        ))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vocab_file", default="build/data/bert_tf_v1_1_large_fp32_384_v2/vocab.txt", help="Path to vocab.txt")
    parser.add_argument(
        "--val_data", default="build/data/dev-v1.1.json", help="Path to validation data")
    parser.add_argument("--log_file", default="build/logs/mlperf_log_accuracy.json",
                        help="Path to LoadGen accuracy log")
    parser.add_argument("--out_file", default="build/result/predictions.json",
                        help="Path to output predictions file")
    parser.add_argument("--features_cache_file",
                        default="eval_features.pickle", help="Path to features' cache file")
    parser.add_argument("--output_transposed",
                        action="store_true", help="Transpose the output")
    parser.add_argument("--output_dtype", default="float32",
                        choices=dtype_map.keys(), help="Output data type")
    parser.add_argument("--max_examples", type=int,
                        help="Maximum number of examples to consider (not limited by default)")
    args = parser.parse_args()

    output_dtype = dtype_map[args.output_dtype]

    print("Reading examples...")
    eval_examples = read_squad_examples(input_file=args.val_data,
                                        is_training=False, version_2_with_negative=False)

    eval_features = []
    # Load features if cached, convert from examples otherwise.
    cache_path = args.features_cache_file
    if os.path.exists(cache_path):
        print("Loading cached features from '%s'..." % cache_path)
        with open(cache_path, 'rb') as cache_file:
            eval_features = pickle.load(cache_file)
    else:
        print("No cached features at '%s'... converting from examples..." % cache_path)

        print("Creating tokenizer...")
        tokenizer = BertTokenizer(args.vocab_file)

        print("Converting examples to features...")

        def append_feature(feature):
            eval_features.append(feature)

        convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=False,
            output_fn=append_feature,
            verbose_logging=False)

        print("Caching features at '%s'..." % cache_path)
        with open(cache_path, 'wb') as cache_file:
            pickle.dump(eval_features, cache_file)

    print("Loading LoadGen logs...")
    results = load_loadgen_log(
        args.log_file, eval_features, output_dtype, args.output_transposed)

    print("Post-processing predictions...")
    write_predictions(eval_examples, eval_features, results,
                      20, 30, True, args.out_file, args.max_examples)

    print("Evaluating predictions...")
    cmd = "python3 {:}/evaluate_v1.1.py {:} {:} {}".format(
        os.path.dirname(os.path.abspath(__file__)), args.val_data,
        args.out_file, '--max_examples {}'.format(
            args.max_examples) if args.max_examples else '')
    subprocess.check_call(cmd, shell=True)


if __name__ == "__main__":
    main()
