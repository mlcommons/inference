# coding=utf-8
# Copyright 2023 AMD
#

import array
import json
import os
import sys

sys.path.insert(0, os.getcwd())

import mlperf_loadgen as lg
import numpy as np
import migraphx
from squad_QSL import get_squad_QSL


class BERT_migraphx_SUT:
    def __init__(self, args):

        print("Loading ONNX model...")
        self.quantized = args.quantized
        self.batch_size = args.batch_size
        self.model_path = args.model
        self.model = migraphx.parse_onnx(
            self.model_path, default_dim_value=self.batch_size
        )
        print("Quantize to fp16...")
        migraphx.quantize_fp16(self.model)
        print("Compile for gpu...")
        self.model.compile(migraphx.get_target("gpu"))

        print("Constructing SUT...")
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        print("Finished constructing SUT.")

        print("Warmup...")
        self.fd = {
            "input_ids": np.zeros((self.batch_size, 384), dtype=np.int64),
            "input_mask": np.zeros((self.batch_size, 384), dtype=np.int64),
            "segment_ids": np.zeros((self.batch_size, 384), dtype=np.int64),
        }
        self.model.run(self.fd)

        self.qsl = get_squad_QSL(args)

    def issue_queries(self, query_samples):
        idx = [q.index for q in query_samples]
        query_id = [q.id for q in query_samples]
        bs = self.batch_size
        for i in range(0, len(idx), bs):
            eval_features = self.qsl.get_features(idx[i : i + bs])
            actual_batchsize = len(eval_features)
            self.fd["input_ids"][:actual_batchsize] = np.array(
                [eval_feature.input_ids for eval_feature in eval_features]
            ).astype(np.int64)
            self.fd["input_mask"][:actual_batchsize] = np.array(
                [eval_feature.input_mask for eval_feature in eval_features]
            ).astype(np.int64)
            self.fd["segment_ids"][:actual_batchsize] = np.array(
                [eval_feature.segment_ids for eval_feature in eval_features]
            ).astype(np.int64)
            results = self.model.run(self.fd)
            scores = [np.array(result) for result in results]
            outputs = np.stack(scores, axis=-1)

            response_array_refs = []
            response = []
            for i, qid in enumerate(query_id[i : i + bs]):
                response_array = array.array(
                    "B", np.array(outputs[i], np.float32).tobytes()
                )
                response_array_refs.append(response_array)
                bi = response_array.buffer_info()
                response.append(lg.QuerySampleResponse(qid, bi[0], bi[1]))
            lg.QuerySamplesComplete(response)

    def flush_queries(self):
        pass

    def __del__(self):
        print("Finished destroying SUT.")

def get_migraphx_sut(args):
    return BERT_migraphx_SUT(args)
