import array
import json
import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.getcwd(), "DeepLearningExamples", "PyTorch", "LanguageModeling", "BERT"
    ),
)
sys.path.insert(0, os.getcwd())

import json
from pathlib import Path

import mlperf_loadgen as lg
import numpy as np
import torch
import transformers
from furiosa_llm_models.bert.symbolic.mlperf_submission import \
    BertForQuestionAnswering
from pytorch_SUT import BERT_PyTorch_SUT
from RNGD_encoder import BertMLPerfSubmissionEncoder, stack_tensors
from squad_QSL import get_squad_QSL
from torch.fx import GraphModule
from transformers import BertConfig

import tqdm

BUCKET_SIZE = 384
PAD_TOKEN_ID: int = 0  # EOS token


class BERT_RNGD_SUT(BERT_PyTorch_SUT):
    def __init__(self, args):
        print("Loading BERT configs...")
        config_path = Path(__file__).parent.joinpath("bert_config.json")
        with open(config_path, "r") as f:
            config_json = json.load(f)

        config = BertConfig(
            attention_probs_dropout_prob=config_json["attention_probs_dropout_prob"],
            hidden_act=config_json["hidden_act"],
            hidden_dropout_prob=config_json["hidden_dropout_prob"],
            hidden_size=config_json["hidden_size"],
            initializer_range=config_json["initializer_range"],
            intermediate_size=config_json["intermediate_size"],
            max_position_embeddings=config_json["max_position_embeddings"],
            num_attention_heads=config_json["num_attention_heads"],
            num_hidden_layers=config_json["num_hidden_layers"],
            type_vocab_size=config_json["type_vocab_size"],
            vocab_size=config_json["vocab_size"],
        )

        self.network = args.network
        self.dev = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.version = transformers.__version__

        self.dump_path = args.dump_path
        if not self.dump_path.exists():
            with open(self.dump_path, "w") as f:
                json.dump([], f)
        self.dump = {}

        print("Loading PyTorch model...")
        self.model = BertForQuestionAnswering(config)
        self.model.to(self.dev)
        self.model.eval()
        model_file = os.environ.get(
            "ML_MODEL_FILE_WITH_PATH",
            "build/data/bert_tf_v1_1_large_fp32_384_v2/model.pytorch",
        )
        self.model.load_state_dict(torch.load(model_file), strict=False)

        if args.quantize:
            from quantization import quantize_model
            from quantization.utils import random_seed, set_optimization

            random_seed()
            set_optimization(args.torch_numeric_optim)

            if not args.gpu:
                raise ValueError(
                    "Inference on a device other than GPU is not supported yet."
                )
            traced_model = self.model.trace()
            self.model = quantize_model(
                traced_model,
                args.quant_param_path,
                args.quant_format_path,
            )
        else:
            self.model = self.model.trace()

        self.encoder = BertMLPerfSubmissionEncoder(
            self.model, bucket_size=BUCKET_SIZE, pad_token_id=PAD_TOKEN_ID
        )
        print("Constructing SUT...")
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        print("Finished constructing SUT.")

        self.qsl = get_squad_QSL(args.max_examples)

    def issue_queries(self, query_samples):
        for i in tqdm.tqdm(range(len(query_samples)), unit="queries"):
            eval_features = self.qsl.get_features(query_samples[i].index)
            if self.dump_path:
                self.dump.update({"qsl_idx": query_samples[i].index})
            self.process_sample(eval_features, query_samples[i].id)

            if self.dump_path:
                with open(self.dump_path, "r") as f:
                    data = json.load(f)

                data.append(self.dump)
                data = sorted(data, key=lambda x: x["qsl_idx"])

                with open(self.dump_path, "w") as f:
                    json.dump(data, f)

    def process_sample(self, sample_input, query_id=None):
        if self.network == "sut":
            input_ids = sample_input["input_ids"]
            input_mask = sample_input["input_mask"]
            segment_ids = sample_input["segment_ids"]
        else:
            input_ids = sample_input.input_ids
            input_mask = sample_input.input_mask
            segment_ids = sample_input.segment_ids

        query = {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids,
        }

        if self.dump_path:
            self.dump.update({"input": query})

        with torch.no_grad():
            model_output = self.encoder.encode(
                input_ids=torch.LongTensor(input_ids).unsqueeze(0).to(self.dev),
                attention_mask=torch.LongTensor(input_mask).unsqueeze(0).to(self.dev),
                token_type_ids=torch.LongTensor(segment_ids).unsqueeze(0).to(self.dev),
            )
            if self.dump_path:
                assert len(model_output) == 1
                self.dump.update({"output": {"output_ids": model_output[0].tolist()}})

            input_length = torch.LongTensor(input_ids).unsqueeze(0).shape[-1]
            output = stack_tensors(model_output, max_shape=[input_length, 2])
            start_logits, end_logits = output.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1).contiguous()
            end_logits = end_logits.squeeze(-1).contiguous()

            output = (
                torch.stack([start_logits, end_logits], axis=-1)
                .squeeze(0)
                .cpu()
                .numpy()
            )
            if self.network == "sut":
                return output.tolist()

            response_array = array.array("B", output.tobytes())
            bi = response_array.buffer_info()
            response = lg.QuerySampleResponse(query_id, bi[0], bi[1])
            lg.QuerySamplesComplete([response])


def get_rngd_sut(args):
    return BERT_RNGD_SUT(args)
