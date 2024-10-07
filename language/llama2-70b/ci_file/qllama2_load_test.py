import yaml
from transformers import AutoConfig
import torch
from torch.utils.data import DataLoader
import json
import model_compressor

import joblib
import pickle

import argparse
from quantization.calibrate import load_pytorch_model
from transformers import LlamaConfig
from transformers import AutoTokenizer

from quantization.quantize import quantize_model
from RNGD_generator import MLPerfSubmissionGreedySearch
from transformers.generation.logits_process import MinNewTokensLengthLogitsProcessor
from transformers.generation.stopping_criteria import MaxLengthCriteria
from transformers.generation.utils import BeamSearchScorer
from furiosa_llm_models.llama.symbolic.mlperf_submission_slice import LlamaForCausalLM


import furiosa_llm_models
import gc 
from torch.nn.functional import pad
import yaml 
import os
import accelerate

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Assume BLOCK_SIZE, NUM_BLOCKS, BUCKET_SIZE are fixed for now.
BLOCK_SIZE = 1
# bucket size would simply be a max value such as 2048 since we only provide one bucket
EARLY_STOPPING = True
PAD_TOKEN_ID = EOS_TOKEN_ID = 2
MAX_LENGTH = 2048
MAX_NEW_TOKENS = 1024
MIN_NEW_TOKENS = 1
NUM_BEAMS = 1
DO_SAMPLE = False
RETURN_DICT_IN_GENERATE = False
LOGITS_PROCESSOR = MinNewTokensLengthLogitsProcessor
STOPPING_CRITERIA = MaxLengthCriteria
KV_DTYPE = torch.float32
QUANT_KV_DTYPE = torch.int8
BUCKET_SIZE = 2048

gen_kwargs = {
    "early_stopping": True,
    "min_new_tokens": 1,
    "max_new_tokens": 1024,
    "num_beams": 1,
    "do_sample": False
}

DEVICE = 'cpu'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="path to gpt-j model")
    parser.add_argument("--quant_config_path", help="a config for model quantization")
    parser.add_argument("--quant_data_path", help="path of submission quant data path")

    args = parser.parse_args()
    return args


def get_qlv4_load_models(
    model_path,
    output_path,
    qparam_out_path,
    qformat_out_path,
    prefill_exported_model_out_path,
    decode_exported_model_out_path,
):
    """
    Test 수행할 QLV4 quantized model 생성
    1. empty weight model 로딩
    2. qlv4 model로 변환
    3. qlv4 weight 로딩
    """
    CONFIG_PATH = os.path.join(model_path, "config.json")
    with open(CONFIG_PATH, "r") as f:
        config_dict = json.load(f)
        custom_config = LlamaConfig.from_dict(config_dict)
        
    with accelerate.init_empty_weights():
        model = LlamaForCausalLM.from_pretrained(model_path)

    model_type = type(model)
    traced_model = model.trace_all()
    prefill_model = traced_model['prefill']
    decode_model = traced_model['decode']

    test_prefill_quantized_model = model_compressor.create_quantsim_model(
        prefill_model,
        qformat_path=qformat_out_path,
        qparam_path=qparam_out_path,
        qlevel=4,
        target_machine='RGDA0',
        decode_phase=False,
        output_path=output_path,
    )

    test_decode_quantized_model = model_compressor.create_quantsim_model(
        decode_model,
        qformat_path=qformat_out_path,
        qparam_path=qparam_out_path,
        qlevel=4,
        target_machine='RGDA0',
        decode_phase=True,
        quantized_prefill_model=test_prefill_quantized_model,
        disable_auto_node_mapping=True,
    )

    map_location = torch.device(DEVICE)
    model_compressor.load(
        test_prefill_quantized_model, prefill_exported_model_out_path, map_location=map_location
    )
    model_compressor.load(
        test_decode_quantized_model, decode_exported_model_out_path, map_location=map_location
    )

    test_prefill_quantized_model = test_prefill_quantized_model.to(DEVICE)
    test_decode_quantized_model = test_decode_quantized_model.to(DEVICE)

    return model_type, test_prefill_quantized_model, test_decode_quantized_model



#load model_script
def create_qlv4_model(args):


    tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            model_max_length=1024,
            padding_side="left",
            use_fast=False,
        )
    

    qparam_path = os.path.join(args.quant_data_path, 'quant_param.npy')
    qformat_path = os.path.join(args.quant_data_path, 'quant_format.yaml')
    
    prefill_state_dict_path = os.path.join(args.quant_data_path, 'prefill.bin')
    decode_state_dict_path = os.path.join(args.quant_data_path, 'decode.bin')
    
    model_type, prefill_quantized_model, decode_quantized_model = get_qlv4_load_models(
        args.model_path,
        args.quant_data_path,
        qparam_path, 
        qformat_path, 
        prefill_state_dict_path, 
        decode_state_dict_path
        )
    
    
if __name__ == "__main__":
    args = get_args()
    create_qlv4_model(args)
    print("llama-70b qlv4 load test is passed")