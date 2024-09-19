import argparse
import difflib
import gc
import json
import pickle

import furiosa_llm_models
import joblib
import model_compressor
import torch
import yaml
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers.generation.logits_process import \
    MinNewTokensLengthLogitsProcessor
from transformers.generation.stopping_criteria import MaxLengthCriteria
from transformers.generation.utils import BeamSearchScorer

from ci_file.utils.check_logit_equality import compare_logits
from ci_file.utils.compare_output_yaml import compare_output_yaml
from ci_file.utils.gen_test_data import gen_test_data
from ci_file.utils.turn_on_mcp_dumping import turn_on_mcp_dumping
from dataset import Dataset
from quantization.calibrate import load_pytorch_model
from quantization.quantize import quantize_model
from RNGD_generator import MLPerfSubmissionGreedySearch

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
    "do_sample": False,
}

def check_diff(idx, ref_sentence, gen_sentence, results, result_flag):
    if ref_sentence == gen_sentence:
        results.append(
            {
                "index": idx,
                "status": "PASS",
                "generated_sentence": gen_sentence,
                "reference_sentence": ref_sentence,
            }
        )
    else:
        result_flag = False
        diff = list(
            difflib.unified_diff(
                ref_sentence.split(), gen_sentence.split(), lineterm=""
            )
        )
        diff_result = " ".join(diff) 
        results.append(
            {
                "index": idx,
                "status": "DIFFERENT",
                "reference_sentence": ref_sentence,
                "generated_sentence": gen_sentence,
                "differences": diff_result,
            }
        )
    return result_flag


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="path to llama3.1-70b model")
    parser.add_argument("--quant_config_path", help="a config for model quantization")
    parser.add_argument("--golden_quant_format_path", help="path of golden qformat_path")
    parser.add_argument("--golden_quant_param_path", help="path of golden qparam path")
    parser.add_argument("--submission_quant_format_path", help="path of submission qformat_path")
    parser.add_argument("--submission_quant_param_path", help="path of submission qparam path")
    parser.add_argument("--gpu", action="store_true", help="use GPU instead of CPU for the inference")
    parser.add_argument("--dataset_path", help="path of the evaluation file to use")
    parser.add_argument("--n_data", type=int, default=2, help="number of dataset to calibrate")
    parser.add_argument("--logit_folder_path",default=None,help="path of the folder in which logit pickle files are to be stored",)
    parser.add_argument("--generation_result_folder_path",help="path of the folder in which the log files of generation results are to be stored",)
    parser.add_argument("--n_layers", type=int, default=-1, help="the number of layers")
    parser.add_argument("--mcp_dumping_on",action="store_true",help="turn on mcp dumping to compare the equality of logits at each decoding step",)
    parser.add_argument("--ref_path", help="path of reference data")
    parser.add_argument("--res_path", help="path of ci result")
    parser.add_argument("--config_dtype", help="int8 or fp8")
    parser.add_argument("--update_gen_list", action="store_true", help="wheter to update gen_list")
    parser.add_argument("--param_num", default='8b', help="wheter to update gen_list")
    args = parser.parse_args()
    return args


def obtain_traced_model_dict(model):

    if (
        type(model)
        == furiosa_llm_models.llama.symbolic.huggingface_rope.LlamaForCausalLM
    ):
        (
            prefill_model,
            prefill_input_names,
            prefill_concrete_args,
        ) = model_compressor.helper.llama_custom_symbolic_trace(
            model,
            input_names=["input_ids", "attention_mask", "position_ids"],
            disable_check=True,
        )
        (
            decode_model,
            decode_input_names,
            decode_concrete_args,
        ) = model_compressor.helper.llama_custom_symbolic_trace(
            model,
            input_names=[
                "input_ids",
                "past_key_values",
                "attention_mask",
                "position_ids",
            ],
            disable_check=True,
        )

        traced_models = {"prefill": prefill_model, "decode": decode_model}

        input_names = {
            "prefill_input_names": prefill_input_names,
            "decode_input_names": decode_input_names,
        }

        concrete_args = {
            "prefill_concrete_args": prefill_concrete_args,
            "decode_concrete_args": decode_concrete_args,
        }

    elif (
        type(model)
        == furiosa_llm_models.llama.symbolic.mlperf_submission_slice.LlamaForCausalLM
    ):
        traced_models = model.trace_all()

        input_names = {
            "prefill_input_names": traced_models["prefill"].input_names,
            "decode_input_names": traced_models["decode"].input_names,
        }

        concrete_args = {
            "prefill_concrete_args": traced_models["prefill"].concrete_args,
            "decode_concrete_args": traced_models["decode"].concrete_args,
        }

    else:
        raise NotImplementedError

    return traced_models, input_names, concrete_args


def obtain_quant_graphs(model, quant_param_path, quant_format_path):

    traced_model_dict = obtain_traced_model_dict(model)

    quant_models = quantize_model(
        traced_model_dict,
        quant_param_path,
        quant_format_path,
    )

    return quant_models


def get_generator_for_golden_model(
    model_path,
    qconfig_path,
    golden_quant_param_path,
    golden_quant_format_path,
    gpu,
    n_layers,
    logit_folder_path,
    mcp_dumping_on,
):

    if mcp_dumping_on and logit_folder_path == None:
        raise ValueError("Logit folder path is required to enable mcp dumping")

    golden_model = load_pytorch_model(
        model_source="furiosa_llm_rope",
        model_path=model_path,
        use_gpu=gpu,
        n_layers=n_layers,
    )

    golden_model_type = type(golden_model)
    assert (
        golden_model_type
        == furiosa_llm_models.llama.symbolic.huggingface_rope.LlamaForCausalLM
    )

    traced_golden_models, golden_input_names, golden_concrete_args = (
        obtain_traced_model_dict(golden_model)
    )

    quant_golden_models = quantize_model(
        traced_golden_models, golden_quant_param_path, golden_quant_format_path
    )

    if mcp_dumping_on:
        turn_on_mcp_dumping(
            quant_golden_models,
            logit_folder_path + "/golden_prefill_logits.pkl",
            logit_folder_path + "/golden_decode_logits.pkl",
        )

    quant_golden_models = {
        "prefill_model": quant_golden_models["prefill"],
        "decode_model": quant_golden_models["decode"],
    }

    return model_compressor.helper.QuantCausalLM(
        quant_golden_models, golden_model_type, golden_input_names, golden_concrete_args
    )


def get_generator_for_submission_model(
    model_path,
    qconfig_path,
    submission_quant_param_path,
    submission_quant_format_path,
    gpu,
    n_layers,
    logit_folder_path,
    mcp_dumping_on,
):

    if mcp_dumping_on and logit_folder_path == None:
        raise ValueError("Logit folder path is required to enable mcp dumping")

    submission_model = load_pytorch_model(
        model_source="mlperf_submission_slice",
        model_path=model_path,
        use_gpu=gpu,
        n_layers=n_layers,
    )

    assert (
        type(submission_model)
        == furiosa_llm_models.llama.symbolic.mlperf_submission_slice.LlamaForCausalLM
    )

    # Needs to place paged attention key value blocks on the same device as the transformer layers
    if hasattr(submission_model, "hf_device_map"):
        TRANSFORMER_LAYER_MODULE = "model.layers"  # valid only for LlamaForCausalLM
        device_map = {
            k.split(TRANSFORMER_LAYER_MODULE + ".")[1]: v
            for k, v in submission_model.hf_device_map.items()
            if TRANSFORMER_LAYER_MODULE in k
        }
    else:
        device_map = None

    traced_submission_models, _, _ = obtain_traced_model_dict(submission_model)

    quant_submission_models = quantize_model(
        traced_submission_models,
        submission_quant_param_path,
        submission_quant_format_path,
    )

    if mcp_dumping_on:
        turn_on_mcp_dumping(
            quant_submission_models,
            logit_folder_path + "/submission_prefill_logits.pkl",
            logit_folder_path + "/submission_decode_logits.pkl",
        )

    return MLPerfSubmissionGreedySearch(
        model=quant_submission_models, device_map=device_map
    )


def perform_generation(
    generator,
    test_data_list,
    logit_file_path,
    generation_result_file_path,
    tokenizer,
    ref_path=None,
    res_path=None,
    config_dtype=None,
    update_gen_list=False,
    param_num='8b',
):
    if type(generator) == MLPerfSubmissionGreedySearch:  # mlperf submission generate
        # load reference generated tokens.
        update_ref_path = ref_path + f"/generated_data_list_{config_dtype}.json"
        # with open(update_ref_path, "r") as file:
        #     ref_data = json.load(file)

        results = []
        result_flag = True
        if update_gen_list:
            generated_data_list = []

    generation_output_dictionary = dict()
    with torch.no_grad():
        for idx, test_data in enumerate(test_data_list):
            if type(generator) == model_compressor.helper.QuantCausalLM:
                output = generator.generate(**test_data, **gen_kwargs)
            elif type(generator) == MLPerfSubmissionGreedySearch:
                # mlperf submission
                input_ids_tensor = []
                input_masks_tensor = []
                max_seq_len = 1024

                input_ids_tensor.append(
                    pad(
                        test_data["input_ids"],
                        (max_seq_len - test_data["input_ids"].shape[-1], 0, 0, 0),
                        value=tokenizer.pad_token_id,
                    )
                )

                input_masks_tensor.append(
                    pad(
                        test_data["attention_mask"],
                        (max_seq_len - test_data["attention_mask"].shape[-1], 0, 0, 0),
                        value=0,
                    )
                )

                input_ids_tensor = torch.cat(input_ids_tensor)
                input_masks_tensor = torch.cat(input_masks_tensor)

                logits_processor = LOGITS_PROCESSOR(
                    input_ids_tensor.shape[-1], MIN_NEW_TOKENS, EOS_TOKEN_ID
                )

                stopping_criteria = STOPPING_CRITERIA(
                    MAX_LENGTH,
                    None,
                )

                output = generator.generate(
                    input_ids=input_ids_tensor,
                    attention_mask=input_masks_tensor,
                    logits_processor=logits_processor,
                    stopping_criteria=stopping_criteria,
                    max_length=MAX_LENGTH,
                    pad_token_id=PAD_TOKEN_ID,
                    eos_token_id=EOS_TOKEN_ID,
                    return_dict_in_generate=RETURN_DICT_IN_GENERATE,
                    kv_dtype=QUANT_KV_DTYPE,
                    bucket_size=BUCKET_SIZE,
                )

                generated_token = output[0][len(input_ids_tensor[0]) :]
                gen_sentence = tokenizer.decode(
                    generated_token, skip_special_tokens=True
                )
                if update_gen_list:
                    inp_decoded_text = tokenizer.decode(
                        input_ids_tensor[0], skip_special_tokens=True
                    )
                    generated_data = {
                        "inp_text": inp_decoded_text,
                        "gen_text": gen_sentence,
                    }
                    generated_data_list.append(generated_data)
                print(f"생성 토큰 문장 {idx}: {gen_sentence}")
                # compare submission model's decoded_test with reference sentences.
                # ref_sentence = ref_data[idx]["gen_text"]
                result_flag = check_diff(idx, ref_sentence, gen_sentence, results, result_flag)

            generation_output_dictionary[idx] = tokenizer.decode(
                output[0], skip_special_tokens=True
            )

        with open(generation_result_file_path, "w") as f:
            yaml.dump(generation_output_dictionary, f)

        if (type(generator) == MLPerfSubmissionGreedySearch):  # mlperf submission generate
            # compare_results_path = res_path + f"/llama3.1-{param_num}_compare_result_{config_dtype}.json"
            # with open(compare_results_path, "w") as file:
            #     json.dump(results, file, indent=4)
            #     print(f"토큰 동치비교 결과가 저장되었습니다. dir: {compare_results_path}")
            if update_gen_list:
                with open(update_ref_path, "w") as file:
                    json.dump(generated_data_list, file, indent=4)
                print(
                    f"새로운 토큰 결과로 reference가 업데이트 되었습니다. dir: {update_ref_path}"
                )
            return result_flag


# load model_script
def compare_model_outputs(args):

    test_data_list = gen_test_data(args.dataset_path, args.n_data)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=1024,
        padding_side="left",
        use_fast=False,
    )

    golden_model_generator = get_generator_for_golden_model(
        args.model_path,
        args.quant_config_path,
        args.golden_quant_param_path,
        args.golden_quant_format_path,
        args.gpu,
        args.n_layers,
        args.logit_folder_path,
        args.mcp_dumping_on,
    )

    golden_generation_result_file_path = (
        args.generation_result_folder_path + "/golden_generation_output.yaml"
    )

    perform_generation(
        golden_model_generator,
        test_data_list,
        args.logit_folder_path,
        golden_generation_result_file_path,
        tokenizer,
    )

    del golden_model_generator
    gc.collect()
    torch.cuda.empty_cache()
    
    submission_model_generator = get_generator_for_submission_model(
        args.model_path,
        args.quant_config_path,
        args.submission_quant_param_path,
        args.submission_quant_format_path,
        args.gpu,
        args.n_layers,
        args.logit_folder_path,
        args.mcp_dumping_on,
    )

    submission_generation_result_file_path = (
        args.generation_result_folder_path + "/submission_generation_output.yaml"
    )
    result_flag = perform_generation(
        submission_model_generator,
        test_data_list,
        args.logit_folder_path,
        submission_generation_result_file_path,
        tokenizer,
        ref_path=args.ref_path,
        res_path=args.res_path,
        config_dtype=args.config_dtype,
        update_gen_list=args.update_gen_list,
        param_num=args.param_num
    )
    print("----------------------------------------------")
    print(f"토큰 동치 비교 결과 : {result_flag}")
    print("----------------------------------------------")

    golden_generation_result_file_path = (
        args.generation_result_folder_path + "/golden_generation_output.yaml"
    )
    submission_generation_result_file_path = (
        args.generation_result_folder_path + "/submission_generation_output.yaml"
    )
    compare_output_yaml(
        golden_generation_result_file_path, submission_generation_result_file_path
    )

    if args.mcp_dumping_on:
        compare_logits(args.logit_folder_path, is_slice=True)


if __name__ == "__main__":
    args = get_args()
    compare_model_outputs(args)
    print("llama3.1 forward ci test is passed")
