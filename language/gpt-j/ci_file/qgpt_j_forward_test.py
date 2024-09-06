import argparse
import difflib
import json
import pickle

import joblib
import model_compressor
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoConfig
from transformers.generation.logits_process import MinNewTokensLengthLogitsProcessor
from transformers.generation.stopping_criteria import MaxLengthCriteria
from transformers.generation.utils import BeamSearchScorer

from ci_file.utils.check_logit_equality import compare_logits
from ci_file.utils.turn_on_mcp_dumping import turn_on_mcp_dumping
from dataset import Dataset
from generator_RNGD import (MLPerfSubmissionBeamSearch,expand_inputs_for_generation)
from quantization.calibrate import (load_mlperf_submission_model,load_pytorch_model)
from quantization.quantize import quantize_model

EARYLY_STOPPING = True
PAD_TOKEN_ID = EOS_TOKEN_ID = 50256
MAX_LENGTH = 2048
MAX_NEW_TOKENS = 128
MIN_NEW_TOKENS = 30
NUM_BEAMS = 4
LENGTH_PENALTY = 1.0
NUM_RETURN_SEQUENCES = 1
RETURN_DICT_IN_GENERATE = False
LOGITS_PROCESSOR = MinNewTokensLengthLogitsProcessor
STOPPING_CRITERIA = MaxLengthCriteria
KV_DTYPE = torch.int8
BUCKET_SIZE = 2048
NUM_REAL_BATCH = 1

gen_kwargs = {
    "early_stopping": True,
    "max_new_tokens": 128,
    "min_new_tokens": 30,
    "num_beams": 4,
}


def get_tokenizer():
    from tokenizer_GPTJ import get_transformer_autotokenizer

    tokenizer = get_transformer_autotokenizer("EleutherAI/gpt-j-6B")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


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
    parser.add_argument("--model_path", help="path to gpt-j model")
    parser.add_argument("--quant_config_path", help="a config for model quantization")
    parser.add_argument("--golden_quant_format_path", help="path of golden qformat_path")
    parser.add_argument("--golden_quant_param_path", help="path of golden qparam path")
    parser.add_argument("--submission_quant_format_path", help="path of submission qformat_path")
    parser.add_argument("--submission_quant_param_path", help="path of submission qparam path")
    parser.add_argument("--gpu", action="store_true", help="use GPU instead of CPU for the inference")
    parser.add_argument("--dataset_path", help="path of the evaluation file to use")
    parser.add_argument("--n_data", type=int, default=2, help="number of dataset to calibrate")
    parser.add_argument("--logit_folder_path",help="path of the folder in which logit pickle files are to be stored",)
    parser.add_argument("--ref_path", help="path of reference data")
    parser.add_argument("--res_path", help="path of ci result")
    parser.add_argument("--config_dtype", help="int8 or fp8")
    parser.add_argument("--update_gen_list", action="store_true", help="wheter to update gen_list")
    args = parser.parse_args()
    return args


def obtain_quant_graphs(
    model, golden_quant_param_path, golden_quant_format_path, use_gpu
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
    quant_models = quantize_model(
        traced_models,
        golden_quant_param_path,
        golden_quant_format_path,
    )
    return quant_models, input_names, concrete_args


def get_generator_for_golden_model(
    model_path,
    qconfig_path,
    golden_quant_param_path,
    golden_quant_format_path,
    gpu,
    logit_folder_path,
):
    golden_model = load_pytorch_model(model_path, gpu)
    golden_model_type = type(golden_model)
    quant_golden_models, golden_input_names, golden_concrete_args = obtain_quant_graphs(
        golden_model, golden_quant_param_path, golden_quant_format_path, gpu
    )

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
    logit_folder_path,
):
    submission_model = load_mlperf_submission_model(model_path, gpu)
    model_config = submission_model.config
    (
        quant_submission_models,
        input_names,
        concrete_args,
    ) = obtain_quant_graphs(
        submission_model, submission_quant_param_path, submission_quant_format_path, gpu
    )
    turn_on_mcp_dumping(
        quant_submission_models,
        logit_folder_path + "/submission_prefill_logits.pkl",
        logit_folder_path + "/submission_decode_logits.pkl",
    )
    return MLPerfSubmissionBeamSearch(
        model=quant_submission_models, model_config=model_config
    )

def generate_compare_gen_token(
    golden_model_generator,
    submission_model_generator,
    dataset_path,
    n_data,
    ref_path,
    res_path,
    config_dtype,
    update_gen_list=False,
):
    validation_dataset = Dataset(dataset_path, total_count_override=n_data)
    device = golden_model_generator.prefill_model.device
    tokenizer = get_tokenizer()
    # load reference generated tokens.
    update_ref_path = ref_path + f"/generated_data_list_{config_dtype}.json"
    with open(update_ref_path, "r") as file:
        ref_data = json.load(file)

    results = []
    result_flag = True
    if update_gen_list:
        generated_data_list = []

    print("----------------------------------------------")
    for idx in range(n_data):
        input_batch = dict()
        input_batch["input_ids"] = validation_dataset.source_encoded_input_ids[idx].to(device)
        input_batch["attention_mask"] = validation_dataset.source_encoded_attn_masks[idx].to(device)
        seq_len = input_batch["input_ids"].shape[1]

        # Run golden generator
        output_batch_golden = golden_model_generator.generate(
            **input_batch,
            **gen_kwargs,
            pad_token_id=golden_model_generator.config.eos_token_id,
        )

        # Prepare to run submission generator
        logits_processor = LOGITS_PROCESSOR(
            input_batch["input_ids"].shape[-1], MIN_NEW_TOKENS, EOS_TOKEN_ID
        )
        stopping_criteria = None

        beam_scorer = BeamSearchScorer(
            batch_size=input_batch["input_ids"].shape[0],
            num_beams=NUM_BEAMS,
            device=input_batch["input_ids"].device,
            length_penalty=LENGTH_PENALTY,
            do_early_stopping=EARYLY_STOPPING,
            num_beam_hyps_to_keep=NUM_RETURN_SEQUENCES,
            max_length=MAX_LENGTH,
        )
        input_ids_tensor, input_masks_tensor_dict = expand_inputs_for_generation(
            input_ids=input_batch["input_ids"],
            expand_size=NUM_BEAMS,
            attention_mask=input_batch["attention_mask"],
        )
        input_masks_tensor = input_masks_tensor_dict["attention_mask"]

        # Run submission generator
        output_batch = submission_model_generator.generate(
            input_ids=input_ids_tensor,
            attention_mask=input_masks_tensor,
            beam_scorer=beam_scorer,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            max_length=MAX_LENGTH,
            pad_token_id=PAD_TOKEN_ID,
            eos_token_id=EOS_TOKEN_ID,
            return_dict_in_generate=RETURN_DICT_IN_GENERATE,
            kv_dtype=KV_DTYPE,
            bucket_size=BUCKET_SIZE,
        )
        generated_token = output_batch[0][len(input_ids_tensor[0]) :]
        gen_sentence = tokenizer.decode(generated_token, skip_special_tokens=True)
        if update_gen_list:
            inp_decoded_text = tokenizer.decode(input_ids_tensor[0], skip_special_tokens=True)
            generated_data = {"inp_text": inp_decoded_text, "gen_text": gen_sentence}
            generated_data_list.append(generated_data)
        print(f"생성 토큰 문장 {idx}: {gen_sentence}")
        # compare submission model's decoded_test with reference sentences.
        ref_sentence = ref_data[idx]["gen_text"]
        result_flag = check_diff(idx, ref_sentence, gen_sentence, results, result_flag)

    compare_results_path = res_path + f"/qgpt_j_compare_result_{config_dtype}.json"
    with open(compare_results_path, "w") as file:
        json.dump(results, file, indent=4)
        print(f"토큰 동치비교 결과가 저장되었습니다. dir: {compare_results_path}")
    if update_gen_list:
        with open(update_ref_path, "w") as file:
            json.dump(generated_data_list, file, indent=4)
        print(f"새로운 토큰 결과로 reference가 업데이트 되었습니다. dir: {update_ref_path}")
    return result_flag


# load model_script
def compare_model_outputs(args):
    args = get_args()

    golden_model_generator = get_generator_for_golden_model(
        args.model_path,
        args.quant_config_path,
        args.golden_quant_param_path,
        args.golden_quant_format_path,
        args.gpu,
        args.logit_folder_path,
    )

    submission_model_generator = get_generator_for_submission_model(
        args.model_path,
        args.quant_config_path,
        args.submission_quant_param_path,
        args.submission_quant_format_path,
        args.gpu,
        args.logit_folder_path,
    )

    result_flag = generate_compare_gen_token(
        golden_model_generator,
        submission_model_generator,
        args.dataset_path,
        args.n_data,
        args.ref_path,
        args.res_path,
        args.config_dtype,
        update_gen_list=args.update_gen_list,
    )
    print("----------------------------------------------")
    print(f"토큰 동치 비교 결과 : {result_flag}")
    print("----------------------------------------------")
    compare_logits(args.logit_folder_path)


if __name__ == "__main__":
    args = get_args()
    compare_model_outputs(args)
    print("gptj Golden <-> Submission logit 동치비교: PASS")
