import yaml
from transformers import AutoConfig
import torch
from torch.utils.data import DataLoader
import json
import model_compressor

from dataset import Dataset
import joblib
import pickle

import argparse
from quantization.calibrate import load_pytorch_model, load_mlperf_submission_model
from quantization.quantize import quantize_model
from generator_RNGD import MLPerfSubmissionBeamSearch, expand_inputs_for_generation
from transformers.generation.logits_process import MinNewTokensLengthLogitsProcessor
from transformers.generation.stopping_criteria import MaxLengthCriteria
from transformers.generation.utils import BeamSearchScorer

from ci_file.utils.check_logit_equality import compare_logits

from ci_file.utils.turn_on_mcp_dumping import turn_on_mcp_dumping


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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="path to gpt-j model")
    parser.add_argument("--quant_config_path", help="a config for model quantization")
    parser.add_argument("--golden_quant_format_path", help="path of golden qformat_path")
    parser.add_argument("--golden_quant_param_path", help="path of golden qparam path")
    parser.add_argument("--submission_quant_format_path", help="path of submission qformat_path")
    parser.add_argument("--submission_quant_param_path", help="path of submission qparam path")
    parser.add_argument(
        "--gpu", action="store_true", help="use GPU instead of CPU for the inference"
    )
    parser.add_argument('--dataset_path', help="path of the evaluation file to use")
    parser.add_argument(
        "--n_data", type=int, default=2, help="number of dataset to calibrate"
    )
    parser.add_argument(
        "--logit_folder_path", help="path of the folder in which logit pickle files are to be stored"
    )


    args = parser.parse_args()
    return args



    
def obtain_quant_graphs(model, golden_quant_param_path, golden_quant_format_path, use_gpu):

    traced_models = model.trace_all()

    input_names = {
        "prefill_input_names": traced_models["prefill"].input_names,
        "decode_input_names": traced_models["decode"].input_names,
        }

    concrete_args = {
        "prefill_concrete_args": traced_models["prefill"].concrete_args,
        "decode_concrete_args": traced_models["decode"].concrete_args,
        }


    quant_models = quantize_model(traced_models, golden_quant_param_path, golden_quant_format_path,)
    
    return quant_models, input_names, concrete_args 
                
    


def get_generator_for_golden_model(model_path, qconfig_path, golden_quant_param_path, golden_quant_format_path, gpu, logit_folder_path):
    golden_model = load_pytorch_model(model_path, gpu)
    golden_model_type = type(golden_model)
    quant_golden_models, golden_input_names, golden_concrete_args = obtain_quant_graphs(
                                                                            golden_model,
                                                                            golden_quant_param_path, 
                                                                            golden_quant_format_path, 
                                                                            gpu
                                                                            ) 

    turn_on_mcp_dumping(quant_golden_models, logit_folder_path + '/golden_prefill_logits.pkl', logit_folder_path + '/golden_decode_logits.pkl')

    quant_golden_models = {"prefill_model": quant_golden_models["prefill"], "decode_model": quant_golden_models["decode"]}         
    
    return model_compressor.helper.QuantCausalLM(quant_golden_models, golden_model_type, golden_input_names, golden_concrete_args)

def get_generator_for_submission_model(model_path, qconfig_path, submission_quant_param_path, submission_quant_format_path, gpu, logit_folder_path):
    
    submission_model = load_mlperf_submission_model(model_path, gpu)
    model_config = submission_model.config
    quant_submission_models, input_names, concrete_args, = obtain_quant_graphs(
                                                                            submission_model,
                                                                            submission_quant_param_path, 
                                                                            submission_quant_format_path, 
                                                                            gpu
                                                                            ) 

    turn_on_mcp_dumping(quant_submission_models, logit_folder_path + '/submission_prefill_logits.pkl', logit_folder_path + '/submission_decode_logits.pkl',)

    return MLPerfSubmissionBeamSearch(model = quant_submission_models, model_config=model_config)




    
    
def perform_generation_to_check_equality(golden_model_generator, submission_model_generator, dataset_path, n_data):
    validation_dataset = Dataset(dataset_path)
    device = golden_model_generator.prefill_model.device
    
    for idx in range(n_data):
        input_batch = dict()
        input_batch['input_ids'] = validation_dataset.source_encoded_input_ids[idx].to(device)
        input_batch['attention_mask'] = validation_dataset.source_encoded_attn_masks[idx].to(device)
        seq_len = input_batch['input_ids'].shape[1]


        # Run golden generator
        output_batch_golden = golden_model_generator.generate(**input_batch, **gen_kwargs, pad_token_id = golden_model_generator.config.eos_token_id)


        # Prepare to run submission generator
        logits_processor = LOGITS_PROCESSOR(
                input_batch['input_ids'].shape[-1], MIN_NEW_TOKENS, EOS_TOKEN_ID
            )
            # stopping_criteria = STOPPING_CRITERIA(
        #         MAX_LENGTH,
        #         getattr(submission_generator.model_config, "max_position_embeddings", None),
        #     )
            #The stopping_criteria cannot be used for MLPerf BeamSearch, as the length of every input_ids is fixed to max_prompt_length

        stopping_criteria = None


        beam_scorer = BeamSearchScorer(
            batch_size=input_batch['input_ids'].shape[0],
            num_beams=NUM_BEAMS,
            device=input_batch['input_ids'].device,
            length_penalty=LENGTH_PENALTY,
            do_early_stopping=EARYLY_STOPPING,
            num_beam_hyps_to_keep=NUM_RETURN_SEQUENCES,
            max_length=MAX_LENGTH,
        )
        input_ids_tensor, input_masks_tensor_dict = expand_inputs_for_generation(
            input_ids=input_batch['input_ids'],
            expand_size=NUM_BEAMS,
            attention_mask= input_batch['attention_mask'],
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
    


#load model_script
def compare_model_outputs(args):
    args = get_args()

    golden_model_generator = get_generator_for_golden_model(args.model_path,
                                                        args.quant_config_path, 
                                                        args.golden_quant_param_path, 
                                                        args.golden_quant_format_path, 
                                                        args.gpu,
                                                        args.logit_folder_path,)

    submission_model_generator = get_generator_for_submission_model(args.model_path,
                                                        args.quant_config_path, 
                                                        args.submission_quant_param_path, 
                                                        args.submission_quant_format_path, 
                                                        args.gpu,
                                                        args.logit_folder_path,)



    perform_generation_to_check_equality(golden_model_generator, submission_model_generator, args.dataset_path, args.n_data)
    
   

    compare_logits(args.logit_folder_path)

    
    

if __name__ == "__main__":
    args = get_args()
    compare_model_outputs(args)
    print("gptj forward ci test is passed")