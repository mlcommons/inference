import argparse
import difflib
import pickle
import json

import torch
from torch.utils.data import DataLoader

from ci_file.utils.check_logit_equality import is_logit_same
from ci_file.utils.turn_on_mcp_dumping import turn_on_mcp_dumping
from quantization import quantize_model
from quantization.calibrate import (load_mlperf_submission_model,load_pytorch_model)
from quantization.utils import random_seed, set_optimization
from RNGD_encoder import BertMLPerfSubmissionEncoder

BUCKET_SIZE = 384
PAD_TOKEN_ID = 0

def check_diff(idx, inp_text, ref_sentence, gen_sentence, results, result_flag):
    if ref_sentence == gen_sentence:
        results.append(
            {
                "index": idx,
                "status": "PASS",
                "inp_text": inp_text,
                "generated_sentence": gen_sentence,
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
                "inp_text": inp_text,
                "reference_sentence": ref_sentence,
                "generated_sentence": gen_sentence,
                "differences": diff_result,
            }
        )
    return result_flag

def decode_inp_out(tokenizer, sample_input, submission_output):
    input_ids = sample_input['input_ids'][0]
    inp_decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
    
    start_idx = torch.argmax(submission_output[0][:, 0])
    end_idx = torch.argmax(submission_output[0][:, 1])

    answer = tokenizer.decode(input_ids[start_idx:end_idx + 1], skip_special_tokens=True)
    return inp_decoded_text, answer
        

def get_tokenizer():
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    return tokenizer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="path to bert model")
    parser.add_argument("--model_config_path", help="path to bert model config")
    parser.add_argument("--quant_config_path", help="a config for model quantization")
    parser.add_argument("--golden_quant_format_path", help="path of golden qformat_path")
    parser.add_argument("--golden_quant_param_path", help="path of golden qparam path")
    parser.add_argument("--submission_quant_format_path", help="path of submission qformat_path")
    parser.add_argument("--submission_quant_param_path", help="path of submission qparam path")
    parser.add_argument("--gpu", action="store_true", help="use GPU instead of CPU for the inference")
    parser.add_argument("--dataset_path", help="path of the evaluation file to use")
    parser.add_argument("--n_data",type=int,default=1,help="number of dataset to use for equivalence test",)
    parser.add_argument("--logit_folder_path",help="path of the folder in which logit pickle files are to be stored",)
    parser.add_argument("--ref_path", help="path of reference data")
    parser.add_argument("--res_path", help="path of ci result")
    parser.add_argument("--config_dtype", help="int8 or fp8")
    parser.add_argument("--update_gen_list", action="store_true", help="wheter to update gen_list")
    args = parser.parse_args()
    return args

def get_golden_model(
    model_path,
    model_config_path,
    golden_quant_param_path,
    golden_quant_format_path,
    gpu,
    logit_folder_path,
):
    golden_model = load_pytorch_model(model_path, model_config_path, gpu)
    golden_model = golden_model.trace()

    quant_golden_model = quantize_model(
        golden_model,
        golden_quant_param_path,
        golden_quant_format_path,
    )
    turn_on_mcp_dumping(quant_golden_model, logit_folder_path + "/golden_logits.pkl")

    return quant_golden_model


def get_submission_model(
    model_path,
    model_config_path,
    submission_quant_param_path,
    submission_quant_format_path,
    gpu,
    logit_folder_path,
):

    submission_model = load_mlperf_submission_model(model_path, model_config_path, gpu)
    submission_model = submission_model.trace()

    quant_submission_model = quantize_model(
        submission_model,
        submission_quant_param_path,
        submission_quant_format_path,
    )

    turn_on_mcp_dumping(
        quant_submission_model, logit_folder_path + "/submission_logits.pkl"
    )

    return BertMLPerfSubmissionEncoder(
        quant_submission_model, bucket_size=BUCKET_SIZE, pad_token_id=PAD_TOKEN_ID
    )


def perform_generation_to_check_equality(
    golden_model,
    submission_model,
    dataset_path,
    n_data,
    ref_path,
    res_path,
    config_dtype,
    update_gen_list=False,
):
    tokenizer = get_tokenizer()
    with open(dataset_path, "rb") as f:
        val_features = pickle.load(f)
    # # load reference generated tokens.
    update_ref_path = ref_path + f"/generated_data_list_{config_dtype}.json"
    with open(update_ref_path, "r") as file:
        ref_data = json.load(file)

    results = []
    result_flag = True
    if update_gen_list:
        generated_data_list = []

    print("----------------------------------------------")

    data_list = [
        {
            "input_ids": torch.LongTensor(feature.input_ids).to(torch.device("cuda:0")),
            "attention_mask": torch.LongTensor(feature.input_mask).to(
                torch.device("cuda:0")
            ),
            "token_type_ids": torch.LongTensor(feature.segment_ids).to(
                torch.device("cuda:0")
            ),
        }
        for feature in val_features[:n_data]
    ]

    dataloader = DataLoader(data_list, batch_size=1)

    # only check 1st input
    for idx, data in enumerate(dataloader):
        sample_input = data
        golden_model_test_output = sample_input
        comparison_model_test_output = sample_input
        golden_output = golden_model(**sample_input)
        submission_output = submission_model.encode(**sample_input)

        
        inp_seq, gen_output = decode_inp_out(tokenizer, sample_input, submission_output)
        if update_gen_list:
            generated_data = {"inp_text": inp_seq, "gen_text": gen_output}
            generated_data_list.append(generated_data)
        print(f"생성 토큰 문장 {idx}: {gen_output}")
        # compare submission model's decoded_test with reference sentences.
        ref_seq = ref_data[idx]["gen_text"]
        result_flag = check_diff(idx, inp_seq, ref_seq, gen_output, results, result_flag)
        break # todos
    
            
    compare_results_path = res_path + f"/bert_compare_result_{config_dtype}.json"
    with open(compare_results_path, "w") as file:
        json.dump(results, file, indent=4)
        print(f"토큰 동치비교 결과가 저장되었습니다. dir: {compare_results_path}")
    if update_gen_list:
        with open(update_ref_path, "w") as file:
            json.dump(generated_data_list, file, indent=4)
        print(f"새로운 토큰 결과로 reference가 업데이트 되었습니다. dir: {update_ref_path}")

    return golden_model_test_output, comparison_model_test_output, result_flag


# load model_script
def compare_model_outputs(args):
    args = get_args()

    golden_model_generator = get_golden_model(
        args.model_path,
        args.model_config_path,
        args.golden_quant_param_path,
        args.golden_quant_format_path,
        args.gpu,
        args.logit_folder_path,
    )

    submission_model_generator = get_submission_model(
        args.model_path,
        args.model_config_path,
        args.submission_quant_param_path,
        args.submission_quant_format_path,
        args.gpu,
        args.logit_folder_path,
    )

    golden_model_test_output, comparison_model_test_output, result_flag = (
        perform_generation_to_check_equality(
            golden_model_generator,
            submission_model_generator,
            args.dataset_path,
            args.n_data,
            args.ref_path,
            args.res_path,
            args.config_dtype,
            update_gen_list=args.update_gen_list,
        )
    )
    print("----------------------------------------------")
    print(f"토큰 동치 비교 결과 : {result_flag}")
    print("----------------------------------------------")
    
    if is_logit_same(
        args.logit_folder_path,
        golden_model_test_output,
        comparison_model_test_output,
        mcm_name_to_check="qa_outputs",
    ):
        print("bert Golden <-> Submission logit 동치비교: PASS")
    else:
        print("bert Golden <-> Submission logit 동치비교: FAIL")


if __name__ == "__main__":
    args = get_args()
    random_seed()
    set_optimization(False)
    compare_model_outputs(args)
    print("qbert forward ci test is passed")
