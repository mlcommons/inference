import torch

import pickle
from torch.utils.data import DataLoader

import argparse
from quantization import quantize_model
from quantization.utils import random_seed, set_optimization
from quantization.calibrate import load_pytorch_model, load_mlperf_submission_model
from RNGD_encoder import BertMLPerfSubmissionEncoder

from ci_file.utils.check_logit_equality import is_logit_same
from ci_file.utils.turn_on_mcp_dumping import turn_on_mcp_dumping


BUCKET_SIZE = 384
PAD_TOKEN_ID = 0


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="path to gpt-j model")
    parser.add_argument("--model_config_path", help="path to bert model config")
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
        "--n_data", type=int, default=1, help="number of dataset to use for equivalence test"
    )
    parser.add_argument(
        "--logit_folder_path", help="path of the folder in which logit pickle files are to be stored"
    )


    args = parser.parse_args()
    return args


def get_golden_model(model_path, model_config_path, golden_quant_param_path, golden_quant_format_path, gpu, logit_folder_path):
    golden_model = load_pytorch_model(model_path, model_config_path, gpu)
    golden_model = golden_model.trace()

    quant_golden_model= quantize_model(
            golden_model,
            golden_quant_param_path, 
            golden_quant_format_path,
            ) 

    turn_on_mcp_dumping(quant_golden_model, logit_folder_path + '/golden_logits.pkl')     
    
    return quant_golden_model


def get_submission_model(model_path, model_config_path, submission_quant_param_path, submission_quant_format_path, gpu, logit_folder_path):
    
    submission_model = load_mlperf_submission_model(model_path, model_config_path, gpu)
    submission_model = submission_model.trace()

    quant_submission_model = quantize_model(
            submission_model,
            submission_quant_param_path,
            submission_quant_format_path,
        )

    turn_on_mcp_dumping(quant_submission_model, logit_folder_path + '/submission_logits.pkl')

    return BertMLPerfSubmissionEncoder(quant_submission_model, bucket_size=BUCKET_SIZE, pad_token_id=PAD_TOKEN_ID)

    
def perform_generation_to_check_equality(golden_model, submission_model, dataset_path, n_data):

    with open(dataset_path, "rb") as f:
        val_features = pickle.load(f)

    data_list = [
        {
            "input_ids": torch.LongTensor(feature.input_ids).to(torch.device("cuda:0")),
            "attention_mask": torch.LongTensor(feature.input_mask).to(torch.device("cuda:0")),
            "token_type_ids": torch.LongTensor(feature.segment_ids).to(torch.device("cuda:0")),
        }
        for feature in val_features[:n_data]
    ]

    dataloader = DataLoader(data_list, batch_size=1)
    
    # only check 1st input
    for data in dataloader:
        sample_input = data
        golden_model_test_output = sample_input
        comparison_model_test_output = sample_input
        golden_model(**sample_input)
        submission_model.encode(**sample_input)

        break

    return golden_model_test_output, comparison_model_test_output



#load model_script
def compare_model_outputs(args):
    args = get_args()

    golden_model_generator = get_golden_model(args.model_path,
                                            args.model_config_path,
                                            args.golden_quant_param_path, 
                                            args.golden_quant_format_path, 
                                            args.gpu,
                                            args.logit_folder_path,)

    submission_model_generator = get_submission_model(args.model_path,
                                                    args.model_config_path,
                                                    args.submission_quant_param_path, 
                                                    args.submission_quant_format_path, 
                                                    args.gpu,
                                                    args.logit_folder_path,)



    golden_model_test_output, comparison_model_test_output = perform_generation_to_check_equality(golden_model_generator, submission_model_generator, args.dataset_path, args.n_data)
    

    if is_logit_same(
        args.logit_folder_path,
        golden_model_test_output,
        comparison_model_test_output,
        mcm_name_to_check="qa_outputs",
    ):
        print("Logits comparison test passed.")

    else:
        print("Logits comparison test failed!!!")

    

if __name__ == "__main__":
    args = get_args()
    random_seed()
    set_optimization(False)
    compare_model_outputs(args)
    print("qbert forward ci test is passed")