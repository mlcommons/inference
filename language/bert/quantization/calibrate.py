import argparse
import json
import pickle

import torch
import yaml
from torch.fx import GraphModule
from torch.utils.data import DataLoader
from transformers import BertConfig, BertForQuestionAnswering

import model_compressor  # isort:skip

from .utils import get_kwargs, random_seed, set_optimization  # isort:skip

PADDING_SIZE = 384
BUCKET_SIZE = 384
PAD_TOKEN_ID = 0

def load_pytorch_model(model_path, model_config_path, use_gpu):
    from furiosa_llm_models.bert.symbolic.huggingface_rngd_gelu import BertForQuestionAnswering
    with open(model_config_path) as f:
        config_json = json.load(f)

    config = BertConfig(**config_json)
    device = torch.device("cuda:0") if use_gpu else torch.device("cpu")

    model = BertForQuestionAnswering(config)
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(model_path), strict=False)
    return model

def load_mlperf_submission_model(model_path, model_config_path, use_gpu):
    from furiosa_llm_models.bert.symbolic.mlperf_submission import BertForQuestionAnswering
    with open(model_config_path) as f:
        config_json = json.load(f)

    config = BertConfig(**config_json)
    device = torch.device("cuda:0") if use_gpu else torch.device("cpu")

    model = BertForQuestionAnswering(config)
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(model_path), strict=False)

    return model


def cal_data_loader(data_path, batch_size, n_calib, model_type, is_equivalence_ci=False):
    with open(data_path, "rb") as f:
        cal_features = pickle.load(f)

    data_list = [
        {
            "input_ids": torch.LongTensor(feature.input_ids),
            "attention_mask": torch.LongTensor(feature.input_mask),
            "token_type_ids": torch.LongTensor(feature.segment_ids),
        }
        for feature in cal_features[:n_calib]
    ]

    if model_type == "golden":
        return DataLoader(data_list, batch_size=batch_size)
    
    elif model_type == "mlperf-submission":
        if is_equivalence_ci:
            for data in data_list:
                data.update(
                    {
                        "attention_mask": data["attention_mask"]
                        .unsqueeze(0)
                        .repeat(PADDING_SIZE, 1),
                        "position_ids": torch.arange(PADDING_SIZE),
                    }
                )

            return DataLoader(data_list, batch_size=batch_size)
        
        else:
            from RNGD_encoder import greedy_attention_packing_bert, bucket_pad

            for data in data_list:
                (
                    input_ids,
                    token_type_ids,
                    attention_mask,
                    position_ids,
                    packed_target_locations,
                ) = greedy_attention_packing_bert(
                    input_ids=bucket_pad(data["input_ids"].unsqueeze(0), BUCKET_SIZE),
                    token_type_ids=bucket_pad(data["token_type_ids"].unsqueeze(0), BUCKET_SIZE),
                    bucketized_attention_mask=bucket_pad(data["attention_mask"].unsqueeze(0), BUCKET_SIZE),
                    pad_token_id=PAD_TOKEN_ID,
                    compact_mask=False, # TODO : do we use compact mask?
                )

                data.update(
                    {
                        "input_ids": input_ids[0],
                        "token_type_ids": token_type_ids[0],
                        "attention_mask": attention_mask[0],
                        "position_ids": position_ids[0],
                    }
                )

            return DataLoader(data_list, batch_size=batch_size)
    
    else:
        ValueError("Unsupported backend: {:}".format(model_type))


def calibrate(model: GraphModule, qconfig, qparam_path, qformat_path, calib_dataloader, save_cache_files):
    model.config.use_cache = False

    model = model_compressor.create_quantsim_model(
        model,
        dataloader=calib_dataloader,
        disable_inout=(True, True),
        **get_kwargs(model_compressor.create_quantsim_model, qconfig),
    )

    model_compressor.calibrate(
        model,
        dataloader=calib_dataloader,
        **get_kwargs(model_compressor.calibrate, qconfig),
    )

    qformat, qparam = model_compressor.extract_qformat_and_qparam(model)
    model_compressor.save_qformat_qparam(qformat_dict=qformat,
                                         qformat_out_path=qformat_path,
                                         qparam_dict=qparam, 
                                         qparam_out_path=qparam_path,
                                         **get_kwargs(model_compressor.save_qformat_qparam, qconfig),
                                         )

    if save_cache_files:
        qlv4_prefill_out_path = qparam_path.replace("quant_param.npy", "bert.bin")
        rblock_json_out_path = qparam_path.replace("quant_param.npy", "graph_patterns.json")
        torch.save(model.state_dict(), qlv4_prefill_out_path)
        
        # model_compressor.save_graph_patterns(model, rblock_json_out_path)


    return


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", choices=["golden", "mlperf-submission"], default="mlperf-submission", help="model_type"
    )
    parser.add_argument("--model_path", help="path to bert model")
    parser.add_argument("--model_config_path", help="path to bert model config")
    parser.add_argument("--quant_config_path", help="a config for model quantization")
    parser.add_argument(
        "--quant_param_path", help="quantization parameters for calibrated layers"
    )
    parser.add_argument(
        "--quant_format_path", help="quantization specifications for calibrated layers"
    )
    parser.add_argument("--calib_data_path", help="path to calibration data")
    parser.add_argument(
        "--n_calib", type=int, default=-1, help="number of dataset to calibrate"
    )
    parser.add_argument(
        "--torch_numeric_optim",
        action="store_true",
        help="use PyTorch numerical optimizaiton for CUDA/cuDNN",
    )
    parser.add_argument(
        "--gpu", action="store_true", help="use GPU instead of CPU for the inference"
    )
    parser.add_argument(
        "--is_equivalence_ci",
        action="store_true",
        default=False,
        help="flag for equivalence_ci",
    )
    parser.add_argument(
        "--save_cache_files",
        action="store_true",
        default=False,
        help="if true qlv4 state_dict and rblock .json will be saved",
    )

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    if args.model_type == "golden":
        if not args.gpu:
            raise ValueError(
                "Calibration on a device other than GPU is not supported yet."
            )
        model = load_pytorch_model(args.model_path, args.model_config_path, args.gpu)
        model = model.trace()

    elif args.model_type == "mlperf-submission":
        if not args.gpu:
            raise ValueError(
                "Calibration on a device other than GPU is not supported yet."
            )
        
        model = load_mlperf_submission_model(args.model_path, args.model_config_path, args.gpu)
        model = model.trace()

    else:
        raise ValueError("Unsupported backend: {:}".format(args.model_type))

    random_seed()
    set_optimization(args.torch_numeric_optim)

    with open(args.quant_config_path, "r") as f:
        qconfig = yaml.safe_load(f)
    dataloader = cal_data_loader(
        args.calib_data_path, qconfig["calib_batch_size"], args.n_calib, args.model_type, args.is_equivalence_ci
    )
    calibrate(
        model,
        qconfig,
        args.quant_param_path,
        args.quant_format_path,
        dataloader,
        args.save_cache_files
    )


if __name__ == "__main__":
    main()
