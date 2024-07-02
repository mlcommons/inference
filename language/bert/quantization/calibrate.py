import argparse
import json
import pickle

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import BertConfig, BertForQuestionAnswering

import model_compressor  # isort:skip

from .custom_symbolic_trace import custom_symbolic_trace  # isort:skip
from .utils import get_kwargs, random_seed, set_optimization  # isort:skip


def load_pytorch_model(model_path, model_config_path, use_gpu):
    with open(model_config_path) as f:
        config_json = json.load(f)

    config = BertConfig(**config_json)
    device = torch.device("cuda:0") if use_gpu else torch.device("cpu")

    model = BertForQuestionAnswering(config)
    model.to(device)
    model.load_state_dict(torch.load(model_path), strict=False)
    return model


def cal_data_loader(data_path, batch_size, n_calib):
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

    return DataLoader(data_list, batch_size=batch_size)


def calibrate(model, qconfig, qparam_path, qformat_path, calib_dataloader):
    model, _, _ = custom_symbolic_trace(model)
    model.config.use_cache = False

    model = model_compressor.create_quantsim_model(
        model,
        dataloader=calib_dataloader,
        disable_inout=(True, True),
        **get_kwargs(model_compressor.create_quantsim_model, qconfig),
    )

    model_compressor.calibrate(
        model,
        calib_dataloader=calib_dataloader,
        **get_kwargs(model_compressor.calibrate, qconfig),
    )

    model_compressor.save(
        model,
        qformat_out_path=qformat_path,
        qparam_out_path=qparam_path,
        **get_kwargs(model_compressor.save, qconfig),
    )

    return


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=["pytorch"], default="pytorch", help="Backend"
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

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    if args.backend == "pytorch":
        if not args.gpu:
            raise ValueError(
                "Inference on a device other than GPU is not supported yet."
            )
        model = load_pytorch_model(args.model_path, args.model_config_path, args.gpu)

    else:
        raise ValueError("Unsupported backend: {:}".format(args.backend))

    random_seed()
    set_optimization(args.torch_numeric_optim)

    with open(args.quant_config_path, "r") as f:
        qconfig = yaml.safe_load(f)
    dataloader = cal_data_loader(
        args.calib_data_path, qconfig["calib_batch_size"], args.n_calib
    )
    calibrate(
        model,
        qconfig,
        args.quant_param_path,
        args.quant_format_path,
        dataloader,
    )


if __name__ == "__main__":
    main()
