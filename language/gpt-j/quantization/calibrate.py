import argparse
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

import model_compressor  # isort:skip

from dataset import Dataset  # isort:skip
from quantization.custom_symbolic_trace import custom_symbolic_trace  # isort:skip
from quantization.utils import get_kwargs, random_seed, set_optimization  # isort:skip


def get_autoscale_calib_config(model_script, model, calib_dataloader):
    from quantization.autoscale import extract_kwargs

    class dotdict(dict):
        """dot.notation access to dictionary attributes"""

        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    args = dotdict(model_script)
    autoscale_calib_cfg = extract_kwargs.get_autoscale_calib_cfg(
        args, model, calib_dataloader
    )
    return autoscale_calib_cfg


def load_pytorch_model(model_path, use_gpu):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto" if not use_gpu else None,
        low_cpu_mem_usage=True if not use_gpu else False,
        torch_dtype=torch.float32,
    )

    if use_gpu:
        print(f"Casting models to GPU...")
        assert torch.cuda.is_available(), "torch gpu is not available, exiting..."
        device = torch.device("cuda:0")
        model.to(device)

    model.eval()
    model = model.to(memory_format=torch.channels_last)
    return model


def cal_data_loader(calib_dataset_path, batch_size, n_calib):
    data_object = Dataset(calib_dataset_path, batch_size)
    data_list = [
        {
                "input_ids": data_object.source_encoded_input_ids[idx],
                "attention_mask": data_object.source_encoded_attn_masks[idx],
                "position_ids": torch.arange(
                    len(data_object.source_encoded_input_ids[idx][0])
                ),
            }
        for idx in range(len(data_object.source_encoded_input_ids))[:n_calib]
    ]
    return DataLoader(data_list, batch_size)


def calibrate(model, qconfig, qparam_path, qformat_path, calib_dataloader):
    run_autoscale = qconfig.get("autoscale", "disabled") != "disabled"
    if run_autoscale:
        autoscale_calib_cfg = get_autoscale_calib_config(
            qconfig, model, calib_dataloader
        )

    model, _, _ = custom_symbolic_trace(model)

    model = model_compressor.create_quantsim_model(
        model,
        dataloader=calib_dataloader,
        disable_inout=(True, True),
        **get_kwargs(model_compressor.create_quantsim_model, qconfig),
    )

    model_compressor.calibrate(
        model,
        calib_dataloader=calib_dataloader,
        autoscale_calib_kwargs=autoscale_calib_cfg if run_autoscale else None,
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
    parser.add_argument("--model_path", help="path to gpt-j model")
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
    sut = None

    if args.backend == "pytorch":
        if not args.gpu:
            raise ValueError(
                "Inference on a device other than GPU is not supported yet."
            )
        model = load_pytorch_model(args.model_path, args.gpu)
    else:
        raise ValueError("Unsupported backend: {:}".format(args.backend))

    random_seed()
    set_optimization(args.torch_numeric_optim)

    with open(args.quant_config_path, "r") as f:
        qconfig = yaml.safe_load(f)
    dataloader = cal_data_loader(
        args.calib_data_path, qconfig["calib_batch_size"], args.n_calib
    )
    calibrate(model, qconfig, args.quant_param_path, args.quant_format_path, dataloader)


if __name__ == "__main__":
    main()
