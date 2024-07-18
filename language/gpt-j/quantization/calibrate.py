import argparse
import os
import sys

import torch
import yaml
from torch.fx import GraphModule
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import model_compressor  # isort:skip
from dataset import Dataset  # isort:skip
from quantization.utils import get_kwargs, random_seed, set_optimization  # isort:skip
from quantization.quantize import quantize_model


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
    from furiosa_llm_models.gptj.symbolic.huggingface_rope_rngd_gelu import GPTJForCausalLM
    
    model = GPTJForCausalLM.from_pretrained(
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

def load_mlperf_submission_model(model_path, use_gpu):
    from backend_RNGD import GPTJForCausalLM 
    
    model = GPTJForCausalLM.from_pretrained(
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


def make_calib_dataloader(calib_dataset_path, batch_size, n_calib):
    data_object = Dataset(calib_dataset_path, batch_size)
    data_list = []

    for idx in range(n_calib):
            bucket_size=2048
            starting_input_ids = data_object.source_encoded_input_ids[idx]
            batch_size, starting_input_len = starting_input_ids.shape
            bucketized_input_ids = torch.zeros((batch_size, bucket_size), dtype=torch.int)
            bucketized_input_ids[:, :starting_input_len] = starting_input_ids
            
            starting_attention_mask =  data_object.source_encoded_attn_masks[idx]
            bucketized_attention_mask = torch.zeros((batch_size, bucket_size), dtype=torch.int)
            bucketized_attention_mask[:, :starting_input_len] = starting_attention_mask
            
            starting_position_ids = torch.arange(len(data_object.source_encoded_input_ids[idx][0])).reshape(1,-1)
            bucketized_position_ids = torch.cat([starting_position_ids, torch.zeros((1, bucket_size - starting_input_len), dtype=torch.long)], dim=1)
            
            data_list.append({'input_ids': bucketized_input_ids,
                            'attention_mask': bucketized_attention_mask,
                            'position_ids': bucketized_position_ids.squeeze(0)})
    
    return DataLoader(data_list, batch_size)


def calibrate(model: GraphModule, model_type, qconfig, qparam_path, qformat_path, calib_dataloader, save_cache_files):
    run_autoscale = qconfig.get("autoscale", "disabled") != "disabled"
    if run_autoscale:
        autoscale_calib_cfg = get_autoscale_calib_config(
            qconfig, model, calib_dataloader
        )

    model_for_calib = model.trace_prefill()

    model_for_calib = model_compressor.create_quantsim_model(
        model_for_calib,
        dataloader=calib_dataloader,
        disable_inout=(True, False),
        **get_kwargs(model_compressor.create_quantsim_model, qconfig),
    )

    model_compressor.calibrate(
        model_for_calib,
        calib_dataloader=calib_dataloader,
        autoscale_calib_kwargs=autoscale_calib_cfg if run_autoscale else None,
        model_type=model_type,
        **get_kwargs(model_compressor.calibrate, qconfig),
    )

    model_compressor.save(
        model_for_calib,
        qformat_out_path=qformat_path,
        qparam_out_path=qparam_path,
        weight_calib_method=qconfig["weight_calib_method"],
        weight_granularity=qconfig["weight_granularity"],
        weight_dtype=qconfig["weight_dtype"],
        weight_nbits=qconfig["weight_nbits"],
        act_calib_method=qconfig["act_calib_method"],
        act_granularity=qconfig["act_granularity"],
        act_dtype=qconfig["act_dtype"],
        act_nbits=qconfig["act_nbits"],
        kv_dtype=qconfig["kv_dtype"] if  "kv_dtype" in qconfig else 'bf16',
        disable_inout=(True, False),
        )
    
    if save_cache_files:

        traced_models = model.trace_all()
        quant_models = quantize_model(traced_models, qparam_path, qformat_path,)

        qlv4_prefill_out_path = qparam_path.replace("quant_param_golden.npy", "prefill.bin")
        qlv4_decode_out_path = qparam_path.replace("quant_param_golden.npy", "decode.bin")
        prefill_rblock_json_out_path = qparam_path.replace("quant_param_golden.npy", "prefill_graph_patterns.json")
        decode_rblock_json_out_path = qparam_path.replace("quant_param_golden.npy", "decode_graph_patterns.json")

        torch.save(quant_models["prefill"].state_dict(), qlv4_prefill_out_path)
        torch.save(quant_models["decode"].state_dict(), qlv4_decode_out_path)
        model_compressor.save_graph_patterns(quant_models["prefill"], prefill_rblock_json_out_path)
        model_compressor.save_graph_patterns(quant_models["decode"], decode_rblock_json_out_path)


    del model_for_calib

    return


def immigrate_qparams(model, golden_qparam_path, golden_qformat_path, quant_param_path, quant_format_path, qconfig):
        
        prefill_model = model_compressor.create_quantsim_model(
            model.trace_prefill(),
            qformat_path = golden_qformat_path,
            qparam_path = golden_qparam_path,
            qlevel=2,
            target_machine=qconfig["target_machine"],
            delete_org_weight=True,
            immigrate_qparams = True,
        )

        model_compressor.save(
                prefill_model,
                qparam_out_path=quant_param_path,
                qformat_out_path=quant_format_path,
                weight_calib_method=qconfig["weight_calib_method"],
                weight_granularity=qconfig["weight_granularity"],
                weight_dtype=qconfig["weight_dtype"],
                weight_nbits=qconfig["weight_nbits"],
                act_calib_method=qconfig["act_calib_method"],
                act_granularity=qconfig["act_granularity"],
                act_dtype=qconfig["act_dtype"],
                act_nbits=qconfig["act_nbits"],
                kv_dtype=qconfig["kv_dtype"] if  "kv_dtype" in qconfig else 'bf16',
                disable_inout=(True, False),
            )
        
        


def get_args():
    parser = argparse.ArgumentParser()

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
    sut = None
    golden_model = load_pytorch_model(args.model_path, args.gpu)

    model_type = type(golden_model)

    random_seed()
    set_optimization(args.torch_numeric_optim)

    with open(args.quant_config_path, "r") as f:
        qconfig = yaml.safe_load(f)
    
    dataloader = make_calib_dataloader(args.calib_data_path, qconfig["calib_batch_size"], args.n_calib)


    golden_quant_param_path = args.quant_param_path.replace('.npy', '_golden.npy')
    golden_quant_format_path = args.quant_format_path.replace('.yaml', '_golden.yaml')

    calibrate(golden_model, model_type, qconfig, golden_quant_param_path, golden_quant_format_path, dataloader, args.save_cache_files)
    
    submission_model = load_mlperf_submission_model(args.model_path, args.gpu)

    immigrate_qparams(submission_model, golden_quant_param_path, golden_quant_format_path, args.quant_param_path, args.quant_format_path, qconfig)



if __name__ == "__main__":
    main()
