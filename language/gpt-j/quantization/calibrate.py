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
import json
from transformers import GPTJConfig

NUM_HIDDEN_LAYERS = 28


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


def load_pytorch_model(model_path, use_gpu, n_layers=-1):
    from furiosa_llm_models.gptj.symbolic.huggingface_rope_rngd_gelu import GPTJForCausalLM

    CONFIG_PATH = os.path.join(model_path, "config.json")
    with open(CONFIG_PATH, "r") as f:
        config_dict = json.load(f)
    custom_config = GPTJConfig.from_dict(config_dict)
    if n_layers != -1:
        custom_config.n_layer = n_layers
    
    model = GPTJForCausalLM.from_pretrained(
        model_path,
        device_map="auto" if not use_gpu else None,
        low_cpu_mem_usage=True if not use_gpu else False,
        torch_dtype=torch.float32,
        config=custom_config,
    )

    if use_gpu:
        print(f"Casting models to GPU...")
        assert torch.cuda.is_available(), "torch gpu is not available, exiting..."
        device = torch.device("cuda:0")
        model.to(device)

    model.eval()
    model = model.to(memory_format=torch.channels_last)
    return model

def load_mlperf_submission_model(model_path, use_gpu, n_layers=-1):
    from backend_RNGD import GPTJForCausalLM

    CONFIG_PATH = os.path.join(model_path, "config.json")
    with open(CONFIG_PATH, "r") as f:
        config_dict = json.load(f)
    custom_config = GPTJConfig.from_dict(config_dict)
    if n_layers != -1:
        custom_config.n_layer = n_layers
    
    model = GPTJForCausalLM.from_pretrained(
        model_path,
        device_map="auto" if not use_gpu else None,
        low_cpu_mem_usage=True if not use_gpu else False,
        torch_dtype=torch.float32,
        config=custom_config,
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


def calibrate(model: GraphModule, qconfig, qparam_path, qformat_path, calib_dataloader):
    run_autoscale = qconfig.get("autoscale", "disabled") != "disabled"
    if run_autoscale:
        autoscale_calib_cfg = get_autoscale_calib_config(
            qconfig, model, calib_dataloader
        )

    model_type = type(model)

    model_for_calib = model.trace_prefill()

    model_for_calib = model_compressor.create_quantsim_model(
        model_for_calib,
        dataloader=calib_dataloader,
        disable_inout=(True, False),
        **get_kwargs(model_compressor.create_quantsim_model, qconfig),
    )

    model_compressor.calibrate(
        model_for_calib,
        autoscale_calib_kwargs=autoscale_calib_cfg if run_autoscale else None,
        model_type=model_type,
        **get_kwargs(model_compressor.calibrate, qconfig),
    )

    qformat, qparam = model_compressor.extract_qformat_and_qparam(model_for_calib)
    model_compressor.save_qformat_qparam(qformat_dict=qformat,
                                         qformat_out_path=qformat_path,
                                         qparam_dict=qparam, 
                                         qparam_out_path=qparam_path,
                                         **get_kwargs(model_compressor.save_qformat_qparam, qconfig),
                                         )
    del model_for_calib

    return


def immigrate_qparams(model, golden_qparam_path, golden_qformat_path, quant_param_path, quant_format_path, qconfig, save_cache_files):
        
    prefill_model = model_compressor.create_quantsim_model(
        model.trace_prefill(),
        qformat_path = golden_qformat_path,
        qparam_path = golden_qparam_path,
        qlevel=2,
        target_machine=qconfig["target_machine"],
        immigrate_qparams = True,
    )

    qformat, qparam = model_compressor.extract_qformat_and_qparam(prefill_model)
    model_compressor.save_qformat_qparam(qformat_dict=qformat,
                                         qformat_out_path=quant_format_path,
                                         qparam_dict=qparam, 
                                         qparam_out_path=quant_param_path,
                                         **get_kwargs(model_compressor.save_qformat_qparam, qconfig),
                                         )
    
    if save_cache_files:

        traced_models = model.trace_all()
        quant_models = quantize_model(traced_models, quant_param_path, quant_format_path,)

        qlv4_prefill_out_path = quant_param_path.replace("quant_param.npy", "prefill.bin")
        qlv4_decode_out_path = quant_param_path.replace("quant_param.npy", "decode.bin")
        # prefill_rblock_json_out_path = quant_param_path.replace("quant_param.npy", "prefill_graph_patterns.json")
        # decode_rblock_json_out_path = quant_param_path.replace("quant_param.npy", "decode_graph_patterns.json")

        torch.save(quant_models["prefill"].state_dict(), qlv4_prefill_out_path)
        torch.save(quant_models["decode"].state_dict(), qlv4_decode_out_path)
        # model_compressor.save_graph_patterns(quant_models["prefill"], prefill_rblock_json_out_path)
        # model_compressor.save_graph_patterns(quant_models["decode"], decode_rblock_json_out_path)

        
        


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
    parser.add_argument(
        "--n_layers", 
        type=int, 
        default=-1,
        help="the number of layers"
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    sut = None
    golden_model = load_pytorch_model(args.model_path, args.gpu, args.n_layers)

    random_seed() # todos
    set_optimization(args.torch_numeric_optim)

    with open(args.quant_config_path, "r") as f:
        qconfig = yaml.safe_load(f)
    
    dataloader = make_calib_dataloader(args.calib_data_path, qconfig["calib_batch_size"], args.n_calib)


    golden_quant_param_path = args.quant_param_path.replace('.npy', '_golden.npy')
    golden_quant_format_path = args.quant_format_path.replace('.yaml', '_golden.yaml')

    calibrate(golden_model, qconfig, golden_quant_param_path, golden_quant_format_path, dataloader)
    
    submission_model = load_mlperf_submission_model(args.model_path, args.gpu, args.n_layers)

    immigrate_qparams(submission_model, golden_quant_param_path, golden_quant_format_path, args.quant_param_path, args.quant_format_path, qconfig, args.save_cache_files)



if __name__ == "__main__":
    main()
