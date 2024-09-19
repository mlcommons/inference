import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader
from torch.nn.functional import pad
import model_compressor
from quantization.utils import get_kwargs, random_seed, set_optimization
from quantization.quantize import quantize_model

import gc
import json
from transformers import LlamaConfig

# Assume BLOCK_SIZE, NUM_BLOCKS, BUCKET_SIZE are fixed for now.
BLOCK_SIZE = 1
# bucket size would simply be a max value such as 2048 since we only provide one bucket
BUCKET_SIZE = 2048


def load_pytorch_model(model_source, model_path, use_gpu, n_layers):
    if use_gpu:
        assert torch.cuda.is_available(), "torch gpu is not available, exiting..."
    amp_dtype = torch.float32

    if model_source == 'furiosa_llm_rope':
        from furiosa_llm_models.llama.symbolic.huggingface_rope import LlamaForCausalLM
    elif model_source == 'mlperf_submission':
        from furiosa_llm_models.llama.symbolic.mlperf_submission import LlamaForCausalLM
    elif model_source == 'mlperf_submission_slice':
        from furiosa_llm_models.llama.symbolic.mlperf_submission_slice import LlamaForCausalLM
    else:
        raise ValueError
    
    if n_layers>0:
        from transformers import AutoConfig
        config_exp =  AutoConfig.from_pretrained(model_path)
        config_exp.num_hidden_layers = n_layers

        model = LlamaForCausalLM.from_pretrained(
            model_path, 
            config=config_exp
        )
        if use_gpu:
            print(f"Casting models to GPU...")
            assert torch.cuda.is_available(), "torch gpu is not available, exiting..."
            device = torch.device("cuda:0")
            model.to(device)
    else:
        model = LlamaForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                low_cpu_mem_usage=True,
                torch_dtype=amp_dtype
            )

    print("Loaded model")

    model.eval()
    model = model.to(memory_format=torch.channels_last)
    
    if hasattr(model, 'hf_device_map'):
            model.device_map = model.hf_device_map
            model.module_name =  model.__class__.__module__ + "." + model.__class__.__name__
    
    return model



def make_calib_dataloader(model, data_path, batch_size, n_calib,):
    if not os.path.isfile(data_path):
        print("Calibration dataset {} not found. Please check that the path is correct".format(data_path))
    
    import pandas as pd
    calib_dataset = pd.read_pickle(data_path)
    
    input_tokens = calib_dataset['tok_input']
    max_length = 2048
    
    data_list = []

    for input_token in input_tokens[:n_calib]:
        padding_size = padding_size = max_length - len(input_token)
        data_list.append(
            {
                "input_ids": pad(torch.tensor(input_token, dtype=torch.int32), (padding_size,0), value=2 ).view(1,-1).squeeze(0),
                "attention_mask": pad(torch.ones((1,len(input_token)), dtype=torch.int32), (padding_size,0) ).squeeze(0),
                'position_ids': pad(torch.arange(0, len(input_token), 1), (padding_size,0)),
            }
                
            )
            
    return DataLoader(data_list, batch_size=batch_size)


def get_autoscale_calib_config(model, autoscale, smoothquant_alpha):
    from quantization.autoscale import extract_kwargs 
    autoscale_calib_cfg = extract_kwargs.get_autoscale_calib_cfg(model, autoscale=autoscale, smoothquant_alpha=smoothquant_alpha)
    return autoscale_calib_cfg


def calibrate(model, qconfig, qparam_path, qformat_path, calib_dataloader):
    if 'autoscale' in qconfig:
        smoothquant_alpha = qconfig.get("smoothquant_alpha", 0.5)
        autoscale_calib_kwargs = get_autoscale_calib_config(model, autoscale=qconfig["autoscale"], smoothquant_alpha=smoothquant_alpha)
    else:
        autoscale_calib_kwargs = None

    model_type = type(model)
    model, _,_ = model_compressor.helper.llama_custom_symbolic_trace(
        model,
        input_names=["input_ids", "attention_mask", "position_ids"], 
        disable_check=True
    )

  
    model = model_compressor.create_quantsim_model(
        model,
        dataloader=calib_dataloader,
        disable_inout=(True, True),
        **get_kwargs(model_compressor.create_quantsim_model, qconfig),
    )

    nodes_excluded_from_auto_scale_calib = [
        'gate_proj',
        'up_proj',
        'o_proj',
        'down_proj',
        'lm_head',
    ]

    model_compressor.calibrate(
        model,
        **get_kwargs(model_compressor.calibrate, qconfig),
        model_type = model_type,
        autoscale_calib_kwargs=autoscale_calib_kwargs,
        nodes_excluded_from_auto_scale_calib=nodes_excluded_from_auto_scale_calib,
    )

    qformat, qparam = model_compressor.extract_qformat_and_qparam(model)
    model_compressor.save_qformat_qparam(qformat_dict=qformat,
                                         qformat_out_path=qformat_path,
                                         qparam_dict=qparam, 
                                         qparam_out_path=qparam_path,
                                         **get_kwargs(model_compressor.save_qformat_qparam, qconfig),
                                         )

    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache() 

    return


def immigrate_qparams(model, golden_qparam_path, golden_qformat_path, quant_param_path, quant_format_path, qconfig, save_cache_files, output_path):
        
    prefill_model = model_compressor.create_quantsim_model(
        model.trace_prefill(),
        qformat_path = golden_qformat_path,
        qparam_path = golden_qparam_path,
        qlevel=2,
        target_machine=qconfig["target_machine"],
        immigrate_qparams = True,
    )

    qformat, qparam = model_compressor.extract_qformat_and_qparam(prefill_model)
    print(f'here: {quant_format_path}')
    model_compressor.save_qformat_qparam(qformat_dict=qformat,
                                         qformat_out_path=quant_format_path,
                                         qparam_dict=qparam, 
                                         qparam_out_path=quant_param_path,
                                         **get_kwargs(model_compressor.save_qformat_qparam, qconfig),
                                         )

    if save_cache_files:

        traced_models = model.trace_all()
        quant_models = quantize_model(traced_models, quant_param_path, quant_format_path, output_path=output_path)

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

    parser.add_argument("--model_path", help="path to model")
    parser.add_argument("--quant_config_path", help="a config for model quantization")
    parser.add_argument(
        "--quant_param_path", help="quantization parameters for calibraed layers"
    )
    parser.add_argument(
        "--quant_format_path", help="quantization specifications for calibrated layers"
    )
    parser.add_argument("--calib_data_path", help="path to calibration data")
    parser.add_argument(
        "--gpu", action="store_true", help="use GPU instead of CPU for the inference"
    )
    parser.add_argument(
        "--n_layers", 
        type=int, 
        default=-1,
        help="the number of layers"
    )
    parser.add_argument(
        "--n_calib", 
        type=int, 
        default=1000,
        help="the number of calibration samples"
    )
    parser.add_argument(
        "--submission_model_source",
        default = "mlperf_submission", 
        help="quantization parameters for calibraed layers"
    )
    parser.add_argument(
        "--save_cache_files",
        action="store_true",
        default=False,
        help="if true qlv4 state_dict and rblock .json will be saved",
    )
    parser.add_argument(
        "--output_path",
        default='./',
        help="skeleton, bin 파일 저장 장소",
    )

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    # print(args.quant_config_path)
    # exit()
    golden_model = load_pytorch_model(
                            model_source = 'furiosa_llm_rope', 
                            model_path = args.model_path, 
                            use_gpu = args.gpu, 
                            n_layers = args.n_layers
                            )
    

    random_seed()
    set_optimization(False)

    with open(args.quant_config_path, "r") as f:
        qconfig = yaml.safe_load(f)

    dataloader = make_calib_dataloader(golden_model, args.calib_data_path, qconfig["calib_batch_size"], args.n_calib,)

    
    golden_quant_param_path = args.quant_param_path.replace('.npy', '_golden.npy')
    golden_quant_format_path = args.quant_format_path.replace('.yaml', '_golden.yaml')

    calibrate(
        golden_model,
        qconfig,
        golden_quant_param_path,
        golden_quant_format_path,
        dataloader,
    )

    golden_model.cpu()
    del golden_model
    gc.collect()
    torch.cuda.empty_cache() 

    submission_model = load_pytorch_model(
                        model_source = args.submission_model_source, 
                        model_path = args.model_path, 
                        use_gpu = args.gpu, 
                        n_layers = args.n_layers
                        )



    immigrate_qparams(submission_model, golden_quant_param_path, golden_quant_format_path, args.quant_param_path, args.quant_format_path, qconfig, args.save_cache_files, output_path=args.output_path)

if __name__ == "__main__":
    main()