import yaml
import os 
from torch.utils.data import DataLoader
from .custom_symbolic_trace import custom_symbolic_trace
import model_compressor
import torch
from .calib_dataloader import make_dataloader
#calib_dataset_path
from transformers.generation.utils import *


def load_model_script(model_script_path):
    with open(model_script_path, 'r') as f:
        model_script = yaml.safe_load(f)

    return model_script


def get_quant_model(sut, model_script_path, n_calib, recalibrate):
    #Load model script and calibration dataloader
    model_script = load_model_script(model_script_path)
    qlevel = model_script["qlevel"]

    output_path='./quantization/output'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    qformat_path = f"{output_path}/qformat_{model_script_path.split('.')[1].split('/')[-1]}.yaml" 
    qparam_path = f"{output_path}/qparam_{model_script_path.split('.')[1].split('/')[-1]}.npy"

    model, input_names, concrete_args = custom_symbolic_trace(sut.model)

    if os.path.exists(qformat_path) and os.path.exists(qparam_path) and recalibrate == False:
        calib_dataloader = None
        org_model = None
    else:
        calib_dataloader = make_dataloader(sut.qsl, model_script['calib_batch_size'], n_calib)
        import copy
        org_model = copy.deepcopy(model) if qlevel >=3 else None

    model.config.use_cache = False

    quant_model = model_compressor.create_quantsim_model(
        model,
        qformat_path = qformat_path if calib_dataloader is None else None,
        qparam_path = qparam_path if calib_dataloader is None else None,
        weight_calib_method=model_script["weight_calib_method"],
        weight_granularity=model_script["weight_granularity"],
        weight_dtype=model_script["weight_dtype"],
        weight_nbits=model_script["weight_nbits"],
        act_calib_method=model_script["act_calib_method"],
        act_granularity=model_script["act_granularity"],
        act_dtype=model_script["act_dtype"],
        act_nbits=model_script["act_nbits"],
        kv_dtype = model_script["kv_dtype"] if "kv_dtype" in model_script else 'bf16',
        qlevel=model_script["qlevel"],
        target_machine=model_script["target_machine"],
        dataloader=calib_dataloader,
        disable_inout=(True,True),
        )
    

    if calib_dataloader:

        model_compressor.calibrate(
            quant_model,
            calib_dataloader=calib_dataloader,
            weight_calib_method=model_script["weight_calib_method"],
            weight_granularity=model_script["weight_granularity"],
            weight_dtype=model_script["weight_dtype"],
            weight_nbits=model_script["weight_nbits"],
            act_calib_method=model_script["act_calib_method"],
            act_granularity=model_script["act_granularity"],
            act_dtype=model_script["act_dtype"],
            act_nbits=model_script["act_nbits"],
            kv_dtype = model_script["kv_dtype"] if "kv_dtype" in model_script else 'bf16',
            percentile=model_script["percentile"],
            target_machine=model_script["target_machine"],
        )

        model_compressor.save(
            quant_model,
            qformat_out_path=qformat_path,
            qparam_out_path=qparam_path,
            weight_calib_method=model_script["weight_calib_method"],
            weight_granularity=model_script["weight_granularity"],
            weight_dtype=model_script["weight_dtype"],
            weight_nbits=model_script["weight_nbits"],
            act_calib_method=model_script["act_calib_method"],
            act_granularity=model_script["act_granularity"],
            act_dtype=model_script["act_dtype"],
            act_nbits=model_script["act_nbits"],
        )

        quant_model.recompile()

    if org_model:
        quant_model = model_compressor.create_quantsim_model(
        org_model,
        qformat_path=qformat_path,
        qparam_path=qparam_path,
        weight_calib_method=model_script["weight_calib_method"],
        weight_granularity=model_script["weight_granularity"],
        weight_dtype=model_script["weight_dtype"],
        weight_nbits=model_script["weight_nbits"],
        act_calib_method=model_script["act_calib_method"],
        act_granularity=model_script["act_granularity"],
        act_dtype=model_script["act_dtype"],
        act_nbits=model_script["act_nbits"],
        kv_dtype = model_script["kv_dtype"] if "kv_dtype" in model_script else 'bf16',
        qlevel=model_script["qlevel"],
        target_machine=model_script["target_machine"],
        dataloader=None,
        disable_inout=(True,True),
        )
    


    return quant_model
