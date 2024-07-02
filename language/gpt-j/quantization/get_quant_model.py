import yaml
import os
import torch
from torch.utils.data import DataLoader
from transformers.utils.fx import symbolic_trace
import model_compressor
from typing import Optional
from .QuantPreTrainedModel import QuantPreTrainedModel
from .custom_symbolic_trace import custom_symbolic_trace
from dataset import Dataset
import copy




gen_kwargs = {
    "early_stopping": True,
    "max_new_tokens": 128,
    "min_new_tokens": 30,
    # only beam_size 4 is allowed for official submission
    "num_beams": int(os.environ.get("GPTJ_BEAM_SIZE", "4")),
}

##To Do: the above function will be fixed later for calibration. 
def make_dummy_dataloader(data_object, batch_size, model_config, use_cache=False, gen_mode=False): 
    data_list = []
    for idx in range(len(data_object.source_encoded_input_ids)):
        if use_cache == False and gen_mode == False:
            data_list.append({'input_ids': data_object.source_encoded_input_ids[idx], 'attention_mask': data_object.source_encoded_attn_masks[idx], 'position_ids': torch.arange(
                len(data_object.source_encoded_input_ids[idx][0]))})
        elif use_cache == True and gen_mode == True:
            data_list.append({'input_ids': data_object.source_encoded_input_ids[idx][0, -1].reshape(1, 1), 'past_key_values': get_dummy_kv_cache(data_object.source_encoded_input_ids[idx], model_config), 'attention_mask': torch.ones(
                len(data_object.source_encoded_input_ids[0][0])+1).unsqueeze(0).type(torch.int), 'position_ids': torch.tensor(len(data_object.source_encoded_input_ids[idx][0])).reshape(1, 1)})
        elif use_cache == True and gen_mode == False:
            data_list.append({'input_ids': data_object.source_encoded_input_ids[idx][0, -1].reshape(1, 1).repeat(gen_kwargs["num_beams"], 1), 'past_key_values': get_dummy_kv_cache(data_object.source_encoded_input_ids[idx], model_config), 'attention_mask': torch.ones(
                len(data_object.source_encoded_input_ids[0][0])+1).unsqueeze(0).repeat(gen_kwargs["num_beams"], 1).type(torch.int), 'position_ids': torch.tensor(len(data_object.source_encoded_input_ids[idx][0])).reshape(1, 1).repeat(gen_kwargs["num_beams"], 1)})
        elif use_cache == False and gen_mode == True:
            raise ValueError(
                "Not implemented yet. Will implement when need arises.")
    return DataLoader(data_list, batch_size)

def make_calib_dataloader(calib_dataset_path, batch_size, num_layer):
    data_object = Dataset(calib_dataset_path, batch_size)
    data_list = []
    for idx in range(len(data_object.source_encoded_input_ids)):
        data_list.append({'input_ids': data_object.source_encoded_input_ids[idx], 'attention_mask': data_object.source_encoded_attn_masks[idx], 'position_ids': torch.arange(
                len(data_object.source_encoded_input_ids[idx][0]))})
    return DataLoader(data_list, batch_size)



def load_model_script(model_script_path):
    with open(model_script_path, 'r') as f:
        model_script = yaml.safe_load(f)

    return model_script


def get_dummy_kv_cache(input_ids, model_config):
    kv_cache = list(range(model_config.n_layer))
    for idx in range(len(kv_cache)):
        kv_cache[idx] = [torch.randn(gen_kwargs["num_beams"], model_config.n_head, len(
            input_ids[0]), int(model_config.n_embd/model_config.n_head)) for _ in range(2)]

    return list(kv_cache)

def get_autoscale_calib_config(model_script, model, calib_dataloader):
    from .autoscale import extract_kwargs 
    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__
    args = dotdict(model_script)
    autoscale_calib_cfg = extract_kwargs.get_autoscale_calib_cfg(args, model, calib_dataloader)
    return autoscale_calib_cfg



def get_quant_model(model, calib_dataset_path, model_script_path, recalibrate):
    # Load model script and calibration dataloader (Refer to inference-compression/language/gpt-j/README.md on how to download evaluation and calibration dataset )
    model_script = load_model_script(model_script_path)

    qformat_path = f"./quantization/output/qformat_{model_script_path.split('.')[1].split('/')[-1]}.yaml" 
    qparam_path = f"./quantization/output/qparam_{model_script_path.split('.')[1].split('/')[-1]}.npy"
  
    
    if os.path.exists(qformat_path) and os.path.exists(qparam_path) and recalibrate == False:
        calib_dataloader = None
    else:
        calib_dataloader = make_calib_dataloader(calib_dataset_path, model_script['calib_batch_size'], model.config.n_layer)
  
    run_autoscale = model_script.get("autoscale", 'disabled') != 'disabled'  
     #prepare for autoscale 
    if run_autoscale:
        autoscale_calib_cfg = get_autoscale_calib_config(model_script, model, calib_dataloader)
 
        
    model_type = type(model)
    model, input_names, concrete_args = custom_symbolic_trace(model)
    
    if calib_dataloader is not None and model_script["qlevel"] > 2:
        org_model = copy.deepcopy(model)
        org_model.config = model.config
    else:
        org_model = None
    
    # Extract necessary parameters to initialize QuantPreTrainedModel
    model = model_compressor.create_quantsim_model(
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
        qlevel=model_script["qlevel"],
        target_machine=model_script["target_machine"],
        act_zp_equalizing=model_script["act_zp_equalizing"] if  "act_zp_equalizing" in model_script else 'disabled',
        dataloader=calib_dataloader,
        disable_inout=(True, True),
        kv_dtype = model_script["kv_dtype"] if "kv_dtype" in model_script else 'bf16'
    )

    if calib_dataloader:
        model_compressor.calibrate(
            model=model,
            model_name=model_script["model"],
            calib_dataloader=calib_dataloader,
            weight_calib_method=model_script["weight_calib_method"],
            weight_granularity=model_script["weight_granularity"],
            weight_dtype=model_script["weight_dtype"],
            weight_nbits=model_script["weight_nbits"],
            act_calib_method=model_script["act_calib_method"],
            act_granularity=model_script["act_granularity"],
            act_dtype=model_script["act_dtype"],
            act_nbits=model_script["act_nbits"],
            percentile=model_script["percentile"],
            target_machine=model_script["target_machine"],
            act_zp_equalizing=model_script["act_zp_equalizing"] if  "act_zp_equalizing" in model_script else 'disabled',
            autoscale=model_script["autoscale"] if run_autoscale else "disabled",
            autoscale_calib_method=(model_script["autoscale_calib_method"] if run_autoscale else 'auto'),
            autoscale_calib_kwargs=autoscale_calib_cfg if run_autoscale else None,
        )

        model_compressor.save(
                model,
                qparam_out_path=qparam_path,
                qformat_out_path=qformat_path,
                weight_calib_method=model_script["weight_calib_method"],
                weight_granularity=model_script["weight_granularity"],
                weight_dtype=model_script["weight_dtype"],
                weight_nbits=model_script["weight_nbits"],
                act_calib_method=model_script["act_calib_method"],
                act_granularity=model_script["act_granularity"],
                act_dtype=model_script["act_dtype"],
                act_nbits=model_script["act_nbits"],
                #disable_mods=args.disable_quant_list,
            )


        model.recompile()


    if org_model:
        model = model_compressor.create_quantsim_model(
            org_model,
            qformat_path = qformat_path,
            qparam_path = qparam_path,
            weight_calib_method=model_script["weight_calib_method"],
            weight_granularity=model_script["weight_granularity"],
            weight_dtype=model_script["weight_dtype"],
            weight_nbits=model_script["weight_nbits"],
            act_calib_method=model_script["act_calib_method"],
            act_granularity=model_script["act_granularity"],
            act_dtype=model_script["act_dtype"],
            act_nbits=model_script["act_nbits"],
            qlevel=model_script["qlevel"],
            target_machine=model_script["target_machine"],
            act_zp_equalizing=model_script["act_zp_equalizing"] if  "act_zp_equalizing" in model_script else 'disabled',
            dataloader=None,
            disable_inout=(True, True),
            kv_dtype = model_script["kv_dtype"] if "kv_dtype" in model_script else 'bf16'
        )

    return QuantPreTrainedModel(model, model_type, input_names, concrete_args)
