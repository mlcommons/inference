import gc
import json
import logging
import sys
from typing import Any, Dict

import model_compressor
import torch

from . import transform_descriptor_utils
from transformers.models.bloom.modeling_bloom import BloomBlock, BloomForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaForCausalLM
from transformers.models.opt.modeling_opt import OPTAttention, OPTForCausalLM
from transformers.models.gptj.modeling_gptj import GPTJAttention, GPTJForCausalLM
from transformers.models.bert.modeling_bert import BertForQuestionAnswering

__all__ = ["get_autoscale_calib_cfg", "valid_check_calib_cfg"]

logger = logging.getLogger(__name__)


def get_autoscale_calib_cfg(
    args, model: torch.nn.Module, loader_calib, cache_ckpt_folder_path='./cache'
) -> Dict[str, Any]:
    logger.info("Prepare autoscale calibration. Get_autoscale_calib_cfg.")

    calib_cfg = {}
    if_outlier_compensated = args.outlier_percentile is not None

    if args.autoscale == 'AWQ':
        layer_kwargs = _extract_layer_kwargs(
            model, next(iter(loader_calib))[0], cache_ckpt_folder_path
        )
        calib_cfg['layer_kwargs'] = layer_kwargs

    if args.use_customized_model:
        (
            graph_preprocessor,
            graph_postprocessor,
        ) = transform_descriptor_utils.create_descriptor_from_args(
            args.autoscale, args.customized_model_node_kwargs_json_path
        )
        calib_cfg['graph_transform_descriptor'] = (graph_preprocessor, graph_postprocessor)

    else:
        (
            graph_preprocessor,
            graph_postprocessor,
        ) = transform_descriptor_utils.load_predefined_settings(
            model, args.autoscale, loader_calib, if_outlier_compensated=if_outlier_compensated
        )
        calib_cfg['graph_transform_descriptor'] = (graph_preprocessor, graph_postprocessor)

    proxy_target_modules = []
    if args.autoscale == 'AWQ':
        if args.split_mode == 'block-by-block':
            proxy_target_modules.extend(_get_proxy_target_modules(args, model))

    calib_cfg['proxy_target_infos'] = proxy_target_modules

    if args.autoscale == 'SmoothQuant':
        calib_cfg['alpha'] = args.smoothquant_alpha

    def is_empty(_list):
        return not bool(_list)

    if is_empty(args.nodes_excluded_from_auto_scale_calib):
        args.nodes_excluded_from_auto_scale_calib = _get_predefined_excluded_from_auto_scale_calib(
            model, args.autoscale
        )
    calib_cfg['nodes_excluded_from_auto_scale_calib'] = args.nodes_excluded_from_auto_scale_calib

    if is_empty(args.nodes_excluded_from_auto_clip_calib):
        args.nodes_excluded_from_auto_clip_calib = _get_predefined_excluded_from_auto_clip_calib(
            model
        )
    calib_cfg['nodes_excluded_from_auto_clip_calib'] = args.nodes_excluded_from_auto_clip_calib

    if args.unify_smooth_factor and not isinstance(model, GPTJForCausalLM):
        raise ValueError("In current, unifying smooth factor feature only supports GPT-J.")
    calib_cfg['unify_smooth_factor'] = args.unify_smooth_factor
    calib_cfg['module_name_to_replace_smooth_factor'] = args.module_name_to_replace_smooth_factor
    calib_cfg['module_name_for_smooth_factor'] = args.module_name_for_smooth_factor

    return calib_cfg


def valid_check_calib_cfg(calib_cfg, args):
    graph_preprocessor = calib_cfg["autoscale"]["graph_transform_descriptor"][0]
    proxy_target_modules = calib_cfg["autoscale"]['proxy_target_infos']

    passed = (
        _check_use_customized_model(graph_preprocessor[0]['nodes_to_replace'])
        and _check_customized_proxy_target(proxy_target_modules)
        and _check_nodes_excluded_from_auto_scale_calib(args.nodes_excluded_from_auto_scale_calib)
        and _check_nodes_excluded_from_auto_clip_calib(args.nodes_excluded_from_auto_clip_calib)
    )
    if passed:
        print("test_customzied_autoscale_args passed")
    else:
        raise ValueError("Failed test_customzied_autoscale_args")
    sys.exit(0)

    return


def _check_use_customized_model(list_to_check):
    gt_list = ['q_proj', 'k_proj', 'v_proj']
    return all([list_to_check[idx] == gt for idx, gt in enumerate(gt_list)])


def _check_customized_proxy_target(proxy_target_modules):
    return len(proxy_target_modules[0][0]) == 14 and proxy_target_modules[0][2][0] == 'q_proj'


def _check_nodes_excluded_from_auto_scale_calib(list_to_check):
    gt_list = ['dense', 'dense_4h_to_h']
    return all([list_to_check[idx] == gt for idx, gt in enumerate(gt_list)])


def _check_nodes_excluded_from_auto_clip_calib(list_to_check):
    gt_list = ['query_key_value']
    return all([list_to_check[idx] == gt for idx, gt in enumerate(gt_list)])


def _get_predefined_proxy_target_module(torch_model):
    if isinstance(torch_model, LlamaForCausalLM):
        # Todo - need to check
        proxy_target_module_type = [LlamaAttention]
        layers2inspect = [['q_proj', 'k_proj', 'v_proj']]
        nodes_using_proxy_target = ['q_proj']
    elif isinstance(torch_model, OPTForCausalLM):
        proxy_target_module_type = [OPTAttention]
        layers2inspect = [['q_proj', 'k_proj', 'v_proj']]
        nodes_using_proxy_target = ['q_proj']
    elif isinstance(torch_model, BloomForCausalLM):
        proxy_target_module_type = [BloomBlock]
        layers2inspect = [None, None]
        nodes_using_proxy_target = ['query_key_value', 'h_to_4h']
    elif isinstance(torch_model, GPTJForCausalLM):
        proxy_target_module_type = [GPTJAttention]
        layers2inspect = [['q_proj', 'k_proj', 'v_proj']]
        nodes_using_proxy_target = ['q_proj']
    elif "mpt" in str(torch_model.__class__).lower():
        proxy_target_module_type = []
        layers2inspect = []
        nodes_using_proxy_target = []
    elif "falcon" in str(torch_model.__class__).lower():
        proxy_target_module_type = []
        layers2inspect = []
        nodes_using_proxy_target = []
    else:
        raise NotImplementedError(type(torch_model))
    return [[proxy_target_module_type, layers2inspect, nodes_using_proxy_target]]


def _get_predefined_excluded_from_auto_scale_calib(torch_model, autoscale):

    if isinstance(torch_model, LlamaForCausalLM):
        # Todo - need to check
        if autoscale == 'AWQ':
            nodes_list = ["lm_head"]
        elif autoscale == 'SmoothQuant':
            nodes_list = ["o_proj", "down_proj", "lm_head"]
    elif isinstance(torch_model, OPTForCausalLM):
        if autoscale == 'AWQ':
            nodes_list = ["lm_head"]
        elif autoscale == 'SmoothQuant':
            nodes_list = ['out_proj', 'fc2', 'lm_head']
    elif isinstance(torch_model, BloomForCausalLM):
        if autoscale == 'AWQ':
            nodes_list = ["dense"]
        elif autoscale == 'SmoothQuant':
            nodes_list = ['dense', 'dense_4h_to_h']
    elif isinstance(torch_model, GPTJForCausalLM):
        if autoscale == 'AWQ':
            nodes_list = ["lm_head"]
        elif autoscale == 'SmoothQuant':
            nodes_list = ['out_proj', 'fc_out', 'lm_head']
    elif isinstance(torch_model, BertForQuestionAnswering):
        if autoscale == 'AWQ':
            nodes_list = ["qa_outputs"]
        elif autoscale == 'SmoothQuant':
            nodes_list = ['output.dense', 'output_dense', 'qa_outputs']

    elif "mpt" in str(torch_model.__class__).lower():
        nodes_list = []
    elif "falcon" in str(torch_model.__class__).lower():
        nodes_list = []
    else:
        logger.warning(
            "There is no predefined nodes_excluded_from_auto_scale_calib. Autoscale is applied for all linear nodes."
        )
        nodes_list = []

    return nodes_list


def _get_predefined_excluded_from_auto_clip_calib(torch_model):
    if isinstance(torch_model, LlamaForCausalLM):
        nodes_list = ['q_proj', 'k_proj', 'lm_head']
    elif isinstance(torch_model, OPTForCausalLM):
        nodes_list = ['q_proj', 'k_proj', 'lm_head']
    elif isinstance(torch_model, BloomForCausalLM):
        nodes_list = ['query_key_value']
    elif isinstance(torch_model, GPTJForCausalLM):
        nodes_list = ['q_proj', 'k_proj', 'lm_head']
    elif "mpt" in str(torch_model.__class__).lower():
        nodes_list = []
    elif "falcon" in str(torch_model.__class__).lower():
        nodes_list = []
    else:
        logger.warning(
            "There is no predefined nodes_excluded_from_auto_clip_calib. Autoclip is applied for all linear nodes."
        )
        nodes_list = ["classifier"]
    return nodes_list


def _map_module_name_to_module_type(customized_proxy_target_json_path, torch_model):
    with open(customized_proxy_target_json_path, encoding="utf-8") as f:
        try:
            customized_proxy_target_info = json.loads(f.read())
        except Exception as e:
            print(e, f"Invalid json file. {customized_proxy_target_json_path}")

    collected_proxy_target_info = []
    for proxy_target_module_name, proxy_target_info in customized_proxy_target_info.items():
        nodes_using_proxy_target = proxy_target_info["nodes_using_proxy_target"]
        layers2inspect = proxy_target_info["layers2inspect"]

        proxy_target_module_type = []
        for name, module in torch_model.named_modules():
            if name.endswith(proxy_target_module_name):
                proxy_target_module_type.append(type(module))
                break

        if not proxy_target_module_type:
            raise ValueError(f"{module} does not exists.")
        collected_proxy_target_info.append(
            [proxy_target_module_type, layers2inspect, nodes_using_proxy_target]
        )

    return collected_proxy_target_info


def _get_proxy_target_modules(args, torch_model):
    if args.customized_proxy_target_for_auto_scale_json_path:
        collected_proxy_target_info = _map_module_name_to_module_type(
            args.customized_proxy_target_for_auto_scale_json_path, torch_model
        )
    else:
        collected_proxy_target_info = _get_predefined_proxy_target_module(torch_model)

    for idx_info, proxy_target_info in enumerate(collected_proxy_target_info):
        proxy_target_modules = _find_proxy_modules_for_each_subgraph(
            torch_model, proxy_target_info[0]
        )
        collected_proxy_target_info[idx_info][0] = proxy_target_modules

    return collected_proxy_target_info


def _find_proxy_modules_for_each_subgraph(
    model: torch.nn.Module, module_types, exclude_first_graph=True, exclude_last_graph=True
):
    modules = []
    if exclude_first_graph:
        modules.append(None)  # for first embedding graph
    for _, module in model.named_modules():
        if any([isinstance(module, module_type) for module_type in module_types]):
            modules.append(module)

    if exclude_last_graph:
        modules.append(None)  # for last lm_head graph

    return modules


def _get_blocks(model):

    if isinstance(model, LlamaForCausalLM):
        layers = model.model.layers
    elif isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, BloomForCausalLM):
        layers = model.transformer.h
    elif isinstance(model, GPTJForCausalLM):
        layers = model.transformer.h
    elif "mpt" in str(model.__class__).lower():
        layers = model.transformer.blocks
    elif "falcon" in str(model.__class__).lower():
        layers = model.transformer.h
    else:
        raise NotImplementedError(type(model))
    return layers


def _extract_layer_kwargs(torch_model, samples, cache_ckpt_folder_path):
    def _return_to_meta_model(model):
        for _, module in model.named_modules():
            module.to(device='meta')
        return model

    layers = _get_blocks(torch_model)

    layer_kwargs = {}

    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    class Catcher_GPTJ(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, **kwargs):
            kwargs.pop('hidden_states')
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    # patch layer 0 to catch input and kwargs
    device = next(torch_model.parameters()).device
    if device in [torch.device('meta'), 'meta']:
        model_path = model_compressor.multi_chip.get_ckpt_upto_first_decoder_block(
            torch_model, type(layers[0]), cache_ckpt_folder_path
        )
        model_compressor.utils.accelerate.big_modeling.load_checkpoint_in_model(
            torch_model, model_path
        )

    if isinstance(torch_model, GPTJForCausalLM):
        layers[0] = Catcher_GPTJ(layers[0])
    else:
        layers[0] = Catcher(layers[0])

    with torch.no_grad():
        try:
            torch_model(samples.to(device))
        except ValueError:  # work with early exit
            pass

    del samples

    layers[0] = layers[0].module.to(device)
    if device in [torch.device('meta'), 'meta']:
        _return_to_meta_model(torch_model)
    gc.collect()
    torch.cuda.empty_cache()
    return layer_kwargs