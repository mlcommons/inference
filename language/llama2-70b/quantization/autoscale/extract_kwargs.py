import gc
import json
import logging
from typing import Any, Dict

import model_compressor
import torch
import furiosa_llm_models 
from .transform_descriptor_utils import load_predefined_settings
from transformers.models.bloom.modeling_bloom import BloomBlock, BloomForCausalLM
from transformers.models.gptj.modeling_gptj import GPTJAttention, GPTJForCausalLM
from transformers.models.opt.modeling_opt import OPTAttention, OPTForCausalLM
from transformers.models.bert.modeling_bert import BertForQuestionAnswering
from .model_dict import LlamaForCausalLM_dict

__all__ = ["get_autoscale_calib_cfg"]

logger = logging.getLogger(__name__)



def get_autoscale_calib_cfg(
    model: torch.nn.Module, autoscale='SmoothQuant', smoothquant_alpha=0.5,
) -> Dict[str, Any]:
    logger.info("Prepare autoscale calibration. Get_autoscale_calib_cfg.")

    calib_cfg = {}
    nodes_excluded_from_auto_scale_calib = ''
    nodes_excluded_from_auto_clip_calib = ''

    (
        graph_preprocessor,
        graph_postprocessor,
    ) = load_predefined_settings(
        model, autoscale, if_outlier_compensated=False
    )
    calib_cfg['graph_transform_descriptor'] = (graph_preprocessor, graph_postprocessor)
    calib_cfg['proxy_target_infos'] = []

    if autoscale == 'SmoothQuant':
        calib_cfg['alpha'] = smoothquant_alpha
    else:
        raise NotImplementedError("Other autscales are not implemented.")

    def is_empty(_list):
        return not bool(_list)

    if is_empty(nodes_excluded_from_auto_scale_calib):
        nodes_excluded_from_auto_scale_calib = _get_predefined_excluded_from_auto_scale_calib(
            model, autoscale
        )
    calib_cfg['nodes_excluded_from_auto_scale_calib'] = nodes_excluded_from_auto_scale_calib

    if is_empty(nodes_excluded_from_auto_clip_calib):
        nodes_excluded_from_auto_clip_calib = _get_predefined_excluded_from_auto_clip_calib(
            model
        )
    calib_cfg['nodes_excluded_from_auto_clip_calib'] = nodes_excluded_from_auto_clip_calib

    calib_cfg['unify_smooth_factor'] = False # unify_smooth_factor is implemented only for GPT-J at the moment.
    calib_cfg['module_name_to_replace_smooth_factor'] = 'fc_in'
    calib_cfg['module_name_for_smooth_factor'] = 'q_proj'
    calib_cfg['nodes_excluded_from_auto_scale_calib'].extend(['gate_proj', 'up_proj'])

    return calib_cfg


def _get_predefined_excluded_from_auto_scale_calib(torch_model, autoscale):

    if type(torch_model) in LlamaForCausalLM_dict.keys():
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
    if type(torch_model) in LlamaForCausalLM_dict.keys():
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