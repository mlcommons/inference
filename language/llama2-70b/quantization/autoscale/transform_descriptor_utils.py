import json
import logging

import torch
from .model_dict import LlamaForCausalLM_dict

__all__ = ["create_descriptor_from_args", "load_predefined_settings"]

logger = logging.getLogger(__name__)


def _define_transform_descriptor_from_model_type(
    model=None,
    autoscale=None,
    if_autoscale_preprocessor=True,
    if_outlier_compensated=False,
    init_dict_with={},
):
    from transformers.models.bloom.modeling_bloom import BloomForCausalLM
    from transformers.models.opt.modeling_opt import OPTForCausalLM
    from transformers.models.bert.modeling_bert import BertForQuestionAnswering
    from transformers.models.gptj.modeling_gptj import GPTJForCausalLM


    if_autoscale_postprocessor = not if_autoscale_preprocessor
    transform_descriptor = []
    # define transform descriptor according to model instance for autoscale or autoclip
    if (
        isinstance(model, OPTForCausalLM)
        or type(model) in  LlamaForCausalLM_dict.keys()
        or isinstance(model, GPTJForCausalLM)
    ):
        if isinstance(model, OPTForCausalLM):
            _model_type = 'OPTForCausalLM'
        elif type(model) in  LlamaForCausalLM_dict.keys():
            _model_type = 'LlamaForCausalLM'
        elif isinstance(model, GPTJForCausalLM):
            _model_type = 'GPTJForCausalLM'

        if if_autoscale_preprocessor:
            transform_descriptor.extend(
                [
                    {
                        'model_type': _model_type,
                        'method': 'qkv_integration',
                        'nodes_to_replace': ['q_proj', 'k_proj', 'v_proj'],
                        'new_nodes_to_create': ['q_proj'],
                    }
                ]
            )

        elif if_autoscale_postprocessor:
            if if_outlier_compensated:
                _method = 'qkv_seperation_with_outlier_module'
            else:
                _method = 'qkv_seperation'

            transform_descriptor.extend(
                [
                    {
                        'model_type': _model_type,
                        'method': _method,
                        'nodes_to_replace': ['q_proj'],
                        'new_nodes_to_create': ['q_proj', 'k_proj', 'v_proj'],
                    }
                ]
            )

    elif isinstance(model, BloomForCausalLM):
        if if_autoscale_preprocessor:
            pass
        elif if_autoscale_postprocessor:
            pass
    elif isinstance(model, BertForQuestionAnswering):
        if if_autoscale_preprocessor:
            transform_descriptor.extend(
                [
                    {
                        'model_type': 'BertForQuestionAnswering',
                        'method': 'qkv_integration',
                        'nodes_to_replace': ['query', 'key', 'value'],
                        'new_nodes_to_create': ['query'],
                    }
                ]
            )

        elif if_autoscale_postprocessor:
            if if_outlier_compensated:
                _method = 'qkv_seperation_with_outlier_module'
            else:
                _method = 'qkv_seperation'

            transform_descriptor.extend(
                [
                    {
                        'model_type':'BertForQuestionAnswering',
                        'method': _method,
                        'nodes_to_replace': ['query'],
                        'new_nodes_to_create': ['query', 'key', 'value'],
                    }
                ]
            ) 
    else:
        raise NotImplementedError(type(model))

    for update_key, update_value in init_dict_with.items():
        [
            trs_desc.update({update_key: update_value})
            for trs_desc in transform_descriptor
            if update_key in trs_desc.keys()
        ]

    return transform_descriptor


def _define_transform_descriptor_from_method(transform_method=None, init_dict_with={}):
    # define transform descriptor according to transform_method

    if transform_method == 'qkv_integration':
        transform_descriptor = {
            'method': 'qkv_integration',
            'nodes_to_replace': ['q_proj', 'k_proj', 'v_proj'],
            'new_nodes_to_create': ['q_proj'],
        }
    elif transform_method == 'qkv_seperation':
        transform_descriptor = {
            'method': 'qkv_seperation',
            'nodes_to_replace': ['q_proj'],
            'new_nodes_to_create': ['q_proj', 'k_proj', 'v_proj'],
        }
    elif transform_method == 'qkv_seperation_with_outlier_module':
        transform_descriptor = {
            'method': 'qkv_seperation_with_outlier_module',
            'nodes_to_replace': ['q_proj'],
            'new_nodes_to_create': ['q_proj', 'k_proj', 'v_proj'],
        }
    elif transform_method == 'set_proxy_target':
        transform_descriptor = {
            'method': 'set_proxy_target',
            'module2inspect': [],
            'layers2inspect': [],
            'nodes_using_proxy_target': [],
            'layer_kwargs': {},
        }
    else:
        raise NotImplementedError(transform_method)

    for update_key, update_value in init_dict_with.items():
        transform_descriptor.update({update_key: update_value})

    return transform_descriptor


def load_predefined_settings(
    model: torch.nn.Module, autoscale, if_outlier_compensated=False
):
    from transformers.models.bloom.modeling_bloom import BloomForCausalLM
    from transformers.models.gptj.modeling_gptj import GPTJForCausalLM
    from transformers.models.opt.modeling_opt import OPTForCausalLM
    from transformers.models.bert.modeling_bert import BertForQuestionAnswering
    
    preprocessor = []
    postprocessor = []

    if (
        isinstance(model, OPTForCausalLM)
        or type(model) in  LlamaForCausalLM_dict.keys()
        or isinstance(model, BertForQuestionAnswering)
        or isinstance(model, GPTJForCausalLM)
    ):
        preprocessor.extend(
            _define_transform_descriptor_from_model_type(
                model, autoscale, if_autoscale_preprocessor=True
            )
        )
        postprocessor.extend(
            _define_transform_descriptor_from_model_type(
                model,
                if_autoscale_preprocessor=False,
                if_outlier_compensated=if_outlier_compensated,
            )
        )

    elif isinstance(model, BloomForCausalLM):
        pass

    else:
        logger.warning(f'There is no predefined graph processor for {type(model)}')

    return preprocessor, postprocessor


def create_descriptor_from_args(autoscale, customized_model_node_kwargs_json_path):
    """
    Only implemented for qkv node seperation/integration transform
    """

    preprocessor = []
    postprocessor = []

    with open(customized_model_node_kwargs_json_path, encoding="utf-8") as f:
        try:
            customized_model_node_kwargs = json.loads(f.read())
        except Exception as e:
            print(e, f"Invalid json file. {customized_model_node_kwargs_json_path}")

    qkv_node_name = [
        customized_model_node_kwargs.pop('q_node', None),
        customized_model_node_kwargs.pop('k_node', None),
        customized_model_node_kwargs.pop('v_node', None),
    ]

    if not all([name is not None for name in qkv_node_name]):
        raise ValueError("Invalid q, k, v node name. Check args.customized_model_node_kwargs.")

    if autoscale == 'AWQ':
        qkv_node_suffix = [node_name.split('.')[-1] for node_name in qkv_node_name]
        if len(qkv_node_suffix) > 1:
            # Need to integrate qkv for autoscale. Preprocessor will redefine integrated linear node at qkv_node_suffix[0].

            preprocessor.append(
                _define_transform_descriptor_from_method(
                    'qkv_integration',
                    {
                        'nodes_to_replace': qkv_node_suffix,
                        'new_nodes_to_create': [qkv_node_suffix[0]],
                    },
                )
            )
            postprocessor.append(
                _define_transform_descriptor_from_method(
                    'qkv_seperation',
                    {
                        'nodes_to_replace': [qkv_node_suffix[0]],
                        'new_nodes_to_create': qkv_node_suffix,
                    },
                )
            )

    return preprocessor, postprocessor