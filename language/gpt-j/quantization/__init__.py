from typing import Optional, Tuple, Union

import torch
import yaml
from transformers import GPTJForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

import model_compressor  # isort:skip

from .custom_symbolic_trace import custom_symbolic_trace  # isort:skip
from .utils import get_kwargs  # isort:skip


def quantize_model(model, qconfig_path, qparam_path, qformat_path):
    with open(qconfig_path, "r") as f:
        qconfig = yaml.safe_load(f)

    model, _, concrete_args = custom_symbolic_trace(model)

    quantized_model = model_compressor.create_quantsim_model(
        model,
        qformat_path=qformat_path,
        qparam_path=qparam_path,
        disable_inout=(True, True),
        **get_kwargs(model_compressor.create_quantsim_model, qconfig),
    )

    return QuantizedGPTJForCausalLM(quantized_model, concrete_args)


class QuantizedGPTJForCausalLM(GPTJForCausalLM):
    def __init__(self, quantized_model, concrete_args):
        super().__init__(quantized_model.config)
        self.__quantized_model = quantized_model
        self.__concrete_args = concrete_args

    # https://huggingface.co/docs/transformers/en/model_doc/gptj#transformers.GPTJForCausalLM.forward
    # https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/models/gptj/modeling_gptj.py#L1104-L1118
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        arguments = locals()

        del arguments["self"]

        for arg in self.__concrete_args:
            if arguments[arg] != self.__concrete_args[arg]:
                raise AssertionError(
                    f"{arg}: {arguments[arg]} != {self.__concrete_args[arg]}"
                )
            del arguments[arg]

        if past_key_values is None:
            arguments["past_key_values"] = (
                (torch.zeros(0, device=self.__quantized_model.device),) * 2,
            ) * self.__quantized_model.config.n_layer

        outputs = self.__quantized_model.forward(**arguments)

        return CausalLMOutputWithPast(outputs) if return_dict else outputs
