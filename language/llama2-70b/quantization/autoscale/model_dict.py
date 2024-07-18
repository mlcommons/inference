import transformers
import furiosa_llm_models


#To DO: Add dictionaries for other models in furiosa-llm-models

LlamaForCausalLM_dict = {
    transformers.models.llama.modeling_llama.LlamaForCausalLM : transformers.models.llama.modeling_llama,
    furiosa_llm_models.llama.symbolic.huggingface.LlamaForCausalLM: furiosa_llm_models.llama.symbolic.huggingface,
    furiosa_llm_models.llama.symbolic.huggingface_rope.LlamaForCausalLM: furiosa_llm_models.llama.symbolic.huggingface_rope,
    furiosa_llm_models.llama.symbolic.mlperf_submission.LlamaForCausalLM: furiosa_llm_models.llama.symbolic.mlperf_submission,
}