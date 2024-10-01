from transformers import AutoTokenizer

def get_transformer_autotokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(
            model_name,
            model_max_length=2048,
            padding_side="left",
            use_fast=False,)