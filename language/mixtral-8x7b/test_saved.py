from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os


model_path = "model/Mixtral-8x7B/"
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", low_cpu_mem_usage=True, torch_dtype=torch.float32
)


print("loaded!")
print(model)
