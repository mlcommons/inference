from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os


model_path = "mistralai/Mixtral-8x7B-Instruct-v0.1"
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", low_cpu_mem_usage=True, torch_dtype=torch.float32
)
print(model)
if not os.path.exists("model/Mixtral-8x7B/"):
    os.makedirs("model/Mixtral-8x7B/", exist_ok=True)
model.save_pretrained("model/Mixtral-8x7B/")
