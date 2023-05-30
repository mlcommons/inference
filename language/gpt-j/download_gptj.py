import sys
import os
import torch
import transformers
from transformers import AutoModelForCausalLM

# if len(sys.argv)<2:
#     print("Please provide a valid path for downloaded model")
#     print("usage : python download_gptj.py <path_where_to_save_model>")
#     exit()
# else:  
#     model_path = sys.argv[1]
#     if not os.path.exists(os.path.dirname(model_path)):
#         print("Error : Please provide a valid path")
#         exit()


model_path = os.path.join(os.getcwd(),"model")

if not os.path.exists(os.path.dirname(model_path)):
    os.mkdir(model_path)

model_name = "EleutherAI/gpt-j-6B"
model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto",torchscript=True)  # torchscript will force `return_dict=False` to avoid jit errors
print("Loaded model")

model.save_pretrained(model_path)

print("Model downloaded and Saved in : ",model_path)