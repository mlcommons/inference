from typing import Optional, List, Union
import migraphx as mgx

import os
import torch
import logging
import sys
import backend
import time
import random
import json
import re

from hip import hip
from PIL import Image
from functools import wraps
from collections import namedtuple
from transformers import CLIPTokenizer, CLIPTextModelWithProjection, CLIPProcessor, CLIPFeatureExtractor
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from argparse import ArgumentParser
from StableDiffusionMGX import StableDiffusionMGX
from huggingface_hub import hf_hub_download, list_repo_files
import numpy as np

HipEventPair = namedtuple('HipEventPair', ['start', 'end'])

logging.basicConfig(level=logging.ERROR)
log = logging.getLogger("backend-mgx")


formatter = logging.Formatter("{levelname} - {message}", style="{")
file_handler = logging.FileHandler("backend_mgx.log", mode="a", encoding="utf-8")
file_handler.setLevel("INFO")
file_handler.setFormatter(formatter)
log.addHandler(file_handler)

def download_model(repo_id, model_path):    
    # Zixian: Nov 10: Comment this out because model_path is current dir. 
    if os.path.exists(model_path):
        log.info(f"MGX models already exists at {model_path}")
        return
    else:
        os.makedirs(model_path, exist_ok=True)
    
    repo_files = list_repo_files(repo_id)
    
    files_to_download = [
        file for file in repo_files
        if not file.endswith(".onnx") and not file.endswith("model_fp32_gpu.mxr")
    ]
    
    for file_name in files_to_download:
        local_file_path = os.path.join(model_path, file_name)
        local_folder = os.path.dirname(local_file_path)

        # Create directory structure if it does not exist
        os.makedirs(local_folder, exist_ok=True)

        # Download the file to the specific path
        try:
            hf_hub_download(repo_id=repo_id, filename=file_name, cache_dir=local_folder, local_dir=local_folder, local_dir_use_symlinks=False)
            # log.info(f"Downloaded {file_name} to {local_file_path}")
        except Exception as e:
            log.error(f"Failed to download {file_name}: {e}")
            
        print (f"Zixian_in_the_log: Downloaded {file_name} to {local_file_path}")

#! Yalu Ouyang [Nov 10 2024] Keep this in case we aren't allowed to modify coco.py
# class Decoder:
#     def __init__(self, vocab_path):
#         # Load the vocabulary with UTF-8 encoding to support non-ASCII characters
#         with open(vocab_path, "r", encoding="utf-8") as f:
#             vocab = json.load(f)
        
#         # Reverse the mapping: token_id -> word
#         self.id_to_word = {int(id_): word for word, id_ in vocab.items()}
    
#     def decode_tokens(self, token_ids):
#         # Ensure token_ids is a list, even if a tensor is passed
#         if isinstance(token_ids, torch.Tensor):
#             token_ids = token_ids.tolist()
        
#         # Handle both single sequences and batches
#         if isinstance(token_ids[0], list):  # Batch of sequences
#             decoded_texts = [self._decode_sequence(sequence) for sequence in token_ids]
#             return decoded_texts
#         else:  # Single sequence
#             return self._decode_sequence(token_ids)
    
#     def _decode_sequence(self, token_ids):
#         # Convert token IDs to words, handling any unknown tokens
#         words = [self.id_to_word.get(token_id, "[UNK]") for token_id in token_ids]
        
#         # Remove special tokens and `</w>` markers
#         text = " ".join(words)
#         text = re.sub(r"(<\|startoftext\|>|<\|endoftext\|>)", "", text)  # Remove special tokens
#         text = text.replace("</w>", "").strip()  # Remove `</w>` markers and extra whitespace
#         return text

class BackendMIGraphX(backend.Backend):
    def __init__(
        self,
        model_path=None,
        model_id="xl",
        guidance=5, #! To match the defaults of MiGraphX
        steps=20,
        batch_size=1,
        device="cuda",
        precision="fp32",
        negative_prompt="normal quality, low quality, worst quality, low res, blurry, nsfw, nude",
    ):
        super(BackendMIGraphX, self).__init__()
        # Zixian: Nov 10: Hard code to set model_path to current dir 
        # self.model_path = model_path
        # self.model_path = os.getcwd()
        self.model_path = os.path.join(os.getcwd(), "downloaded_model_folder")
        if self.model_path is None:            
            raise SystemExit("Provide a valid Model Path to correctly run the program, exiting now...")
        
        self.pipeline_type = None
        if model_id == "xl":
            self.model_id = "SeaSponge/scc24_mlperf_mgx_exhaustive"
            self.pipeline_type = "sdxl"
        else:
            raise ValueError(f"{model_id} is not a valid model id")
        
        download_model(self.model_id, self.model_path)
        log.info(f"[mgx backend]: Returned from download_model")
        
        
        self.device = device if torch.cuda.is_available() else "cpu"
        self.device_num = int(device[-1]) \
            if (device != "cuda" and device != "cpu") else -1
        
        # log.error(f"[mgx backend] self.device -> {self.device} | device_num -> {self.device_num}")        
        
        if precision == "fp16":
            self.dtype = torch.float16
        elif precision == "bf16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        if torch.cuda.is_available():
            self.local_rank = 0
            self.world_size = 1

        self.guidance = guidance
        self.steps = steps
        self.negative_prompt = negative_prompt
        self.max_length_neg_prompt = 77
        self.batch_size = batch_size
        
        self.mgx = None
        tknz_path1 = os.path.join(self.model_path, "tokenizer")
        tknz_path2 = os.path.join(self.model_path, "tokenizer_2")
        # self.scheduler = EulerDiscreteScheduler.from_pretrained(os.path.join(self.model_path, "scheduler"))
        self.scheduler = EulerDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
        log.info(f"Zixian: Loaded scheduler")
        self.pipe = self.Pipe()
        # self.pipe.tokenizer = CLIPTokenizer.from_pretrained(tknz_path1)
        # self.pipe.tokenizer_2 = CLIPTokenizer.from_pretrained(tknz_path2)
        self.pipe.tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer")
        self.pipe.tokenizer_2 = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer_2")
        log.info(f"Zixian: Loaded tokenizer & tokenizer2")
        # log.info(f"Zixian_in_the_log tknz_path1: {tknz_path1}")
        # log.info(f"Zixian_in_the_log tknz_path2: {tknz_path2}")
        # self.decoder1 = Decoder(os.path.join(self.model_path, "tokenizer/vocab.json"))
        # self.decoder2 = Decoder(os.path.join(self.model_path, "tokenizer_2/vocab.json"))
        self.tokenizers = [self.pipe.tokenizer, self.pipe.tokenizer_2]

    class Pipe:
        def __init__(self):
            self.tokenizer = None
            self.tokenizer_2 = None
        
    def version(self):
        return torch.__version__

    def name(self):
        return "pytorch-SUT"

    def image_format(self):
        return "NCHW"

    def load(self):
        if self.model_path is None:
            log.warning(
                "Model path not provided, running with default hugging face weights\n"
                "This may not be valid for official submissions"
            )
            
            raise SystemExit("Provide a valid Model Path to correctly run the program, exiting now...")

        else:
            if self.device_num != -1:
                # log.error(f"Hip set device to -> {self.device_num}")
                hip.hipSetDevice(self.device_num)
            
            # raise SystemExit("Stopping to check")
            
            # Parameter explanations here:
            # onnx_model_path = self.model_path
            # path to compiled .mxr can be left as None
            # Don't want to use refiner model
            use_refiner = False
            # Therefore refiner model path also None
            # refiner compiled model path also None
            
            # set fp16 according to initialization input
            fp16 = "all" if self.dtype == torch.float16 else None
            # Don't want to force .onnx to .mxr compile
            force_compile = False
            # Use exhaustive tune when compilling .onnx -> .mxr
            exhaustive_tune = True
            
            tokenizers = {"clip": self.tokenizers[0], "clip2": self.tokenizers[1]}
            
            self.mgx = StableDiffusionMGX(self.pipeline_type, onnx_model_path=self.model_path,
                compiled_model_path=None, use_refiner=use_refiner,
                refiner_onnx_model_path=None,
                refiner_compiled_model_path=None, fp16=fp16,
                force_compile=force_compile, exhaustive_tune=exhaustive_tune, tokenizers=tokenizers,
                scheduler=self.scheduler)
            
            # log.info(f"[backend_migraphx.py]: after initializing self.mgx")
            
            # self.mgx.warmup(5)
            
            # log.info(f"[backend_migraphx.py]: after mgx.warmup")
            
        return self
    
    def predict(self, inputs):
        images = []
        
        # Explanation for mgx.run() arguments        
        # negative_prompt = self.negative_prompt
        # steps = self.steps
        # scale refers to guidance scale -> scale = self.guidance
        # the default SDXLPipeline chooses a random seed everytime, we'll do so manually here
        # not using refiner, so refiner_step = 0
        # not using refiner, so aesthetic_score = 0
        # not using refiner, so negative_aesthetic_score = 0
        # defaults to not verbose
        verbose = False
        #! The main pipeline from loadgen doesn't have text prompt, only tokens
        
        for i in range(0, len(inputs), self.batch_size):
            latents_input = [inputs[idx]["latents"] for idx in range(i, min(i+self.batch_size, len(inputs)))]
            latents_input = torch.cat(latents_input).to(self.device)            
            if self.batch_size == 1:
                # prompt_token = inputs[i]["input_tokens"]
                # log.info(f"[mgx backend batchsz=1] inputs[i] -> {inputs[i]}")
                prompt_in = inputs[i]["caption"]
                # log.info(f"[mgx backend] i -> {i} | prompt_in -> {prompt_in}")
                seed = random.randint(0, 2**31 - 1)
                
                # prompt_in = self.decoder1.decode_tokens(prompt_token['input_ids'])
                
                result = self.mgx.run(prompt=prompt_in, negative_prompt=self.negative_prompt, steps=self.steps, seed=seed,
                    scale=self.guidance, refiner_steps=0,
                    refiner_aesthetic_score=0,
                    refiner_negative_aesthetic_score=0, verbose=verbose,
                    prompt_tokens=None, device=self.device, latents_in=latents_input)
                
                # result shape = (3, 1024, 1024)
                
                # img_name = f"{self.device}_{random.randint(0, 1000)}.jpg"
                # image = StableDiffusionMGX.convert_to_rgb_image(result)
                # StableDiffusionMGX.save_image(image, img_name)
                # log.info(f"[mgx backend batchsz=1] Image saved to {img_name}")
                #! COCO needs this to be 3-dimensions
                
                new_res = (result / 2 + 0.5).clamp(0, 1)
                
                # log.info(f"[mgx backend] type result: {type(result)} | result shape: {result.shape}")
                # log.info(f"[mgx backend] type new_res: {type(new_res)} | new_res shape: {new_res.shape}")
                # log.info(f"------DIVIDER--------")
                images.extend(new_res)
                
            else:
                prompt_list = []
                for prompt in inputs[i:min(i+self.batch_size, len(inputs))]:
                    assert isinstance(prompt, dict), "prompt (in inputs) isn't a dict"
                    # prompt_token = prompt["input_tokens"]
                    prompt_in = inputs[i]["caption"]
                    
                
                for prompt in prompt_list:
                    seed = random.randint(0, 2**31 - 1)
                    result = self.mgx.run(prompt=prompt, negative_prompt=self.negative_prompt, steps=self.steps, seed=seed,
                        scale=self.guidance, refiner_steps=0,
                        refiner_aesthetic_score=0,
                        refiner_negative_aesthetic_score=0, verbose=verbose,
                        prompt_tokens=None, device=self.device, latents_in=latents_input)

                    new_res = (result / 2 + 0.5).clamp(0, 1)
                    images.extend(new_res)

        return images