"""
implementation of coco dataset
"""

# pylint: disable=unused-argument,missing-docstring

import json
import logging
import os
import time

from PIL import Image
import numpy as np
import pandas as pd
import dataset

import torch
from torchvision import transforms
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image.fid import FrechetInceptionDistance


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("coco")


class Coco(dataset.Dataset):
    def __init__(
        self,
        data_path,
        name=None,
        image_size=None,
        use_preprocessed=False,
        pre_process=None,
        pipe_tokenizer=None,
        pipe_tokenizer_2=None,
        refiner_tokenizer=None,
        refiner_tokenizer_2=None,
        latent_dtype=torch.float32,
        latent_device="cuda",
        latent_framework="torch",
        **kwargs,
    ):
        super().__init__()
        self.captions_df = pd.read_csv(f"{data_path}/captions/captions.tsv", sep="\t")
        self.image_size = image_size
        self.use_preprocessed = use_preprocessed
        self.preprocessed_dir = os.path.abspath(f"{data_path}/preprocessed/")
        self.img_dir = os.path.abspath(f"{data_path}/validation/data/")
        self.name = name

        # Preprocess prompts
        self.captions_df["input_tokens"] = self.captions_df["caption"].apply(
            lambda x: self.preprocess(x, pipe_tokenizer)
        )
        self.captions_df["input_tokens_2"] = self.captions_df["caption"].apply(
            lambda x: self.preprocess(x, pipe_tokenizer_2)
        )
        self.captions_df["refiner_input_tokens"] = self.captions_df["caption"].apply(
            lambda x: self.preprocess(x, refiner_tokenizer)
        )
        self.captions_df["refiner_input_tokens_2"] = self.captions_df["caption"].apply(
            lambda x: self.preprocess(x, refiner_tokenizer_2)
        )
        self.latent_dtype = latent_dtype
        self.latent_device = latent_device if torch.cuda.is_available() else "cpu"
        if latent_framework == "torch":
            self.latents = (
                torch.load(f"{data_path}/latents/latents.pt")
                .to(latent_dtype)
                .to(latent_device)
            )
        elif latent_framework == "numpy":
            self.latents = (
                torch.Tensor(np.load(f"{data_path}/latents/latents.npy"))
                .to(latent_dtype)
                .to(latent_device)
            )
        if self.use_preprocessed:
            os.makedirs(self.preprocessed_dir, exist_ok=True)
            convert_tensor = transforms.ToTensor()
            self.captions_df["preprocessed_path"] = self.captions_df["file_name"].apply(
                lambda x: self.preprocess_images(x, convert_tensor)
            )

    def preprocess(self, prompt, tokenizer):
        converted_prompt = self.convert_prompt(prompt, tokenizer)
        return tokenizer(
            converted_prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

    def preprocess_images(self, file_name, convert_tensor):
        img = Image.open(self.img_dir + "/" + file_name)
        convert_tensor
        tensor = convert_tensor(img)
        target_name = file_name.split(".")[0]
        target_path = self.preprocessed_dir + "/" + target_name + ".pt"
        if not os.path.exists(target_path):
            torch.save(tensor, target_path)
        return target_path

    def convert_prompt(self, prompt, tokenizer):
        tokens = tokenizer.tokenize(prompt)
        unique_tokens = set(tokens)
        for token in unique_tokens:
            if token in tokenizer.added_tokens_encoder:
                replacement = token
                i = 1
                while f"{token}_{i}" in tokenizer.added_tokens_encoder:
                    replacement += f" {token}_{i}"
                    i += 1

                prompt = prompt.replace(token, replacement)

        return prompt

    def get_item(self, id):
        return dict(self.captions_df.loc[id], latents=self.latents)

    def get_item_count(self):
        return len(self.captions_df)

    def get_imgs(self, id_list):
        image_list = []
        if self.use_preprocessed:
            for id in id_list:
                image_list.append(
                    torch.load(self.captions_df.loc[id]["preprocessed_path"])
                )
        else:
            convert_tensor = transforms.ToTensor()
            for id in id_list:
                img = Image.open(
                    self.img_dir + "/" + self.captions_df.loc[id]["file_name"]
                )
                image_list.append(convert_tensor(img))
        return image_list

    def get_captions(self, id_list):
        return [self.get_item(id)["caption"] for id in id_list]
    
    def get_item_loc(self, id):
        if self.use_preprocessed:
            return self.captions_df.loc[id]["preprocessed_path"]
        else:
            return self.img_dir + "/" + self.captions_df.loc[id]["file_name"]

class PostProcessCoco:
    """
    Post processing for tensorflow ssd-mobilenet style models
    """

    def __init__(self, device="cpu"):
        self.results = []
        self.good = 0
        self.total = 0
        self.content_ids = []
        self.device = device if torch.cuda.is_available() else "cpu"

    def add_results(self, results):
        self.results.extend(results)

    def __call__(self, results, ids, expected=None, result_dict=None):
        self.content_ids.extend(ids)
        return [t.to(self.device) for t in results]

    def start(self):
        self.results = []

    def finalize(self, result_dict, ds=None, output_dir=None):
        fid = FrechetInceptionDistance(feature=2048)
        fid.to(self.device)
        clip = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32")
        clip.to(self.device)
        dataset_size = ds.get_item_count()
        log.info("Accumulating results")
        for i in range(0, dataset_size):
            target = (
                torch.stack(ds.get_imgs([self.content_ids[i]]))
                .to(torch.uint8)
                .to(self.device)
            )
            captions = ds.get_captions([self.content_ids[i]])
            generated = torch.stack([self.results[i]]).to(torch.uint8).to(self.device)
            fid.update(target, real=True)
            fid.update(generated, real=False)
            clip.update(generated, captions)
        result_dict["FID_SCORE"] = float(fid.compute().item())
        result_dict["CLIP_SCORE"] = float(clip.compute().item())

        return result_dict
