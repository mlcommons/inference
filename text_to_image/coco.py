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
from tools.clip.clip_encoder import CLIPEncoder
from tools.fid.fid_score import compute_fid


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("coco")


class Coco(dataset.Dataset):
    def __init__(
        self,
        data_path,
        name=None,
        image_size=None,
        pre_process=None,
        pipe_tokenizer=None,
        pipe_tokenizer_2=None,
        latent_dtype=torch.float32,
        latent_device="cuda",
        latent_framework="torch",
        **kwargs,
    ):
        super().__init__()
        self.captions_df = pd.read_csv(f"{data_path}/captions/captions.tsv", sep="\t")
        self.image_size = image_size
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

    def preprocess(self, prompt, tokenizer):
        converted_prompt = self.convert_prompt(prompt, tokenizer)
        return tokenizer(
            converted_prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

    def image_to_tensor(self, img):
        img = np.asarray(img)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        tensor = torch.Tensor(img.transpose([2, 0, 1])).to(torch.uint8)
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        return tensor

    def preprocess_images(self, file_name):
        img = Image.open(self.img_dir + "/" + file_name)
        tensor = self.image_to_tensor(img)
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

    def get_img(self, id):
        img = Image.open(self.img_dir + "/" + self.captions_df.loc[id]["file_name"])
        return self.image_to_tensor(img)

    def get_imgs(self, id_list):
        image_list = []
        for id in id_list:
            image_list.append(self.get_img(id))
        return image_list

    def get_caption(self, i):
        return self.get_item(i)["caption"]

    def get_captions(self, id_list):
        return [self.get_caption(id) for id in id_list]

    def get_item_loc(self, id):
        return self.img_dir + "/" + self.captions_df.loc[id]["file_name"]


class PostProcessCoco:
    def __init__(
        self, device="cpu", dtype="uint8", statistics_path=os.path.join(os.path.dirname(__file__), "tools", "val2014.npz")
    ):
        self.results = []
        self.good = 0
        self.total = 0
        self.content_ids = []
        self.clip_scores = []
        self.fid_scores = []
        self.device = device if torch.cuda.is_available() else "cpu"
        if dtype == "uint8":
            self.dtype = torch.uint8
            self.numpy_dtype = np.uint8
        else:
            raise ValueError(f"dtype must be one of: uint8")
        self.statistics_path = statistics_path

    def add_results(self, results):
        self.results.extend(results)

    def __call__(self, results, ids, expected=None, result_dict=None):
        self.content_ids.extend(ids)
        return [
            (t.cpu().permute(1, 2, 0).float().numpy() * 255).round().astype(self.numpy_dtype)
            for t in results
        ]
    
    def save_images(self, ids, ds):
        info = []
        idx = {}
        for i, id in enumerate(self.content_ids):
            if id in ids:
                idx[id] = i
        if not os.path.exists("images/"):
            os.makedirs("images/", exist_ok=True)
        for id in ids:
            caption = ds.get_caption(id)
            generated = Image.fromarray(self.results[idx[id]])
            image_path_tmp = f"images/{self.content_ids[idx[id]]}.png"
            generated.save(image_path_tmp)
            info.append((self.content_ids[idx[id]], caption))
        with open("images/captions.txt", "w+") as f:
            for id, caption in info:
                f.write(f"{id}  {caption}\n")

    def start(self):
        self.results = []

    def finalize(self, result_dict, ds=None, output_dir=None):
        clip = CLIPEncoder(device=self.device)
        dataset_size = len(self.results)
        log.info("Accumulating results")
        for i in range(0, dataset_size):
            caption = ds.get_caption(self.content_ids[i])
            generated = Image.fromarray(self.results[i])
            self.clip_scores.append(
                100 * clip.get_clip_score(caption, generated).item()
            )

        fid_score = compute_fid(self.results, self.statistics_path, self.device)
        result_dict["FID_SCORE"] = fid_score
        result_dict["CLIP_SCORE"] = np.mean(self.clip_scores)

        return result_dict
