"""
Tool to calculate accuracy for loadgen accuracy output found in mlperf_log_accuracy.json
We assume that loadgen's query index is in the same order as
the images in coco's captions/captions.tsv.
"""


import argparse
import json
import os

from PIL import Image
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image.fid import FrechetInceptionDistance



def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlperf-accuracy-file", required=True, help="path to mlperf_log_accuracy.json")
    parser.add_argument("--dataset-dir", required=True, help="openimages directory")
    parser.add_argument("--verbose", action="store_true", help="verbose messages")
    parser.add_argument("--output-file", default="coco-results.json", help="path to output file")
    parser.add_argument("--use-preprocessed", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["gpu", "cpu"])
    args = parser.parse_args()
    return args


def preprocess_image(img_dir, file_name):
    img = Image.open(img_dir + "/" + file_name)
    img = np.asarray(img)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    tensor = torch.Tensor(np.asarray(img).transpose([2,0,1])).to(torch.uint8)
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3,1,1)
    return tensor.unsqueeze(0)

def main():
    args = get_args()
    annotations_file = os.environ.get('DATASET_ANNOTATIONS_FILE_PATH')
    if not annotations_file:
        annotations_file = os.path.join(args.dataset_dir, "captions", "captions.tsv")
    
    result_dict = {}

    # Load dataset annotations
    df_captions = pd.read_csv(annotations_file, sep="\t")

    # Load model outputs
    with open(args.mlperf_accuracy_file, "r") as f:
        results = json.load(f)

    # set device
    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "gpu":
        device = "cuda"

    # Load torchmetrics modules
    fid = FrechetInceptionDistance(feature=2048)
    fid.to(device)
    clip = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32")
    clip.to(device)

    image_ids = set()
    seen = set()
    no_results = 0
    for j in results:
        idx = j['qsl_idx']
        if idx in seen:
            continue
        seen.add(idx)

        # Load generated image
        generated_img = np.frombuffer(bytes.fromhex(j['data']), np.float32).reshape(1, 3, 1024, 1024)
        generated_img = torch.Tensor(generated_img).to(torch.uint8).to(device)
        # Load Ground Truth
        caption = df_captions.iloc[idx]["caption"]
        if args.use_preprocessed:
            file_name = df_captions.iloc[idx]["file_name"].replace(".jpg", ".pt")
            target_img = torch.load(args.dataset_dir + "/preprocessed/" + file_name).to(torch.uint8).to(device)
        else:
            target_img = preprocess_image(args.dataset_dir + "/validation/data", df_captions.iloc[idx]["file_name"]).to(torch.uint8).to(device)
        fid.update(target_img, real=True)
        fid.update(generated_img, real=False)
        clip.update(generated_img, caption)
        
    result_dict["FID_SCORE"] = float(fid.compute().item())
    result_dict["CLIP_SCORE"] = float(clip.compute().item())

    with open(args.output_file, "w") as fp:
        json.dump(result_dict, fp, sort_keys=True, indent=4)

if __name__ == "__main__":
    main()