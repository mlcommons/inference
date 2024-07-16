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
from clip.clip_encoder import CLIPEncoder
from fid.inception import InceptionV3
from fid.fid_score import compute_fid, compute_statistics_of_path, get_activations, calculate_frechet_distance
from tqdm import tqdm
import ijson



def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlperf-accuracy-file", required=True, help="path to mlperf_log_accuracy.json")
    parser.add_argument("--caption-path", default="coco2014/captions/captions_source.tsv", help="path to coco captions")
    parser.add_argument("--statistics-path", default=None, help="path to statistics")
    parser.add_argument("--verbose", action="store_true", help="verbose messages")
    parser.add_argument("--output-file", default="coco-results.json", help="path to output file")
    parser.add_argument("--compliance-images-path", required=False, help="path to dump 10 stable diffusion xl compliance images")
    parser.add_argument("--device", default="cpu", choices=["gpu", "cpu"])
    parser.add_argument("--low_memory", action="store_true", help="If device is has limited memory (<70G), use the memory saving path.")
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

    # Load dataset annotations
    df_captions = pd.read_csv(args.caption_path, sep="\t")

    # set device
    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "gpu":
        device = "cuda"

    # set statistics path
    statistics_path = args.statistics_path
    if args.statistics_path is None:
        statistics_path = os.path.join(os.path.dirname(__file__), "val2014.npz")

    # Set compliance images path
    dump_compliance_images = False
    compliance_images_idx_list = []
    if args.compliance_images_path:
        if not os.path.exists(args.compliance_images_path):
            os.makedirs(args.compliance_images_path)
        dump_compliance_images = True
        compliance_images_idx_list = []
        with open(os.path.join(os.path.dirname(__file__), "sample_ids.txt"), 'r') as compliance_id_file:
            for line in compliance_id_file:
                idx = int(line.strip())
                compliance_images_idx_list.append(idx)
        # Dump caption.txt
        with open(os.path.join(args.compliance_images_path, "captions.txt"), "w+") as caption_file:
            for idx in compliance_images_idx_list:
                caption_file.write(f"{idx}  {df_captions.iloc[idx]['caption']}\n")

    # Compute accuracy
    if args.low_memory:
        print(f"Device has low memory, running memory saving path!")
        compute_accuracy_low_memory(
            args.mlperf_accuracy_file,
            args.output_file,
            device,
            dump_compliance_images,
            compliance_images_idx_list,
            args.compliance_images_path,
            df_captions,
            statistics_path,
        )
    else:
        compute_accuracy(
            args.mlperf_accuracy_file,
            args.output_file,
            device,
            dump_compliance_images,
            compliance_images_idx_list,
            args.compliance_images_path,
            df_captions,
            statistics_path,
        )        


def compute_accuracy(
    mlperf_accuracy_file, 
    output_file,
    device,
    dump_compliance_images,
    compliance_images_idx_list,
    compliance_images_path,
    df_captions,
    statistics_path,    
):
    # Load torchmetrics modules
    clip = CLIPEncoder(device=device)
    clip_scores = []
    seen = set()
    result_list = []
    result_dict = {}

    # Load model outputs
    with open(mlperf_accuracy_file, "r") as f:
        results = json.load(f)

    for j in tqdm(results):
        idx = j['qsl_idx']
        if idx in seen:
            continue
        seen.add(idx)

        # Load generated image
        generated_img = np.frombuffer(bytes.fromhex(j['data']), np.uint8).reshape(1024, 1024, 3)
        result_list.append(generated_img)
        generated_img = Image.fromarray(generated_img)

        # Dump compliance images
        if dump_compliance_images and idx in compliance_images_idx_list:
            generated_img.save(os.path.join(compliance_images_path, f"{idx}.png"))

        # generated_img = torch.Tensor(generated_img).to(torch.uint8).to(device)
        # Load Ground Truth
        caption = df_captions.iloc[idx]["caption"]
        clip_scores.append(
            100 * clip.get_clip_score(caption, generated_img).item()
        )
    fid_score = compute_fid(result_list, statistics_path, device)

    result_dict["FID_SCORE"] = fid_score
    result_dict["CLIP_SCORE"] = np.mean(clip_scores)
    print(f"Accuracy Results: {result_dict}")

    with open(output_file, "w") as fp:
        json.dump(result_dict, fp, sort_keys=True, indent=4)


def compute_accuracy_low_memory(
    mlperf_accuracy_file, 
    output_file,
    device,
    dump_compliance_images,
    compliance_images_idx_list,
    compliance_images_path,
    df_captions,
    statistics_path,
    batch_size=256,
    inception_dims=2048,
    num_workers=1,
):    
    if num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            # os.sched_getaffinity is not available under Windows, use
            # os.cpu_count instead (which may not return the *available* number
            # of CPUs).
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = num_workers    

    # Load torchmetrics modules
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_dims]
    inception_model = InceptionV3([block_idx]).to(device)
    clip_model = CLIPEncoder(device=device)
    
    clip_scores = []
    seen = set()
    result_batch = []
    result_dict = {}
    activations = np.empty((0, inception_dims))
    
    # Load model outputs
    with open(mlperf_accuracy_file, "r") as f:
        results = ijson.items(f, "item")

        for j in tqdm(results):
            idx = j['qsl_idx']
            if idx in seen:
                continue
            seen.add(idx)

            # Load generated image
            generated_img = np.frombuffer(bytes.fromhex(j['data']), np.uint8).reshape(1024, 1024, 3)
            generated_img = Image.fromarray(generated_img)

            # Dump compliance images
            if dump_compliance_images and idx in compliance_images_idx_list:
                generated_img.save(os.path.join(compliance_images_path, f"{idx}.png"))

            # Load Ground Truth
            caption = df_captions.iloc[idx]["caption"]
            clip_scores.append(
                100 * clip_model.get_clip_score(caption, generated_img).item()
            )

            result_batch.append(generated_img.convert("RGB"))

            if len(result_batch) == batch_size:
                act = get_activations(result_batch, inception_model, batch_size, inception_dims, device, num_workers)
                activations = np.append(activations, act, axis=0)
                result_batch.clear()
        
        # Remaining data for last batch
        if len(result_batch) > 0:
            act = get_activations(result_batch, inception_model, len(result_batch), inception_dims, device, num_workers)
            activations = np.append(activations, act, axis=0)
            
    m1, s1 = compute_statistics_of_path(
        statistics_path,
        inception_model,
        batch_size,
        inception_dims,
        device,
        num_workers,
        None,
        None,
    )

    m2 = np.mean(activations, axis=0)
    s2 = np.cov(activations, rowvar=False)

    fid_score = calculate_frechet_distance(m1, s1, m2, s2)

    result_dict["FID_SCORE"] = fid_score
    result_dict["CLIP_SCORE"] = np.mean(clip_scores)
    print(f"Accuracy Results: {result_dict}")

    with open(output_file, "w") as fp:
        json.dump(result_dict, fp, sort_keys=True, indent=4)

if __name__ == "__main__":
    main()
