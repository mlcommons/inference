import argparse
import json
import logging
from multiprocessing import Pool
import numpy as np
import pandas as pd
import os
import tqdm
import urllib.request
import zipfile

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("coco")


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir", default="./coco-2014", help="Dataset download location"
    )
    parser.add_argument(
        "--max-images",
        default=500,
        type=int,
        help="Size of the calibration dataset",
    )
    parser.add_argument(
        "--calibration-dir", default=None, help="Calibration ids location"
    )
    parser.add_argument(
        "--seed", type=int, default=2023, help="Seed to choose the dataset"
    )
    parser.add_argument(
        "--keep-raw", action="store_true", help="Keep raw folder"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    dataset_dir = os.path.abspath(args.dataset_dir)
    calibration_dir = args.calibration_dir if args.calibration_dir is not None else os.path.join(os.path.dirname(__file__), "..", "..", "calibration", "COCO-2014")
    
    # Check if raw annotations file already exist
    if not os.path.exists(f"{dataset_dir}/raw/annotations/captions_train2014.json"):
        # Download annotations
        os.makedirs(f"{dataset_dir}/raw/", exist_ok=True)
        os.makedirs(f"{dataset_dir}/download_aux/", exist_ok=True)
        os.system(
            f"cd {dataset_dir}/download_aux/ && \
                wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip --show-progress"
        )

        # Unzip file
        with zipfile.ZipFile(
            f"{dataset_dir}/download_aux/annotations_trainval2014.zip", "r"
        ) as zip_ref:
            zip_ref.extractall(f"{dataset_dir}/raw/")

    # Convert to dataframe format and extract the relevant fields
    with open(f"{dataset_dir}/raw/annotations/captions_train2014.json") as f:
        captions = json.load(f)
        annotations = captions["annotations"]
        images = captions["images"]
    df_annotations = pd.DataFrame(annotations)
    df_images = pd.DataFrame(images)

    # Calibration images 
    df_annotations = df_annotations.drop_duplicates(
        subset=["image_id"], keep="first"
    )
    # Sort, shuffle and choose the final dataset
    df_annotations = df_annotations.sort_values(by=["id"])
    df_annotations = df_annotations.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    df_annotations = df_annotations.iloc[: args.max_images]
    df_annotations['caption'] = df_annotations['caption'].apply(lambda x: x.replace('\n', '').strip())
    df_annotations = (
        df_annotations.merge(
            df_images, how="inner", left_on="image_id", right_on="id"
        )
        .drop(["id_y"], axis=1)
        .rename(columns={"id_x": "id"})
        .sort_values(by=["id"])
        .reset_index(drop=True)
    )
    # Save ids
    with open(f"{calibration_dir}/coco_cal_images_list.txt", "w+") as f:
        s = "\n".join([str(_) for _ in df_annotations["id"].values])
        f.write(s)
    # Remove Folder
    os.system(f"rm -rf {dataset_dir}")

