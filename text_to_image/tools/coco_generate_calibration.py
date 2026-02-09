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
import shutil
from pathlib import Path
import requests

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
        "--keep-raw",
        action="store_true",
        help="Keep raw folder")

    args = parser.parse_args()
    return args

def download_file(url: str, output_dir: Path, filename: str | None = None):
    os.makedirs(str(output_dir), exist_ok=True)

    if filename is None:
        filename = os.path.basename(url)

    output_path = output_dir / filename

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("Content-Length", 0))
        with open(str(output_path), "wb") as f, tqdm.tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=filename,
        ) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    return output_path

if __name__ == "__main__":
    args = get_args()
    dataset_dir = os.path.abspath(args.dataset_dir)
    dataset_dir = Path(dataset_dir)

    calibration_dir = (
        args.calibration_dir
        if args.calibration_dir is not None
        else os.path.join(
            os.path.dirname(__file__), "..", "..", "calibration", "COCO-2014"
        )
    )
    calibration_dir = Path(calibration_dir)

    # Check if raw annotations file already exist
    if not (dataset_dir / "raw" / "annotations" / "captions_train2014.json").exists():
            # Download annotations
            os.makedirs(str(dataset_dir / "raw"), exist_ok=True)
            os.makedirs(str(dataset_dir / "download_aux"), exist_ok=True)
            download_file(
                url="http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
                output_dir=dataset_dir / "download_aux",
            )
            # Unzip file
            zipfile_path = dataset_dir / "download_aux" / "annotations_trainval2014.zip"
            # Unzip file
            with zipfile.ZipFile(
                str(zipfile_path), "r"
            ) as zip_ref:
                zip_ref.extractall(str(dataset_dir / "raw/"))

    # Convert to dataframe format and extract the relevant fields
    with open(dataset_dir / "raw" / "annotations" / "captions_train2014.json") as f:
        captions = json.load(f)
        annotations = captions["annotations"]
        images = captions["images"]
    df_annotations = pd.DataFrame(annotations)
    df_images = pd.DataFrame(images)

    # Calibration images
    df_annotations = df_annotations.drop_duplicates(
        subset=["image_id"], keep="first")
    # Sort, shuffle and choose the final dataset
    df_annotations = df_annotations.sort_values(by=["id"])
    df_annotations = df_annotations.sample(frac=1, random_state=args.seed).reset_index(
        drop=True
    )
    df_annotations = df_annotations.iloc[: args.max_images]
    df_annotations["caption"] = df_annotations["caption"].apply(
        lambda x: x.replace("\n", "").strip()
    )
    df_annotations = (
        df_annotations.merge(
            df_images,
            how="inner",
            left_on="image_id",
            right_on="id")
        .drop(["id_y"], axis=1)
        .rename(columns={"id_x": "id"})
        .sort_values(by=["id"])
        .reset_index(drop=True)
    )
    # Save ids
    os.makedirs(str(calibration_dir), exist_ok=True)
    with open(calibration_dir / "coco_cal_images_list.txt", "w+") as f:
        s = "\n".join([str(_) for _ in df_annotations["id"].values])
        f.write(s)
    # Remove Folder
    shutil.rmtree(dataset_dir)