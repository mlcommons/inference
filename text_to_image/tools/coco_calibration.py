import argparse
import json
import logging
from multiprocessing import Pool
import numpy as np
import pandas as pd
import os
import tqdm
import urllib.request
import hashlib
import zipfile

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("coco")

# SHA-256 of the pinned COCO 2014 annotations archive
COCO_ANNOTATIONS_TRAINVAL2014_SHA256 = (
    "031296bbc80c45a1d1f76bf9a90ead27e94e99ec629208449507a4917a3bf009"
)


def _verify_sha256(path: str, expected: str) -> None:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    actual = hasher.hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: expected {expected}, got {actual}"
        )


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir", default="./coco-2014", help="Dataset download location"
    )
    parser.add_argument(
        "--tsv-path", default=None, help="Precomputed tsv file location"
    )
    parser.add_argument(
        "--num-workers",
        default=1,
        type=int,
        help="Number of processes to download images",
    )
    parser.add_argument(
        "--calibration-dir", default=None, help="Calibration ids location"
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep the raw dataset")
    parser.add_argument(
        "--download-images", action="store_true", help="Download the calibration set"
    )
    args = parser.parse_args()
    return args


def download_img(args):
    img_url, target_folder, file_name = args
    # Upgrade http://images.cocodataset.org/ URLs to the S3 origin
    img_url = img_url.replace(
        "http://images.cocodataset.org/",
        "https://s3.amazonaws.com/images.cocodataset.org/",
        1,
    )
    if os.path.exists(target_folder + file_name):
        log.warning(f"Image {file_name} found locally, skipping download")
    else:
        urllib.request.urlretrieve(img_url, target_folder + file_name)


if __name__ == "__main__":
    args = get_args()
    dataset_dir = os.path.abspath(args.dataset_dir)
    calibration_dir = (
        args.calibration_dir
        if args.calibration_dir is not None
        else os.path.join(
            os.path.dirname(__file__), "..", "..", "calibration", "COCO-2014"
        )
    )
    # Check if the annotation dataframe is there
    if os.path.exists(f"{dataset_dir}/calibration/captions.tsv"):
        df_annotations = pd.read_csv(
            f"{dataset_dir}/calibration/captions.tsv", sep="\t"
        )
    elif args.tsv_path is not None and os.path.exists(f"{args.tsv_path}"):
        os.makedirs(f"{dataset_dir}/calibration/", exist_ok=True)
        os.system(f"cp {args.tsv_path} {dataset_dir}/calibration/")
        df_annotations = pd.read_csv(
            f"{dataset_dir}/calibration/captions.tsv", sep="\t"
        )
    else:
        # Check if raw annotations file already exist
        if not os.path.exists(
                f"{dataset_dir}/raw/annotations/captions_train2014.json"):
            # Download annotations
            os.makedirs(f"{dataset_dir}/raw/", exist_ok=True)
            os.makedirs(f"{dataset_dir}/download_aux/", exist_ok=True)
            os.system(
                f"cd {dataset_dir}/download_aux/ && \
                    wget https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip --show-progress"
            )
            # Verify archive integrity before unzip
            _verify_sha256(
                f"{dataset_dir}/download_aux/annotations_trainval2014.zip",
                COCO_ANNOTATIONS_TRAINVAL2014_SHA256,
            )

            # Unzip file
            with zipfile.ZipFile(
                f"{dataset_dir}/download_aux/annotations_trainval2014.zip", "r"
            ) as zip_ref:
                zip_ref.extractall(f"{dataset_dir}/raw/")

        # Move captions to target folder
        os.makedirs(f"{dataset_dir}/captions/", exist_ok=True)
        os.system(
            f"mv {dataset_dir}/raw/annotations/captions_train2014.json {dataset_dir}/captions/"
        )
        if not args.keep_raw:
            os.system(f"rm -rf {dataset_dir}/raw")
        os.system(f"rm -rf {dataset_dir}/download_aux")
        # Convert to dataframe format and extract the relevant fields
        with open(f"{dataset_dir}/captions/captions_train2014.json") as f:
            captions = json.load(f)
            annotations = captions["annotations"]
            images = captions["images"]
        df_annotations = pd.DataFrame(annotations)
        df_images = pd.DataFrame(images)

        # Calibration images
        with open(f"{calibration_dir}/coco_cal_captions_list.txt") as f:
            calibration_ids = f.readlines()
            calibration_ids = [int(id.replace("\n", ""))
                               for id in calibration_ids]
            calibration_ids = calibration_ids

        df_annotations = df_annotations[np.isin(
            df_annotations["id"], calibration_ids)]
        df_annotations = df_annotations.sort_values(by=["id"])
        df_annotations["caption"] = df_annotations["caption"].apply(
            lambda x: x.replace("\n", "").strip()
        )
        df_annotations = (
            df_annotations.merge(
                df_images, how="inner", left_on="image_id", right_on="id"
            )
            .drop(["id_y"], axis=1)
            .rename(columns={"id_x": "id"})
            .sort_values(by=["id"])
            .reset_index(drop=True)
        )
    # Download images
    os.makedirs(f"{dataset_dir}/calibration/", exist_ok=True)
    if args.download_images:
        os.makedirs(f"{dataset_dir}/calibration/data/", exist_ok=True)
        tasks = [
            (row["coco_url"],
             f"{dataset_dir}/calibration/data/",
             row["file_name"])
            for i, row in df_annotations.iterrows()
        ]
        pool = Pool(processes=args.num_workers)
        [
            _
            for _ in tqdm.tqdm(
                pool.imap_unordered(download_img, tasks), total=len(tasks)
            )
        ]
    # Finalize annotations
    df_annotations[
        ["id", "image_id", "caption", "height", "width", "file_name", "coco_url"]
    ].to_csv(f"{dataset_dir}/calibration/captions.tsv", sep="\t", index=False)
