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
    target_folder = Path(target_folder)
    if (target_folder / file_name).exists():
        log.warning(f"Image {file_name} found locally, skipping download")
    else:
        urllib.request.urlretrieve(img_url, str(target_folder / file_name))

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
    tsv_path = Path(args.tsv_path) if args.tsv_path is not None else None

    # Check if the annotation dataframe is there
    calibration_captions_path = dataset_dir / "calibration" / "captions.tsv"
    if calibration_captions_path.exists():
        df_annotations = pd.read_csv(
            str(calibration_captions_path), sep="\t"
        )
    elif tsv_path is not None and tsv_path.exists():
        os.makedirs(f"{dataset_dir}/calibration/", exist_ok=True)
        shutil.copy(tsv_path, str(calibration_captions_path))
        df_annotations = pd.read_csv(
            str(calibration_captions_path), sep="\t"
        )
    else:
        

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

        # Move captions to target folder
        os.makedirs(str(dataset_dir / "captions"), exist_ok=True)
        shutil.move(
            str(dataset_dir / "raw" / "annotations" / "captions_train2014.json"),
            str(dataset_dir / "captions" / "captions_train2014.json"))
        if not args.keep_raw:
            shutil.rmtree(str(dataset_dir / "raw"))
        shutil.rmtree(str(dataset_dir / "download_aux"))

        # Convert to dataframe format and extract the relevant fields
        with open(dataset_dir / "captions" / "captions_train2014.json") as f:
            captions = json.load(f)
            annotations = captions["annotations"]
            images = captions["images"]
        df_annotations = pd.DataFrame(annotations)
        df_images = pd.DataFrame(images)

        # Calibration images
        with open(calibration_dir / "coco_cal_captions_list.txt") as f:
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
    os.makedirs(str(dataset_dir / "calibration"), exist_ok=True)
    if args.download_images:
        os.makedirs(str(dataset_dir / "calibration" / "data"), exist_ok=True)
        tasks = [
            (row["coco_url"],
             str(dataset_dir / "calibration" / "data" / ""),
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
    ].to_csv(str(dataset_dir / "calibration" / "captions.tsv"), sep="\t", index=False)
