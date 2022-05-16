import os
import json
import argparse
import fiftyone as fo
import fiftyone.zoo as foz


def get_args():
    parser = argparse.ArgumentParser(
        description="Download OpenImages using FiftyOne", add_help=True
    )
    parser.add_argument(
        "--dataset-dir", default="/open-images-v6", help="dataset download location"
    )
    parser.add_argument(
        "--classes",
        default=None,
        nargs="+",
        type=str,
        help="Classes to download. default to all classes",
    )
    parser.add_argument(
        "--output-labels",
        default="labels.json",
        type=str,
        help="Classes to download. default to all classes",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    print("Downloading open-images dataset ...")
    dataset = foz.load_zoo_dataset(
        name="open-images-v6",
        classes=args.classes,
        split="validation",
        label_types="detections",
        dataset_name="open-images",
        dataset_dir=args.dataset_dir,
    )

    print("Converting dataset to coco format ...")
    output_fname = os.path.join(args.dataset_dir, "annotations", args.output_labels)
    dataset.export(
        labels_path=output_fname,
        dataset_type=fo.types.COCODetectionDataset,
        label_field="detections",
        classes=args.classes,
    )
    # Add iscrowd label to openimages annotations
    with open(output_fname) as fp:
        labels = json.load(fp)
    for annotation in labels["annotations"]:
        annotation["iscrowd"] = int(annotation["IsGroupOf"])
    with open(output_fname, "w") as fp:
        json.dump(labels, fp)
