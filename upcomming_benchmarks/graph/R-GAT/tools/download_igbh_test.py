import argparse
import os
from igb import download


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-path",
        type=str,
        default="igbh/",
        help="Download path for the dataset",
    )
    parser.add_argument(
        "--dataset-size",
        type=str,
        default="tiny",
        choices=["tiny", "small", "medium"],
        help="Size of the dataset, only full for official submissions",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="heterogeneous",
        choices=["homogeneous", "heterogeneous"],
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.target_path, exist_ok=True)
    download.download_dataset(
        path=args.target_path,
        dataset_size=args.dataset_size,
        dataset_type=args.dataset_type,
    )
