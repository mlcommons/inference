import argparse
import numpy as np
import pandas as pd


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=248,
        help="Dataset download location",
    )
    parser.add_argument(
        "--output-path", default="random_ids.txt", help="Dataset download location"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Dataset download location")
    parser.add_argument(
        "--seed", "-s", type=int, default=3234044599, help="Dataset download location"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    np.random.seed(args.seed)
    sample_ids = list(np.random.choice(range(248), args.n))
    with open(args.output_path, "w+") as f:
        for i, sample in enumerate(sample_ids):
            if i != (len(sample_ids) - 1):
                f.write(str(sample) + "\n")
            else:
                f.write(str(sample))
