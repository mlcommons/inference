
import argparse
import numpy as np
import pandas as pd

def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tsv-path", default="../coco2014/captions/captions_source.tsv", help="Dataset download location"
    )
    parser.add_argument(
        "--output-path", default="sample_ids.txt", help="Dataset download location"
    )
    parser.add_argument(
        "--n", type=int, default=10, help="Dataset download location"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=633994880, help="Dataset download location"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    np.random.seed(args.seed)
    df_annotations = pd.read_csv(f"{args.tsv_path}", sep="\t")
    sample_ids = list(np.random.choice(df_annotations.shape[0], args.n))
    with open(args.output_path, "w+") as f:
        for i, sample in enumerate(sample_ids):
            if i != (len(sample_ids)-1):
                f.write(str(sample) + "\n")
            else:
                f.write(str(sample))
    