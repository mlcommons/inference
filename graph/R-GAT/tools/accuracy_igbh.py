import argparse
import numpy as np
import torch
import json
import os


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mlperf-accuracy-file", required=True, help="path to mlperf_log_accuracy.json"
    )
    parser.add_argument(
        "--dataset-path",
        default="igbh",
        help="Path to IHGB dataset",
    )
    parser.add_argument(
        "--dataset-size",
        default="full",
        choices=["tiny", "small", "medium", "large", "full"]
    )
    parser.add_argument(
        "--no-memmap",
        action="store_true",
        help="do not use memmap even for large/full size variants")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose messages")
    parser.add_argument(
        "--output-file", default="results.json", help="path to output file"
    )
    parser.add_argument(
        "--dtype",
        default="uint8",
        choices=["uint8", "float32", "int32", "int64"],
        help="data type of the label",
    )
    args = parser.parse_args()
    return args


def load_labels(base_path, dataset_size, use_label_2K=True, no_memmap=False):
    # load labels
    paper_nodes_num = {
        "tiny": 100000,
        "small": 1000000,
        "medium": 10000000,
        "large": 100000000,
        "full": 269346174,
    }
    label_file = (
        "node_label_19.npy" if not use_label_2K else "node_label_2K.npy"
    )
    paper_lbl_path = os.path.join(
        base_path,
        dataset_size,
        "processed",
        "paper",
        label_file)

    if dataset_size in ["large", "full"] and not no_memmap:
        mmap_mode = 'r'
        paper_node_labels = torch.from_numpy(
            np.memmap(
                paper_lbl_path, dtype="float32", mode=mmap_mode, shape=(paper_nodes_num[dataset_size])
            )
        ).to(torch.long)
    else:
        mmap_mode = None
        paper_node_labels = torch.from_numpy(
            np.load(
                paper_lbl_path,
                mmap_mode=mmap_mode)).to(
            torch.long)
    labels = paper_node_labels
    val_idx = torch.load(
        os.path.join(
            base_path,
            dataset_size,
            "processed",
            "val_idx.pt"))
    return labels, val_idx


def get_labels(labels, val_idx, id_list):
    return labels[val_idx[id_list]]


if __name__ == "__main__":
    args = get_args()
    dtype_map = {
        "uint8": np.uint8,
        "float32": np.float32,
        "int32": np.int32,
        "int64": np.int64}

    with open(args.mlperf_accuracy_file, "r") as f:
        mlperf_results = json.load(f)

    labels, val_idx = load_labels(
        args.dataset_path, args.dataset_size, no_memmap=args.no_memmap)
    results = {}

    seen = set()
    good = 0
    total = 0
    for result in mlperf_results:
        idx = result["qsl_idx"]
        if idx in seen:
            continue
        seen.add(idx)

        # get ground truth
        label = get_labels(labels, val_idx, idx)
        # get prediction
        data = int(np.frombuffer(bytes.fromhex(
            result["data"]), dtype_map[args.dtype])[0])
        if label == data:
            good += 1
        total += 1
    results["accuracy"] = good / total
    results["model"] = "rgat"
    results["number_correct_samples"] = good
    results["performance_sample_count"] = total

    with open(args.output_file, "w") as fp:
        fp.write("accuracy={:.3f}%, good={}, total={}".format(
            100.0 *
            results["accuracy"], results["number_correct_samples"], results["performance_sample_count"]
        ))
