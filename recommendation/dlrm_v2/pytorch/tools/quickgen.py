'''
quick generator of random samples for debugging
'''

import sys
import argparse
import numpy as np

def quickgen(num_samples, num_t, num_d, multihot_sizes, text_file=None):
    # generate place holder random array, including dense features
    dense_features = np.random.uniform(0., 9., size = (num_samples, num_d)).astype(np.float32)
    # generate targets
    labels = np.random.randint(0, 2, (num_samples, num_t), dtype=np.int32)
    # generate sparse features
    sparse_features = {}
    limit = 2
    for k, size in enumerate(multihot_sizes):
        sparse_features[str(k)] = np.random.randint(0, limit, (num_samples, size), dtype=np.int32)
    # generate print format
    if text_file is not None:
        np.save(text_file + "_dense_debug.npy", dense_features)
        np.save(text_file + "_labels_debug.npy", labels)
        np.savez(text_file + "_sparse_multi_hot_debug.npz", **sparse_features)

    return dense_features, sparse_features, labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quick generator of random samples for debugging."
    )
    parser.add_argument("--num-samples",            type=int, default=4096)
    parser.add_argument("--num-dense-features",     type=int, default=13)
    parser.add_argument("--num-multihot-features",  type=str, default="4,3,2")
    parser.add_argument("--num-targets",            type=int, default=1)
    parser.add_argument("--day",                    type=int, default=23)
    parser.add_argument("--numpy-rand-seed",        type=int, default=123)
    parser.add_argument("--output-name",            type=str, default="day_")
    parser.add_argument("--output-dir",             type=str, default="./")
    args = parser.parse_args()

    np.random.seed(args.numpy_rand_seed)

    out_name = args.output_name
    multihot_sizes = np.fromstring(args.num_multihot_features, dtype=int, sep=",")

    num_d   = args.num_dense_features
    num_t   = args.num_targets
    out_dir = args.output_dir
    text_file =  out_dir + out_name + str(args.day)
    print(text_file)
    quickgen(args.num_samples, num_t, num_d, multihot_sizes, text_file)
