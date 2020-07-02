'''
quick generator of random samples for debugging
'''

import sys
import argparse
import numpy as np

def quickgen(num_samples, num_t, num_d, num_s, ln_emb, text_file=None):
    # generate place holder random array, including dense features
    a = np.random.randint(0, 10, (num_t + num_d + num_s, num_samples))
    # generate targets
    a[0, :] = np.random.randint(0, 2, num_samples)
    # generate sparse features
    for k, size in enumerate(ln_emb):
        a[num_t + num_d + k, :] = np.random.randint(0, size, num_samples)
    a = np.transpose(a)

    # generate print format
    lstr = []
    for _ in range(num_t + num_d):
        lstr.append("%d")
    for _ in range(num_s):
        lstr.append("%x")
    if text_file is not None:
        np.savetxt(text_file, a, fmt=lstr, delimiter='\t',)

    return a

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quick generator of random samples for debugging."
    )
    parser.add_argument("--num-samples",         type=int, default=4096)
    parser.add_argument("--num-dense-features",  type=int, default=13)
    parser.add_argument("--num-sparse-features", type=str, default="4-3-2")
    parser.add_argument("--num-targets",         type=int, default=1)
    parser.add_argument("--profile",             type=str, default="")  # kaggle|terabyte0875|terabyte
    parser.add_argument("--num-days",            type=int, default=24)
    parser.add_argument("--numpy-rand-seed",     type=int, default=123)
    parser.add_argument("--output-name",         type=str, default="day_")
    parser.add_argument("--output-dir",          type=str, default="./")
    args = parser.parse_args()

    np.random.seed(args.numpy_rand_seed)

    num_days = args.num_days
    out_name = args.output_name
    ln_emb = np.fromstring(args.num_sparse_features, dtype=int, sep="-")
    if args.profile == "kaggle":
        # 1. Criteo Kaggle Display Advertisement Challenge Dataset (see ./bench/dlrm_s_criteo_kaggle.sh)
        num_days = 1
        out_name = "train.txt"
        ln_emb = np.array([1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572])
    elif args.profile == "terabyte0875":
        # 2. Criteo Terabyte (see ./bench/dlrm_s_criteo_terabyte.sh [--sub-sample=0.875] --max-in-range=10000000)
        num_days = 24
        out_name = "day_"
        ln_emb = np.array([9980333,36084,17217,7378,20134,3,7112,1442,61, 9758201,1333352,313829,10,2208,11156,122,4,970,14, 9994222, 7267859, 9946608,415421,12420,101, 36])
    elif args.profile == "terabyte":
        # 3. Criteo Terabyte MLPerf training (see ./bench/run_and_time.sh --max-in-range=40000000)
        num_days = 24
        out_name = "day_"
        ln_emb=np.array([39884406,39043,17289,7420,20263,3,7120,1543,63,38532951,2953546,403346,10,2208,11938,155,4,976,14,39979771,25641295,39664984,585935,12972,108,36])

    num_d   = args.num_dense_features
    num_s   = len(ln_emb)
    num_t   = args.num_targets
    out_dir = args.output_dir
    for k in range(num_days):
        text_file =  out_dir + out_name + ("" if args.profile == "kaggle" else str(k))
        print(text_file)

        quickgen(args.num_samples, num_t, num_d, num_s, ln_emb, text_file)
