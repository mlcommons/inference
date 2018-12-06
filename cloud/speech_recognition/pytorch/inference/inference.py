import argparse
import os
import os.path as osp
import csv
import json
import sys
sys.path.append('../')

from collections import OrderedDict
import numpy as np
import torch

from dataset.data_loader import AudioDataLoader, SpectrogramDataset
from model.decoder import GreedyDecoder
from model.model import DeepSpeech

import model.params as params
from model.eval_model import eval_model_verbose
from model.utils import get_model, get_labels, get_audio_conf


def main(args):
    params.cuda = not bool(args.cpu)
    print("Use cuda: {}".format(params.cuda))

    torch.manual_seed(args.seed)
    if params.cuda:
        torch.cuda.manual_seed_all(args.seed)

    labels = get_labels(params)
    audio_conf = get_audio_conf(params)

    if args.use_set == 'libri':
        testing_manifest = params.test_manifest + ("_held{}".format(args.hold_idx) if args.hold_idx >= 0 else "")
    else:
        assert False, "Only the librispeech dataset is currently supported."

    if args.batch_size_val > 0:
        params.batch_size_val = args.batch_size_val

    print("Testing on: {}".format(testing_manifest))
    test_dataset = SpectrogramDataset(audio_conf=audio_conf,
                                      manifest_filepath=testing_manifest,
                                      labels=labels,
                                      normalize=True,
                                      augment=False)
    test_loader = AudioDataLoader(test_dataset,
                                  batch_size=params.batch_size_val,
                                  num_workers=1)

    model = get_model(params)

    print("=======================================================")
    for arg in vars(args):
        print("*** {} = {} ".format(arg.ljust(25), getattr(args, arg)))
    print("=======================================================")


    decoder = GreedyDecoder(labels)

    if args.continue_from:
        print("Loading checkpoint model {}".format(args.continue_from))

        if params.cuda:
            package = torch.load(args.continue_from)
        else:
            package = torch.load(args.continue_from, map_location=lambda storage, loc: storage)

        model.load_state_dict(package['state_dict'])

    if params.cuda:
        model = torch.nn.DataParallel(model).cuda()

    print(model)
    print("Number of parameters: {}".format(DeepSpeech.get_param_size(model)))

    model.eval()
    wer, cer, trials = eval_model_verbose(model, test_loader, decoder, params.cuda, args.n_trials)
    root = os.getcwd()

    prefix = "inference_bs{}_idx{}_{}_gpu".format(params.batch_size_val,
                                                args.hold_idx,
                                                'use' if params.cuda else 'no')
    csvfile = osp.join(root, prefix + '.csv')
    jsonfile = osp.join(root, prefix + '_metric.json')

    print("Exporting inference to: {}".format(csvfile))
    N = len(trials.array)
    percentile_50 = np.percentile(trials.array, 50) / params.batch_size_val / args.hold_sec
    percentile_99 = np.percentile(trials.array, 99) / params.batch_size_val / args.hold_sec

    metric_dict = OrderedDict([
                    ('batch_times_pre_normalized_by_hold_sec', args.hold_sec),
                    ('wer', wer),
                    ('cer', cer),
                    ('avg_batch_time', trials.avg/args.hold_sec),
                    ('50%-tile_latency', percentile_50),
                    ('99%-tile_latency', percentile_99),
                    ('throughput(samples/sec)', 1/percentile_50),
                ])

    write_dict = OrderedDict([
                    ('batch_time', np.array([x / args.hold_sec for x in trials.array]))
                ])

    with open(jsonfile, 'w+') as f:
        json.dump(metric_dict, f)

    with open(csvfile, 'w+') as f:
        csvwriter = csv.DictWriter(f, fieldnames=write_dict.keys())
        csvwriter.writeheader()
        for i in range(N):
            csvwriter.writerow({k:v[i] for k,v in write_dict.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepSpeech training')

    parser.add_argument('--checkpoint', dest='checkpoint', action='store_true',
                        help='Enables checkpoint saving of model')
    parser.add_argument('--model_path', default='./deepspeech_final.pth',
                        help='Location to save best validation model')
    parser.add_argument('--continue_from', required=True,
                        help='Continue from checkpoint model')
    parser.add_argument('--seed', default=0xdeadbeef,
                        type=int, help='Random Seed')
    parser.add_argument('--use_set', default="libri",
                        choices=['libri', 'ov'], help='ov = OpenVoice test set, libri = Librispeech val set')
    parser.add_argument('--cpu',
                        action='store_true', help='Use cpu to do inference or not')
    parser.add_argument('--hold_idx', default=-1,
                        type=int, help='Input idx to hold the test dataset at')
    parser.add_argument('--hold_sec', default=1,
                        type=float, help='Speech clip time length')
    parser.add_argument('--batch_size_val', default=-1,
                        type=int, help='Batch size used for validaton')
    parser.add_argument('--n_trials', default=-1,
                        type=int, help='Limit the number of trial ran, useful when holding idx')
    args = parser.parse_args()

    main(args)
