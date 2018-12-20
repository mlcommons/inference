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
    params.cuda = args.device == "gpu"
    print("Use cuda: {}".format(params.cuda))

    torch.manual_seed(args.seed)
    if params.cuda:
        torch.cuda.manual_seed_all(args.seed)

    labels = get_labels(params)
    audio_conf = get_audio_conf(params)

    if args.use_set == 'libri':
        testing_manifest = params.test_manifest
    else:
        assert False, "Only the librispeech dataset is currently supported."

    if args.batch_size_val > 0:
        params.batch_size_val = args.batch_size_val

    print("Testing on: {}".format(testing_manifest))
    test_dataset = SpectrogramDataset(audio_conf=audio_conf,
                                      manifest_filepath=testing_manifest,
                                      labels=labels,
                                      normalize=True,
                                      augment=False,
				      force_duration=args.force_duration,
                                      slice=args.slice)
    test_loader = AudioDataLoader(test_dataset,
                                  batch_size=params.batch_size_val,
                                  num_workers=1,
			          with_meta=True)

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
    
    root = os.getcwd()
    
    # First, we will extract the batch_1_info
    if args.batch_1_file is not None and args.batch_1_file != "none":
        with open(osp.join(root, args.batch_1_file)) as f:
            batch1_data = [x for x in csv.DictReader(f)]
    else:
        batch1_data = []

    # Prepare the export files and write to it as we infer
    print("=======================================================")
    for arg in vars(args):
        print("***%s = %s " % (arg.ljust(25), getattr(args, arg)))
    print("=======================================================")
        
    prefix = "inference_bs{batch_size}_{processor}".format(batch_size=params.batch_size_val,
                                                    processor='gpu' if params.cuda else 'cpu')
    csvfile = osp.join(root, prefix + '.csv')
    jsonfile = osp.join(root, prefix + '_metric.json')
    
    print("Exporting inference to: {}".format(csvfile))

    json_dict = {k:getattr(args, k) for k in vars(args)}

    model.eval()
    wer, cer, trials, warmup_time = eval_model_verbose(model,
                                                      test_loader,
                                                      decoder,
                                                      params.cuda,
                                                      csvfile,
                                                      batch1_data,
                                                      warmups=args.warmups,
                                                      meta=True)

    percentile_50 = np.percentile(trials.array, 50)
    percentile_99 = np.percentile(trials.array, 99)
    json_dict.update(
        {
            'best_wer': wer,
            'best_cer': cer,
            'warmup_time': warmup_time,
            'avg_batch_latency': trials.avg,
            '50%_batch_latency': percentile_50,
            '99%_batch_latency': percentile_99,
            'dataset_latency': sum(trials.array)
        }
    )

    with open(jsonfile, 'w+') as f:
        json.dump(json_dict, f)


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
    parser.add_argument('--device', default='cpu',
                        choices=['cpu', 'gpu'], help='Use cpu to do inference instead of the default gpu')
    parser.add_argument('--batch_size_val', default=-1,
                        type=int, help='Batch size used for inference, overrides the param settings')
    parser.add_argument('--warmups', default=0,
                        type=int, help='Number of samples to warmup on')
    parser.add_argument('--force_duration', default=-1,
                        type=float, help='Desired duration of inputs')
    parser.add_argument('--slice',
                        action='store_true', help='Enable slicing the inputs')
    parser.add_argument('--batch_1_file', default=None,
                        type=str, help='Path to batch 1 result csv file')
    args = parser.parse_args()

    main(args)
