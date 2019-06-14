import sys
sys.path.append('../')
import time
import csv
from collections import OrderedDict

import torch

from params import cuda
from utils import AverageMeter

csv_header = ['batch_num',
              'batch_latency',
              'batch_duration_s',
              'batch_seq_len',
              'batch_size_kb',
              'item_num',
              'item_latency',
              'item_duration_s',
              'item_seq_len',
              'item_size_kb',
              'word_count',
              'char_count',
              'word_err_count',
              'char_err_count',
              'pred',
              'target']


def eval_model(model, test_loader, decoder):
    """
    Model evaluation -- used during training.
    """
    total_cer, total_wer = 0, 0
    word_count, char_count = 0, 0
    model.eval()
    # For each batch in the test_loader, make a prediction and calculate the WER CER
    for data in test_loader:
        inputs, targets, input_percentages, target_sizes = data
        inputs = torch.autograd.Variable(inputs)

        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size

        if cuda:
            inputs = inputs.cuda()

        out = model(inputs)
        out = out.transpose(0, 1)  # TxNxH
        seq_length = out.size(0)
        sizes = input_percentages.mul_(int(seq_length)).int()

        # Decode the ouput to actual strings and compare to label
        # Get the LEV score and the word, char count
        decoded_output = decoder.decode(out.data, sizes)
        target_strings = decoder.process_strings(decoder.convert_to_strings(split_targets))
        for x in range(len(target_strings)):
            total_wer += decoder.wer(decoded_output[x], target_strings[x])
            total_cer += decoder.cer(decoded_output[x], target_strings[x])
            word_count += len(target_strings[x].split())
            char_count += len(target_strings[x])

        if cuda:
            torch.cuda.synchronize()
        del out

    # WER, CER
    wer = total_wer / float(word_count)
    cer = total_cer / float(char_count)
    wer *= 100
    cer *= 100

    return wer, cer


def eval_model_verbose(model,
                       test_loader,
                       decoder,
                       cuda,
                       out_path,
                       item_info_array,
                       warmups=0,
                       meta=False):
    """
    Model evaluation -- used during inference.

    returns wer, cer, batch time array and warm up time
    """
    # Warming up
    end = time.time()
    total_trials = len(test_loader)
    for i, data in enumerate(test_loader):
        if i >= warmups:
            break
        sys.stdout.write("\rWarmups ({}/{}) ".format(i+1, warmups))
        sys.stdout.flush()
        if meta:
            inputs, targets, input_percentages, target_sizes, batch_meta, item_meta = data
        else:
            inputs, targets, input_percentages, target_sizes = data
        inputs = torch.autograd.Variable(inputs, volatile=False)

        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size

        if cuda:
            inputs = inputs.cuda()

        out = model(inputs)
    warmup_time = time.time() - end
    if warmups > 0: print("Warmed up in {}s").format(warmup_time)

    # the actual inference trial
    total_cer, total_wer = 0, 0
    word_count, char_count = 0, 0
    model.eval()
    batch_time = AverageMeter()

    # For each batch in the test_loader, make a prediction and calculate the WER CER
    item_num = 1
    with open(out_path, 'wb') as f:
        csvwriter = csv.DictWriter(f, fieldnames=csv_header)
        csvwriter.writeheader()
        for i, data in enumerate(test_loader):
            batch_num = i + 1
            if meta:
                inputs, targets, input_percentages, target_sizes, batch_meta, item_meta = data
            else:
                inputs, targets, input_percentages, target_sizes = data

            inputs = torch.autograd.Variable(inputs, volatile=False)

            # unflatten targets
            split_targets = []
            offset = 0
            for size in target_sizes:
                split_targets.append(targets[offset:offset + size])
                offset += size

            if cuda:
                inputs = inputs.cuda()
            end = time.time()  # Timing start (Inference only)
            out = model(inputs)
            batch_time.update(time.time() - end)  # Timing end (Inference only)
            out = out.transpose(0, 1)  # TxNxH
            seq_length = out.size(0)
            sizes = input_percentages.mul_(int(seq_length)).int()

            # Decode the ouput to actual strings and compare to label
            # Get the LEV score and the word, char count
            decoded_output = decoder.decode(out.data, sizes)
            target_strings = decoder.process_strings(decoder.convert_to_strings(split_targets))
            batch_we = batch_wc = batch_ce = batch_cc = 0
            for x in range(len(target_strings)):
                this_we = decoder.wer(decoded_output[x], target_strings[x])
                this_ce = decoder.cer(decoded_output[x], target_strings[x])
                this_wc = len(target_strings[x].split())
                this_cc = len(target_strings[x])
                this_pred = decoded_output[x]
                this_true = target_strings[x]
                if item_num <= len(item_info_array):
                    item_latency = item_info_array[item_num - 1]['batch_latency']
                else:
                    item_latency = "-9999"

                out_data = [batch_num,
                            batch_time.array[-1],
                            batch_meta[2],
                            batch_meta[4],
                            batch_meta[3],
                            item_num,
                            item_latency,
                            item_meta[x][2],
                            item_meta[x][4],
                            item_meta[x][3],
                            this_wc, this_cc,
                            this_we, this_ce,
                            this_pred, this_true]

                csv_dict = {k:v for k, v in zip(csv_header, out_data)}
                csvwriter.writerow(csv_dict)

                item_num += 1
                batch_we += this_we
                batch_ce += this_ce
                batch_wc += this_wc
                batch_cc += this_cc

            total_wer += batch_we
            total_cer += batch_ce
            word_count += batch_wc
            char_count += batch_cc

            print('[{0}/{1}]\t'
                  'Batch: latency (running average) {batch_time.val:.4f} ({batch_time.avg:.3f})\t\t'
                  'WER {2:.1f} \t CER {3:.1f}'
                  .format((i + 1), total_trials,
                          batch_we / float(batch_wc),
                          batch_ce / float(batch_cc),
                          batch_time=batch_time))
            if cuda:
                torch.cuda.synchronize()
            del out

    # WER, CER
    wer = total_wer / float(word_count)
    cer = total_cer / float(char_count)
    wer *= 100
    cer *= 100

    return wer, cer, batch_time, warmup_time
