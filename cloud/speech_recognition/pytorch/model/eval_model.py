import sys
sys.path.append('../')
import time
import numpy as np

import torch
from torch.autograd import Variable

from params import cuda
from utils import AverageMeter



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
        inputs = torch.Tensor(inputs)

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


def eval_model_verbose(model, test_loader, decoder, cuda, n_trials=-1):
    """
    Model evaluation -- used during inference.
    """
    total_cer, total_wer = 0, 0
    word_count, char_count = 0, 0
    model.eval()
    batch_time = AverageMeter()

    # We allow the user to specify how many batches (trials) to run
    trials_ran = min(n_trials if n_trials != -1 else len(test_loader), len(test_loader))

    for i, data in enumerate(test_loader):
        if i >= n_trials != -1:
            break
        else:
            end = time.time()
            inputs, targets, input_percentages, target_sizes = data
            inputs = Variable(inputs, volatile=False)

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

            # Measure elapsed batch time (time per trial)
            batch_time.update(time.time() - end)

            print('[{0}/{1}]\t'
                  'Unorm batch time {batch_time.val:.4f} ({batch_time.avg:.3f})'
                  '50%|99% {2:.4f} | {3:.4f}\t'.format(
                (i + 1), trials_ran, np.percentile(batch_time.array, 50),
                np.percentile(batch_time.array, 99), batch_time=batch_time))

            if cuda:
                torch.cuda.synchronize()
            del out

    # WER, CER
    wer = total_wer / float(word_count)
    cer = total_cer / float(char_count)
    wer *= 100
    cer *= 100

    return wer, cer, batch_time
