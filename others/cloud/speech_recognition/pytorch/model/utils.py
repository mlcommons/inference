import os
import errno
import sys

sys.path.append('../')
import json

from model import DeepSpeech, supported_rnns


def make_folder(folder):
    try:
        os.makedirs(folder)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory already exists.')
        else:
            raise


def get_labels(params):
    with open(params.labels_path) as label_file:
        labels = str(''.join(json.load(label_file)))
    return labels


def get_audio_conf(params):
    audio_conf = dict(sample_rate=params.sample_rate,
                      window_size=params.window_size,
                      window_stride=params.window_stride,
                      window=params.window,
                      noise_dir=params.noise_dir,
                      noise_prob=params.noise_prob,
                      noise_levels=(params.noise_min, params.noise_max))
    return audio_conf


def get_model(params):
    if params.rnn_type == 'gru' and params.rnn_act_type != 'tanh':
        print("ERROR: GRU does not currently support activations other than tanh")
        sys.exit()

    if params.rnn_type == 'rnn' and params.rnn_act_type != 'relu':
        print("ERROR: We should be using ReLU RNNs")
        sys.exit()

    with open(params.labels_path) as label_file:
        labels = str(''.join(json.load(label_file)))

    audio_conf = dict(sample_rate=params.sample_rate,
                      window_size=params.window_size,
                      window_stride=params.window_stride,
                      window=params.window,
                      noise_dir=params.noise_dir,
                      noise_prob=params.noise_prob,
                      noise_levels=(params.noise_min, params.noise_max))

    rnn_type = params.rnn_type.lower()
    assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"

    model = DeepSpeech(rnn_hidden_size=params.hidden_size,
                       nb_layers=params.hidden_layers,
                       labels=labels,
                       rnn_type=supported_rnns[rnn_type],
                       audio_conf=audio_conf,
                       bidirectional=False,
                       rnn_activation=params.rnn_act_type,
                       bias=params.bias)

    return model


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.array = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.array.append(val)
