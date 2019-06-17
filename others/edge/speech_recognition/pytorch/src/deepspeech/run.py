import argparse
import logging
import os
import re

import torch
from torch.utils.data import DataLoader

from deepspeech.data.datasets import LibriSpeech
from deepspeech.data.loader import collate_input_sequences
from deepspeech.decoder import BeamCTCDecoder
from deepspeech.decoder import GreedyCTCDecoder
from deepspeech.global_state import GlobalState
from deepspeech.logging import LogLevelAction
from deepspeech.models import DeepSpeech
from deepspeech.models import DeepSpeech2
from deepspeech.models import Model


MODEL_CHOICES = ['ds1', 'ds2']


def main(args=None):
    """Train and evaluate a DeepSpeech or DeepSpeech2 network.

    Args:
        args (list str, optional): List of arguments to use. If `None`,
            defaults to `sys.argv`.
    """
    args = get_parser().parse_args(args)

    global_state = GlobalState(exp_dir=args.exp_dir,
                               log_frequency=args.slow_log_freq)

    init_logger(global_state.exp_dir, args.log_file)

    logging.debug(args)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    decoder_cls, decoder_kwargs = get_decoder(args)

    model = get_model(args, decoder_cls, decoder_kwargs, global_state.exp_dir)

    train_loader = get_train_loader(args, model)

    dev_loader = get_dev_loader(args, model)

    if train_loader is not None:
        for epoch in range(model.completed_epochs, args.n_epochs):
            maybe_eval(model, dev_loader, args.dev_log)
            model.train(train_loader)
            _save_model(args.model, model, args.exp_dir)

    maybe_eval(model, dev_loader, args.dev_log)


def maybe_eval(model, dev_loader, dev_log):
    """Evaluates `model` on `dev_loader` for each statistic in `dev_log`.

    Args:
        model: A `deepspeech.models.Model`.
        dev_loader (optional): A `torch.utils.data.DataLoader`. If `None`,
            evaluation is skipped.
        dev_log (optional): A list of strings, where each string refers to the
            name of a statistic to compute. Each statistic will be computed at
            most once. Supported statistics: ['loss', 'wer'].
    """
    if dev_loader is not None:
        for stat in set(dev_log):
            if stat == 'loss':
                model.eval_loss(dev_loader)
            elif stat == 'wer':
                model.eval_wer(dev_loader)
            else:
                raise ValueError('unknown evaluation stat request: %r' % stat)


def get_parser():
    """Returns an `argparse.ArgumentParser`."""
    parser = argparse.ArgumentParser(
        description='train and evaluate a DeepSpeech or DeepSpeech2 network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # logging -----------------------------------------------------------------

    parser.add_argument('--log_level',
                        action=LogLevelAction,
                        default='DEBUG',
                        help='logging level - see `logging` module')

    parser.add_argument('--slow_log_freq',
                        default=500,
                        type=int,
                        help='run slow logs every `slow_log_freq` batches')

    parser.add_argument('--exp_dir',
                        default=None,
                        help='path to directory to keep experimental data - '
                             'see `deepspeech.global_state.GlobalState`')

    parser.add_argument('--log_file',
                        nargs='?',
                        default='log.txt',
                        const=None,
                        help='filename to use for log file - logs written to '
                             'stderr if empty')

    # model -------------------------------------------------------------------

    parser.add_argument('model',
                        choices=MODEL_CHOICES,
                        help='model to train')

    parser.add_argument('--state_dict_path',
                        default=None,
                        help='path to initial state_dict to load into model - '
                             'takes precedence over '
                             '`--no_resume_from_exp_dir`')

    parser.add_argument('--no_resume_from_exp_dir',
                        action='store_true',
                        default=False,
                        help='do not load the last state_dict in exp_dir')

    # decoder -----------------------------------------------------------------

    parser.add_argument('--decoder',
                        default='greedy',
                        choices=['beam', 'greedy'],
                        help='decoder to use')

    parser.add_argument('--lm_path',
                        default=None,
                        help='path to language model - if None, no lm is used')

    parser.add_argument('--lm_weight',
                        default=None,
                        type=float,
                        help='language model weight in loss (i.e. alpha)')

    parser.add_argument('--word_weight',
                        default=None,
                        type=float,
                        help='word bonus weight in loss (i.e. beta)')

    parser.add_argument('--beam_width',
                        default=None,
                        type=int,
                        help='width of beam search')

    # optimizer ---------------------------------------------------------------

    parser.add_argument('--learning_rate',
                        default=0.0003,
                        type=float,
                        help='learning rate of Adam optimizer')

    # data --------------------------------------------------------------------

    parser.add_argument('--cachedir',
                        default='/tmp/data/cache/',
                        help='location to download dataset(s)')

    # training

    TRAIN_SUBSETS = ['train-clean-100',
                     'train-clean-360',
                     'train-other-500']
    parser.add_argument('--train_subsets',
                        default=TRAIN_SUBSETS,
                        choices=TRAIN_SUBSETS,
                        help='LibriSpeech subsets to train on',
                        nargs='*')

    parser.add_argument('--train_batch_size',
                        default=16,
                        type=int,
                        help='number of samples in a training batch')

    parser.add_argument('--train_num_workers',
                        default=4,
                        type=int,
                        help='number of subprocesses for train DataLoader')

    # validation

    parser.add_argument('--dev_log',
                        default=['loss', 'wer'],
                        choices=['loss', 'wer'],
                        nargs='*',
                        help='validation statistics to log')

    parser.add_argument('--dev_subsets',
                        default=['dev-clean', 'dev-other'],
                        choices=['dev-clean', 'dev-other',
                                 'test-clean', 'test-other'],
                        help='LibriSpeech subsets to evaluate loss and WER on',
                        nargs='*')

    parser.add_argument('--dev_batch_size',
                        default=16,
                        type=int,
                        help='number of samples in a validation batch')

    parser.add_argument('--dev_num_workers',
                        default=4,
                        type=int,
                        help='number of subprocesses for dev DataLoader')

    # outer loop --------------------------------------------------------------

    parser.add_argument('--n_epochs',
                        default=15,
                        type=int,
                        help='number of epochs')

    # -------------------------------------------------------------------------

    return parser


def init_logger(exp_dir, log_file):
    """Initialises the `logging.Logger`."""
    logger = logging.getLogger()

    formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(funcName)s -'
                                      ' %(levelname)s: %(message)s')

    if log_file is not None:
        handler = logging.FileHandler(os.path.join(exp_dir, log_file))
    else:
        handler = logging.StreamHandler()

    handler.setFormatter(formatter)

    logger.addHandler(handler)

    logger.setLevel(logging.DEBUG)


def get_model(args, decoder_cls, decoder_kwargs, exp_dir):
    """Returns a `deepspeech.models.Model`.

    Args:
        args: An `argparse.Namespace` for the `argparse.ArgumentParser`
            returned by `get_parser`.
        decoder_cls: See `deepspeech.models.Model`.
        decoder_kwargs: See `deepspeech.models.Model`.
        exp_dir: path to directory where all experimental data will be stored.
    """
    model_cls = {'ds1': DeepSpeech, 'ds2': DeepSpeech2}[args.model]

    model = model_cls(optimiser_cls=torch.optim.Adam,
                      optimiser_kwargs={'lr': args.learning_rate},
                      decoder_cls=decoder_cls,
                      decoder_kwargs=decoder_kwargs)

    state_dict_path = args.state_dict_path
    if state_dict_path is None and not args.no_resume_from_exp_dir:
        # Restore from last saved `state_dict` in `exp_dir`.
        state_dict_path = _get_last_state_dict_path(args.model, exp_dir)

    if state_dict_path is not None:
        # Restore from user-specified `state_dict`.
        logging.debug('restoring state_dict at %s' % state_dict_path)
        model.load_state_dict(torch.load(state_dict_path))
    else:
        logging.debug('using randomly initialised model')
        _save_model(args.model, model, exp_dir)

    return model


def get_decoder(args):
    """Returns a `deepspeech.decoder.Decoder`.

    Args:
        args: An `argparse.Namespace` for the `argparse.ArgumentParser`
            returned by `get_parser`.
    """
    decoder_kwargs = {'alphabet': Model.ALPHABET,
                      'blank_symbol': Model.BLANK_SYMBOL}

    if args.decoder == 'beam':
        decoder_cls = BeamCTCDecoder

        if args.lm_weight is not None:
            decoder_kwargs['alpha'] = args.lm_weight
        if args.word_weight is not None:
            decoder_kwargs['beta'] = args.word_weight
        if args.beam_width is not None:
            decoder_kwargs['beam_width'] = args.beam_width
        if args.lm_path is not None:
            decoder_kwargs['model_path'] = args.lm_path

    elif args.decoder == 'greedy':
        decoder_cls = GreedyCTCDecoder

        beam_args = ['lm_weight', 'word_weight', 'beam_width', 'lm_path']
        for arg in beam_args:
            if getattr(args, arg) is not None:
                raise ValueError('greedy decoder selected but %r is not '
                                 'None' % arg)

    return decoder_cls, decoder_kwargs


def all_state_dicts(model_str, exp_dir):
    """Returns a dict of (epoch, filename) for all state_dicts in `exp_dir`.

    Args:
        model_str: Model whose state_dicts to consider.
        exp_dir: path to directory where all experimental data will be stored.
    """
    state_dicts = {}

    for f in os.listdir(exp_dir):
        match = re.match('(%s-([0-9]+).pt)' % model_str, f)
        if not match:
            continue

        groups = match.groups()
        name = groups[0]
        epoch = groups[1]

        state_dicts[epoch] = name

    return state_dicts


def _get_last_state_dict_path(model_str, exp_dir):
    """Returns the absolute path of the last state_dict in `exp_dir` or `None`.

    Args:
        model_str: Model whose state_dicts to consider.
        exp_dir: path to directory where all experimental data will be stored.
    """
    state_dicts = all_state_dicts(model_str, exp_dir)

    if len(state_dicts) == 0:
        return None

    last_epoch = sorted(state_dicts.keys())[-1]

    return os.path.join(exp_dir, state_dicts[last_epoch])


def _save_model(model_str, model, exp_dir):
    """Saves the model's `state_dict` in `exp_dir`.

    Args:
        model_str: Argument name of `model`.
        model: A `deepspeech.models.Model`.
        exp_dir: path to directory where the `model`'s `state_dict` will be
            stored.
    """
    save_name = '%s-%d.pt' % (model_str, model.completed_epochs)
    save_path = os.path.join(exp_dir, save_name)
    torch.save(model.state_dict(), save_path)


def get_train_loader(args, model):
    """Returns a `torch.nn.DataLoader over the training data.

    Args:
        args: An `argparse.Namespace` for the `argparse.ArgumentParser`
            returned by `get_parser`.
        model: A `deepspeech.models.Model`.
    """
    if len(args.train_subsets) == 0:
        logging.debug('no `train_subsets` specified')
        return

    todo_epochs = args.n_epochs - model.completed_epochs
    if todo_epochs <= 0:
        logging.debug('`n_epochs` <= `model.completed_epochs`')
        return

    train_cache = os.path.join(args.cachedir, 'train')
    train_dataset = LibriSpeech(root=train_cache,
                                subsets=args.train_subsets,
                                transform=model.transform,
                                target_transform=model.target_transform,
                                download=True)

    return DataLoader(train_dataset,
                      collate_fn=collate_input_sequences,
                      pin_memory=torch.cuda.is_available(),
                      num_workers=args.train_num_workers,
                      batch_size=args.train_batch_size,
                      shuffle=True)


def get_dev_loader(args, model):
    """Returns a `torch.nn.DataLoader over the validation data.

    Args:
        args: An `argparse.Namespace` for the `argparse.ArgumentParser`
            returned by `get_parser`.
        model: A `deepspeech.models.Model`.
    """
    if len(args.dev_subsets) == 0:
        logging.debug('no `dev_subsets` specified')
        return

    if len(args.dev_log) == 0:
        logging.debug('no `dev_log` statistics specified')
        return

    dev_cache = os.path.join(args.cachedir, 'dev')
    dev_dataset = LibriSpeech(root=dev_cache,
                              subsets=args.dev_subsets,
                              transform=model.transform,
                              target_transform=model.target_transform,
                              download=True)

    return DataLoader(dev_dataset,
                      collate_fn=collate_input_sequences,
                      pin_memory=torch.cuda.is_available(),
                      num_workers=args.dev_num_workers,
                      batch_size=args.dev_batch_size,
                      shuffle=False)
