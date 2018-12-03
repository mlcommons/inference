import copy
import psutil
import time

import torch
from torch.nn.utils import clip_grad_norm_

from deepspeech.data.alphabet import Alphabet
from deepspeech.decoder import GreedyCTCDecoder
from deepspeech.global_state import GlobalState
from deepspeech.logging import log_call_info
from deepspeech.logging import LoggerMixin
from deepspeech.loss import CTCLoss
from deepspeech.loss import levenshtein
from deepspeech.networks.utils import to_cuda


_BLANK_SYMBOL = '_'


def _gen_alphabet():
    symbols = [_BLANK_SYMBOL]
    symbols.extend("'abcdefghijklmnopqrstuvwxyz ")
    return Alphabet(symbols)


class Model(LoggerMixin):
    """A speech-to-text model.

    Args:
        network: A speech-to-text `torch.nn.Module`.
        optimiser_cls (callable, optional): If not None, this optimiser will be
            instantiated with an OrderedDict of the network parameters as the
            first argument and **optimiser_kwargs as the remaining arguments
            unless they are None.
        optimiser_kwargs (dict, optional): A dictionary of arguments to pass to
            the optimiser when it is created. Defaults to the empty dictionary
            if None.
        decoder_cls (callable, optional): A callable that implements the
            `deepspeech.decoder.Decoder` interface. Defaults to
            `DEFAULT_DECODER_CLS` if None.
        decoder_kwargs (dict): A dictionary of arguments to pass to the decoder
            when it is created. Defaults to `DEFAULT_DECODER_KWARGS` if
            `decoder_kwargs` is None.
        clip_gradients (int, optional): If None no gradient clipping is
            performed. If an int, it is used as the `max_norm` parameter to
            `torch.nn.utils.clip_grad_norm`.

    Attributes:
        BLANK_SYMBOL: The string that denotes the blank symbol in the CTC
            algorithm.
        ALPHABET: A `deepspeech.data.alphabet.Alphabet` - contains
            `BLANK_SYMBOL`.
        DEFAULT_DECODER_CLS: See Args.
        DEFAULT_DECODER_KWARGS: See Args.
        completed_epochs: Number of epochs completed during training.
        network: See Args.
        decoder: A `deepspeech.decoder.BeamCTCDecoder` instance.
        optimiser: An `optimiser_cls` instance or None.
        loss: A `deepspeech.loss.CTCLoss` instance.
        transform: A function that returns a transformed piece of audio data.
        target_transform: A function that returns a transformed target.
    """
    BLANK_SYMBOL = _BLANK_SYMBOL
    ALPHABET = _gen_alphabet()

    DEFAULT_DECODER_CLS = GreedyCTCDecoder
    DEFAULT_DECODER_KWARGS = {'alphabet': ALPHABET,
                              'blank_symbol': BLANK_SYMBOL}

    def __init__(self, network, optimiser_cls=None, optimiser_kwargs=None,
                 decoder_cls=None, decoder_kwargs=None, clip_gradients=None):
        self.completed_epochs = 0

        self._optimiser_cls = optimiser_cls
        self._optimiser_kwargs = optimiser_kwargs
        self._clip_gradients = clip_gradients

        self._init_network(network)
        self._init_decoder(decoder_cls, decoder_kwargs)
        self._init_optimiser()
        self._init_loss()

        self._global_state = GlobalState.get_or_init_singleton()

    def _init_network(self, network):
        if not torch.cuda.is_available():
            self._logger.info('CUDA not available')
        else:
            self._logger.info('CUDA available, moving network '
                              'parameters and buffers to the GPU')
            to_cuda(network)

        self.network = network

    def _init_decoder(self, decoder_cls, decoder_kwargs):
        if decoder_cls is None:
            decoder_cls = self.DEFAULT_DECODER_CLS

        if decoder_kwargs is None:
            decoder_kwargs = copy.copy(self.DEFAULT_DECODER_KWARGS)

        self.decoder = decoder_cls(**decoder_kwargs)

    def _init_optimiser(self):
        self.reset_optimiser()

    def reset_optimiser(self):
        """Assigns a new `self.optimiser` using the current network params."""
        if self._optimiser_cls is None:
            self.optimiser = None
            self._logger.debug('No optimiser specified')
            return

        kwargs = self._optimiser_kwargs or {}
        opt = self._optimiser_cls(self.network.parameters(), **kwargs)

        self.optimiser = opt

    def _init_loss(self):
        blank_index = self.ALPHABET.get_index(self.BLANK_SYMBOL)
        self.loss = CTCLoss(blank_index=blank_index,
                            size_average=False,
                            length_average=False)

    @property
    def transform(self):
        raise NotImplementedError

    @property
    def target_transform(self):
        raise NotImplementedError

    def state_dict(self):
        state = {'completed_epochs': self.completed_epochs,
                 'network': self.network.state_dict(),
                 'global_state': self._global_state.state_dict()}
        if self.optimiser is not None:
            state['optimiser'] = self.optimiser.state_dict()
        return state

    def load_state_dict(self, state_dict):
        self.completed_epochs = state_dict['completed_epochs']
        self.network.load_state_dict(state_dict['network'])
        self._global_state.load_state_dict(state_dict['global_state'])
        if self.optimiser is not None:
            self.optimiser.load_state_dict(state_dict['optimiser'])

    @property
    def _zero_grad(self):
        return lambda: self.network.zero_grad()

    @property
    def _backward(self):
        return lambda batch_loss: batch_loss.backward()

    @property
    def _maybe_clip_gradients(self):
        if self._clip_gradients is None:
            return lambda: None

        return lambda: clip_grad_norm_(self.network.parameters(),
                                       self._clip_gradients)

    @log_call_info
    def train(self, loader):
        """Trains the Model for an epoch.

        Args:
            loader: A `torch.utils.data.DataLoader` that generates batches of
                training data.
        """
        if self.optimiser is None:
            raise AttributeError('Cannot train when optimiser is None!')

        self.network.train()
        self._train_log_init()
        epoch_loss = 0.0
        total_samples = 0

        data_iter = iter(loader)   # Explicit creation to log queue sizes.
        for step, ((x, logit_lens), y) in enumerate(data_iter):
            self._zero_grad()

            logits = self.network(x)

            batch_loss = self.loss(logits, y, logit_lens)

            epoch_loss += batch_loss.item()

            total_samples += len(logit_lens)

            self._backward(batch_loss)

            self._maybe_clip_gradients()

            self.optimiser.step()

            self._train_log_step(step, x, logits, logit_lens, batch_loss.item(), data_iter)  # noqa: E501

            self._global_state.step += 1

            del logits, x, logit_lens, y

        self._train_log_end(epoch_loss, total_samples)
        self.completed_epochs += 1

    @log_call_info
    def eval_wer(self, loader):
        """Evaluates the WER of the Model.

        Args:
            loader: A `torch.utils.data.DataLoader` that generates batches of
                data.
        """
        self.network.eval()

        total_lev = 0
        total_lab_len = 0
        n = 0

        self._logger.debug('idx,model_label_prediction,target,edit_distance')
        for i, ((x, logit_lens), y) in enumerate(loader):
            with torch.no_grad():   # Ensure the gradient isn't computed.
                logits = self.network(x)

            preds = self.decoder.decode(logits.cpu(), logit_lens)
            acts = [''.join(self.ALPHABET.get_symbols(yi.data.numpy()))
                    for yi in y]

            for pred, act in zip(preds, acts):
                lev = levenshtein(pred.split(), act.split())

                self._logger.debug('%d,%r,%r,%d', n, pred, act, lev)

                n += 1
                total_lev += lev
                total_lab_len += len(act.split())

        wer = float(total_lev) / total_lab_len
        self._logger.debug('eval/wer: %r', wer)
        self._global_state.writer.add_scalar('eval/wer',
                                             wer, self._global_state.step)

        return wer

    @log_call_info
    def eval_loss(self, loader):
        """Evaluates the CTC loss of the Model.

        Args:
            loader: A `torch.utils.data.DataLoader` that generates batches of
                data.
        """
        self.network.eval()

        total_loss = 0.0
        total_samples = 0

        self._logger.debug('idx,batch_mean_sample_loss')

        for i, ((x, logit_lens), y) in enumerate(loader):
            with torch.no_grad():   # Ensure the gradient isn't computed.
                logits = self.network(x)

                batch_loss = self.loss(logits, y, logit_lens).item()
                batch_samples = len(logit_lens)

                total_loss += batch_loss
                total_samples += batch_samples

            self._logger.debug('%d,%f', i, batch_loss / batch_samples)

        mean_sample_loss = total_loss / max(1, total_samples)
        self._logger.debug('eval/mean_sample_loss: %f', mean_sample_loss)
        self._global_state.writer.add_scalar('eval/mean_sample_loss',
                                             mean_sample_loss,
                                             self._global_state.step)
        return mean_sample_loss

    def _train_log_init(self):
        header = 'step,global_step,completed_epochs,sum_logit_lens,loss'
        self._logger.debug(header)
        self._cum_batch_size = 0

    def _train_log_step(self, step, x, logits, logit_lens, loss, data_iter):
        start = time.time()

        total_steps = logit_lens.sum().item()

        self._logger.debug('%d,%d,%d,%d,%f',
                           step,
                           self._global_state.step,
                           self.completed_epochs,
                           total_steps,
                           loss)

        self._global_state.writer.add_scalar('train/batch_loss',
                                             loss,
                                             self._global_state.step)
        self._global_state.writer.add_scalar('train/batch_size',
                                             len(logit_lens),
                                             self._global_state.step)

        self._cum_batch_size += len(logit_lens)
        self._global_state.writer.add_scalar('train/epoch_cum_batch_size',
                                             self._cum_batch_size,
                                             self._global_state.step)

        self._global_state.writer.add_scalar('train/batch_len-x-batch_size',
                                             x.size(0) * x.size(1),
                                             self._global_state.step)
        self._global_state.writer.add_scalar('train/sum_logit_lens',
                                             total_steps,
                                             self._global_state.step)
        self._global_state.writer.add_scalar('train/memory_percent',
                                             psutil.Process().memory_percent(),
                                             self._global_state.step)

        self._train_log_step_data_queue(data_iter)

        self._train_log_step_cuda_memory()

        self._train_log_step_grad_param_stats()

        self._global_state.writer.add_scalar('train/log_step_time',
                                             time.time() - start,
                                             self._global_state.step)

    def _train_log_step_data_queue(self, data_iter):
        """Logs the number of batches in the PyTorch DataLoader queue."""
        # If num_workers is 0 then there is no Queue and each batch is loaded
        # when next is called.
        if data_iter.num_workers > 0:
            # Otherwise there exists a queue from which samples are read from.
            if data_iter.pin_memory or data_iter.timeout > 0:
                # The loader iterator in PyTorch 0.4 with pin_memory or a
                # timeout has a single thread fill a queue.Queue from a
                # multiprocessing.SimpleQueue that is filled by num_workers
                # other workers. The queue.Queue is used when next is called.
                # See: https://pytorch.org/docs/0.4.0/_modules/torch/utils/data/dataloader.html#DataLoader   # noqa: E501
                self._global_state.writer.add_scalar(
                    'train/queue_size',
                    data_iter.data_queue.qsize(),
                    self._global_state.step)
            else:
                # Otherwise the loader iterator reads from a
                # multiprocessing.SimpleQueue. This has no size function...
                self._global_state.writer.add_scalar(
                    'train/queue_empty',
                    data_iter.data_queue.empty(),
                    self._global_state.step)

    def _train_log_step_cuda_memory(self):
        """Logs CUDA memory usage."""
        if torch.cuda.is_available():
            self._global_state.writer.add_scalar(
                'train/memory_allocated',
                torch.cuda.memory_allocated(),
                self._global_state.step)
            self._global_state.writer.add_scalar(
                'train/max_memory_allocated',
                torch.cuda.max_memory_allocated(),
                self._global_state.step)
            self._global_state.writer.add_scalar(
                'train/memory_cached',
                torch.cuda.memory_cached(),
                self._global_state.step)
            self._global_state.writer.add_scalar(
                'train/max_memory_cached',
                torch.cuda.max_memory_cached(),
                self._global_state.step)

    def _train_log_step_grad_param_stats(self):
        """Logs gradient and parameter values."""
        if self._global_state.log_step():
            for name, param in self.network.named_parameters():
                self._global_state.writer.add_histogram(
                    'parameters/%s' % name, param, self._global_state.step)

                self._global_state.writer.add_histogram(
                    'gradients/%s' % name, param.grad, self._global_state.step)

    def _train_log_end(self, epoch_loss, total_samples):
        mean_sample_loss = float(epoch_loss) / total_samples
        self._logger.debug('train/mean_sample_loss: %r', mean_sample_loss)
        self._logger.info('epoch %d finished', self.completed_epochs)

        self._global_state.writer.add_scalar('train/mean_sample_loss',
                                             mean_sample_loss,
                                             self._global_state.step)
