import numpy as np
import torch
from torchvision.transforms import Compose

from deepspeech.data import preprocess
from deepspeech.models.model import Model
from deepspeech.networks.deepspeech2 import Network


class DeepSpeech2(Model):
    """Deep Speech 2 Model.

    Args:
        optimiser_cls: See `Model`.
        optimiser_kwargs: See `Model`.
        decoder_cls: See `Model`.
        decoder_kwargs: See `Model`.
        n_hidden (int): Internal hidden unit size.
        rnn_layers (int): Number of recurrent layers to stack.
        winlen (float): Window length in ms to compute input features over.
        winstep (float): Window step size in ms.
        sample_rate (int): Sample rate in Hz of input data.
        clip_gradients: See `Model`.

    Attributes:
        See base class.
    """

    def __init__(self, optimiser_cls=None, optimiser_kwargs=None,
                 decoder_cls=None, decoder_kwargs=None, n_hidden=2560,
                 rnn_layers=3, winlen=0.02, winstep=0.01, sample_rate=16000,
                 clip_gradients=400):

        self._n_hidden = n_hidden
        self._rnn_layers = rnn_layers
        self._winlen = winlen
        self._winstep = winstep
        self._sample_rate = sample_rate

        network = self._get_network()

        super().__init__(network=network,
                         optimiser_cls=optimiser_cls,
                         optimiser_kwargs=optimiser_kwargs,
                         decoder_cls=decoder_cls,
                         decoder_kwargs=decoder_kwargs,
                         clip_gradients=clip_gradients)

    def _get_network(self):
        return Network(in_features=int((self._sample_rate*self._winlen)//2+1),
                       n_hidden=self._n_hidden,
                       out_features=len(self.ALPHABET),
                       rnn_type='gru',
                       rnn_layers=self._rnn_layers,
                       bidirectional=False,
                       bn_between_rnns=False)

    @property
    def transform(self):
        return Compose([lambda t: t.astype(np.float32),
                        preprocess.LogMagnitudeSTFT(
                            winlen=self._winlen,
                            winstep=self._winstep,
                            sample_rate=self._sample_rate),
                        preprocess.Normalize(),
                        torch.from_numpy,
                        lambda t: (t, Network.output_len(len(t))),
                        ])

    @property
    def target_transform(self):
        return Compose([str.lower,
                        self.ALPHABET.get_indices,
                        torch.IntTensor])
