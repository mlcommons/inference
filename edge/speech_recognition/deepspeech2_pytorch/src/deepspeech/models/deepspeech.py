import torch
from torchvision.transforms import Compose

from deepspeech.data import preprocess
from deepspeech.models.model import Model
from deepspeech.networks.deepspeech import Network


class DeepSpeech(Model):
    """Deep Speech Model.

    Args:
        optimiser_cls: See `Model`.
        optimiser_kwargs: See `Model`.
        decoder_cls: See `Model`.
        decoder_kwargs: See `Model`.
        n_hidden (int): Internal hidden unit size.
        n_context (int): Number of context frames to use on each side of the
            current input frame.
        n_mfcc (int): Number of Mel-Frequency Cepstral Coefficients to use as
            input for a single frame.
        drop_prob (float): Dropout drop probability, [0.0, 1.0] inclusive.
        winlen (float): Window length in ms to compute input features over.
        winstep (float): Window step size in ms.
        sample_rate (int): Sample rate in Hz of input data.

    Attributes:
        See base class.
    """

    def __init__(self, optimiser_cls=None, optimiser_kwargs=None,
                 decoder_cls=None, decoder_kwargs=None,
                 n_hidden=2048, n_context=9, n_mfcc=26, drop_prob=0.25,
                 winlen=0.025, winstep=0.02, sample_rate=16000):

        self._n_hidden = n_hidden
        self._n_context = n_context
        self._n_mfcc = n_mfcc
        self._drop_prob = drop_prob
        self._winlen = winlen
        self._winstep = winstep
        self._sample_rate = sample_rate

        network = self._get_network()

        super().__init__(network=network,
                         optimiser_cls=optimiser_cls,
                         optimiser_kwargs=optimiser_kwargs,
                         decoder_cls=decoder_cls,
                         decoder_kwargs=decoder_kwargs,
                         clip_gradients=None)

    def _get_network(self):
        return Network(in_features=self._n_mfcc*(2*self._n_context + 1),
                       n_hidden=self._n_hidden,
                       out_features=len(self.ALPHABET),
                       drop_prob=self._drop_prob)

    @property
    def transform(self):
        return Compose([preprocess.MFCC(self._n_mfcc),
                        preprocess.AddContextFrames(self._n_context),
                        preprocess.Normalize(),
                        torch.FloatTensor,
                        lambda t: (t, len(t))])

    @property
    def target_transform(self):
        return Compose([str.lower,
                        self.ALPHABET.get_indices,
                        torch.IntTensor])
