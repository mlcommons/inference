from collections import OrderedDict

import pytest
import torch

from deepspeech.data.alphabet import Alphabet
from deepspeech.decoder import GreedyCTCDecoder


BLANK = '<blank>'
SYMBOLS = OrderedDict([(s, i) for i, s in enumerate([BLANK] + list('abcd'))])


@pytest.fixture
def alphabet():
    return Alphabet(SYMBOLS.keys())


def test_greedy_ctc_decoder_decode(alphabet):
    """Simple test to ensure GreedyCTCDecoder runs."""
    decoder = GreedyCTCDecoder(alphabet, BLANK)

    logits = torch.tensor([[[0.0, 0.5, 9.0, 0.0]],    # b
                           [[0.1, 0.1, 0.1, 4.5]],    # c
                           [[0.7, 7.2, 0.4, 0.9]],    # a
                           [[0.3, 8.1, 0.9, 0.5]],    # a
                           [[0.3, 8.1, 0.9, 0.5]],    # a
                           [[0.9, 0.9, 0.9, 1.0]],    # c
                           [[1.0, 0.1, 0.1, 0.1]],    # <blank>
                           [[0.5, 0.5, 0.5, 1.8]]])   # c

    actual = decoder.decode(logits, [8])
    assert len(actual) == 1
    assert actual[0] == 'bcacc'
