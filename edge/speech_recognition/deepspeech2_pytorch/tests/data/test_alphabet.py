from collections import OrderedDict

import pytest

from deepspeech.data.alphabet import Alphabet


SYMBOLS = OrderedDict([(symbol, index) for index, symbol in enumerate('abcd')])


@pytest.fixture
def alphabet():
    return Alphabet(SYMBOLS.keys())


def test_duplicate_symbol_raise_valuerror():
    with pytest.raises(ValueError):
        Alphabet('aa')


def test_len(alphabet):
    assert len(alphabet) == len(SYMBOLS)


def test_iterator(alphabet):
    exp_symbols = list(SYMBOLS.keys())
    for index, symbol in enumerate(alphabet):
        assert symbol == exp_symbols[index]


def test_get_symbol(alphabet):
    for symbol, index in SYMBOLS.items():
        assert alphabet.get_symbol(index) == symbol


def test_get_index(alphabet):
    for symbol, index in SYMBOLS.items():
        assert alphabet.get_index(symbol) == index


def test_get_symbols(alphabet):
    sentence = ['a', 'b', 'b', 'c']
    indices = [0, 1, 1, 99, 2]
    actual = alphabet.get_symbols(indices)
    assert len(actual) == len(sentence)
    assert all([a == e for a, e in zip(actual, sentence)])


def test_get_indices(alphabet):
    sentence = ['a', 'b', 'b', 'invalid', 'c']
    indices = [0, 1, 1, 2]
    actual = alphabet.get_indices(sentence)
    assert len(actual) == len(indices)
    assert all([a == e for a, e in zip(actual, indices)])
