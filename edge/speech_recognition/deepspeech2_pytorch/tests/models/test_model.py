import pytest
import torch

from deepspeech.models import Model


class ModelStub(Model):
    def __init__(self):
        super().__init__(network=torch.nn.Linear(1, 1))

    @property
    def transform(self):
        return lambda x: x

    @property
    def target_transform(self):
        return lambda x: x


@pytest.fixture
def model_stub():
    return ModelStub()


def test_model_attrs_exist(model_stub):
    attrs = ['completed_epochs', 'BLANK_SYMBOL', 'ALPHABET', 'network',
             'decoder', 'optimiser', 'loss', 'transform', 'target_transform']
    for attr in attrs:
        assert hasattr(model_stub, attr)
