import pytest
import torch

from deepspeech.models import DeepSpeech2


@pytest.fixture
def model():
    return DeepSpeech2()


def test_load_state_dict_restores_parameters(model):
    act_model = DeepSpeech2()
    act_model.load_state_dict(model.state_dict())

    # Naive equality check: all network parameters are equal.
    exp_params = dict(model.network.named_parameters())
    print(exp_params.keys())
    for act_name, act_param in act_model.network.named_parameters():
        assert torch.allclose(act_param.float(), exp_params[act_name].float())
