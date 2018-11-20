import random

import torch

from deepspeech.networks.deepspeech import Network


def test_forget_gate_bias_correct_initialisation():
    """Ensures the forget gate's bias is set to the desired value.

    NVIDIA's cuDNN LSTM has _two_ bias vectors for the forget gate. i.e.

    ```
    f_t = sigmoid(W_x*x_t + b_x + W_h*h_t + b_h)
    ```

    where for time step `t`:
        - `W_x` is the input matrix
        - `x_t` is the input
        - `b_x` is the input bias
        - `W_h` is the hidden matrix
        - `h_t` is the hidden state
        - `b_h` is the hidden bias

    The sum total of `b_x` and `b_h` should equal the desired
    `forget_gate_bias`.
    """
    for seed in range(10):
        # repeat test for a few different seeds
        random.seed(seed)
        torch.manual_seed(seed)

        # select some arbitrary hyperparameters
        in_features = random.randint(1, 6144)
        n_hidden = random.randint(1, 6144)
        out_features = random.randint(1, 6144)
        drop_prob = random.random()

        # select arbitrary forget gate bias in [-50.0, 50.0)
        forget_gate_bias = (random.random() - 0.5) * 100

        net = Network(in_features=in_features,
                      n_hidden=n_hidden,
                      out_features=out_features,
                      drop_prob=drop_prob,
                      forget_gate_bias=forget_gate_bias)

        params = dict(net.named_parameters())

        for dir in ['', '_reverse']:
            bias_sum = params['bi_lstm.bias_ih_l0' + dir].data
            bias_sum += params['bi_lstm.bias_hh_l0' + dir].data

            assert torch.allclose(bias_sum[n_hidden:2*n_hidden],
                                  torch.tensor(forget_gate_bias))
