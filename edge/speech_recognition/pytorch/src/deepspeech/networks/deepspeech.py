from torch import nn

from deepspeech.networks.utils import OverLastDim


class Network(nn.Module):
    """A network with 3 FC layers, a Bi-LSTM, and 2 FC layers.

    Args:
        in_features: Number of input features per step per batch.
        n_hidden: Internal hidden unit size.
        out_features: Number of output features per step per batch.
        drop_prob: Dropout drop probability.
        relu_clip: ReLU clamp value: `min(max(0, x), relu_clip)`.
        forget_gate_bias: Total initialized value of the bias used in the
            forget gate. Set to None to use PyTorch's default initialisation.
            (See: http://proceedings.mlr.press/v37/jozefowicz15.pdf)
    """

    def __init__(self, in_features, n_hidden, out_features, drop_prob,
                 relu_clip=20.0, forget_gate_bias=1.0):
        super().__init__()

        self._relu_clip = relu_clip
        self._drop_prob = drop_prob

        self.fc1 = self._fully_connected(in_features, n_hidden)
        self.fc2 = self._fully_connected(n_hidden, n_hidden)
        self.fc3 = self._fully_connected(n_hidden, 2*n_hidden)
        self.bi_lstm = self._bi_lstm(2*n_hidden, n_hidden, forget_gate_bias)
        self.fc4 = self._fully_connected(2*n_hidden, n_hidden)
        self.out = self._fully_connected(n_hidden,
                                         out_features,
                                         relu=False,
                                         dropout=False)

    def _fully_connected(self, in_f, out_f, relu=True, dropout=True):
        layers = [nn.Linear(in_f, out_f)]
        if relu:
            layers.append(nn.Hardtanh(0, self._relu_clip, inplace=True))
        if dropout:
            layers.append(nn.Dropout(p=self._drop_prob))
        return OverLastDim(nn.Sequential(*layers))

    def _bi_lstm(self, input_size, hidden_size, forget_gate_bias):
        lstm = nn.LSTM(input_size=input_size,
                       hidden_size=hidden_size,
                       bidirectional=True)
        if forget_gate_bias is not None:
            for name in ['bias_ih_l0', 'bias_ih_l0_reverse']:
                bias = getattr(lstm, name)
                bias.data[hidden_size:2*hidden_size].fill_(forget_gate_bias)
            for name in ['bias_hh_l0', 'bias_hh_l0_reverse']:
                bias = getattr(lstm, name)
                bias.data[hidden_size:2*hidden_size].fill_(0)
        return lstm

    def forward(self, x):
        """Computes a single forward pass through the network.

        Args:
            x: A tensor of shape (seq_len, batch, in_features).

        Returns:
            Logits of shape (seq_len, batch, out_features).
        """
        h = self.fc1(x)
        h = self.fc2(h)
        h = self.fc3(h)
        h, _ = self.bi_lstm(h)
        h = self.fc4(h)
        out = self.out(h)
        return out
