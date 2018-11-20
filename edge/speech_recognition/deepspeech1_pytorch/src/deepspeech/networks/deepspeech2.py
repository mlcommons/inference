import math
from collections import OrderedDict

import torch
from torch.autograd import Variable
from torch import nn
from torch.nn.parameter import Parameter

from deepspeech.networks.utils import OverLastDim


SUPPORTED_RNNS = {
    'gru': nn.GRU,
    'lstm': nn.LSTM,
    'rnn': nn.RNN
}


class Network(nn.Module):
    """A network with 2 conv layers, N recurrent layers, and a FC layer.

    The architecture is based on the Deep Speech 2 paper:
        https://arxiv.org/abs/1512.02595

    The implementation is based on a PyTorch version:
        https://github.com/SeanNaren/deepspeech.pytorch

    Args:
        in_features: Number of input features per step per batch.
        n_hidden: Internal hidden unit size.
        out_features: Number of output features per step per batch.
        rnn_type: Type of recurrent neural network to use. See
            SUPPORTED_RNNS.keys() for a complete list.
        bidirectional: Recurrent neural networks are bidirectional if True.
        rnn_layers: Number of recurrent layers to stack.
        context: Number of look-ahead context frames to use if not
            bidirectional.
        relu_clip: ReLU clamp value: `min(max(0, x), relu_clip)`.
    """

    def __init__(self, in_features, n_hidden, out_features, rnn_type='lstm',
                 bidirectional=True, rnn_layers=5, context=20, relu_clip=20.0):
        super().__init__()
        self._relu_clip = relu_clip

        self._conv_layers()
        conv_features = self._conv_layer_feature_size(in_features)

        self._rnn_layers(in_features=conv_features,
                         n_hidden=n_hidden,
                         rnn_layers=rnn_layers,
                         rnn_type=SUPPORTED_RNNS[rnn_type],
                         bidirectional=bidirectional,
                         context=context)

        fully_connected = nn.Sequential(
            nn.BatchNorm1d(n_hidden),
            nn.Linear(n_hidden, out_features, bias=False)
        )
        self.fc = OverLastDim(fully_connected)

    def _conv_layers(self):
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=(41, 11),
                      stride=(2, 2),
                      padding=(0, 10)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, self._relu_clip, inplace=True),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=(21, 11),
                      stride=(2, 1),
                      padding=(0, 0)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, self._relu_clip, inplace=True)
        )

    @staticmethod
    def _conv_output_size(input_size, filter_size, padding, stride):
        """Returns the length of the output size for a convolution.

        Applies the standard formula:

            ((W - F + 2P) / S) + 1

        where:

            W: `input_size`
            F: `filter_size`
            P: `padding` on one side
            S: `stride`
        """
        return (float(input_size - filter_size + 2*padding) / stride) + 1

    @staticmethod
    def output_len(input_len):
        """Returns the length of the CTC matrix for a given input seq. len."""
        output_len = Network._conv_output_size(input_len, 11, 10, 2)
        output_len = Network._conv_output_size(output_len, 11, 0, 1)
        return int(output_len)

    def _conv_layer_feature_size(self, in_features):
        """Returns the number of features after processing with conv layers."""
        rnn_input_size = Network._conv_output_size(in_features, 41, 0, 2)
        rnn_input_size = Network._conv_output_size(rnn_input_size, 21, 0, 2)
        rnn_input_size *= 32   # Collapse channels.
        return int(rnn_input_size)

    def _rnn_layers(self, in_features, n_hidden, rnn_layers, rnn_type,
                    bidirectional, context):
        rnns = OrderedDict()
        for i in range(rnn_layers):
            rnn = RNNWrapper(input_size=in_features,
                             hidden_size=n_hidden,
                             rnn_type=rnn_type,
                             bidirectional=bidirectional,
                             batch_norm=i > 0)
            rnns[str(i)] = rnn
            in_features = n_hidden
        self.rnns = nn.Sequential(rnns)

        if not bidirectional:
            self.lookahead = nn.Sequential(
                Lookahead(n_hidden, context=context),
                nn.Hardtanh(0, self._relu_clip, inplace=True))

    def forward(self, x):
        """Computes a single forward pass through the network.

        Args:
            x: A tensor of shape (seq_len, batch, in_features).

        Returns:
            Logits of shape (seq_len, batch, out_features).
        """
        # T, N, H = seq_len, batch, features
        x = x.permute(1, 2, 0)   # TxNxH -> NxHxT
        x.unsqueeze_(dim=1)      # NxHxT -> Nx1xHxT
        x = self.conv(x)

        N, H1, H2, T = x.size()
        x = x.view(N, H1*H2, T)
        x = x.permute(2, 0, 1)   # NxHxT -> TxNxH
        x = self.rnns(x.contiguous())

        if hasattr(self, 'lookahead'):
            x = self.lookahead(x)

        return self.fc(x)


class RNNWrapper(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM,
                 bidirectional=False, batch_norm=True):
        """Bias-free RNN wrapper with optional batch norm and bidir summation.

        Instantiates an RNN without bias parameters. Optionally applies a batch
        normalisation layer to the input with the statistics computed over all
        time steps. If the RNN is bidirectional, the output from the forward
        and backward units is summed before return.
        """
        super().__init__()
        if batch_norm:
            self.batch_norm = OverLastDim(nn.BatchNorm1d(input_size))
        self.bidirectional = bidirectional
        self.rnn = rnn_type(input_size=input_size,
                            hidden_size=hidden_size,
                            bidirectional=bidirectional,
                            bias=False)

    def forward(self, x):
        if hasattr(self, 'batch_norm'):
            x = self.batch_norm(x)
        x, _ = self.rnn(x)
        if self.bidirectional:
            # TxNx(H*2) -> TxNxH by sum.
            seq_len, batch_size, _ = x.size()
            x = x.view(seq_len, batch_size, 2, -1) \
                 .sum(dim=2) \
                 .view(seq_len, batch_size, -1)
        return x


class Lookahead(nn.Module):
    """Wang et al 2016: Lookahead Conv. Layer for Unidir. Rec. Neural Nets.

    (31/07/2018, sam) This should be updated as it looks old but we aren't
    using it.
    """
    def __init__(self, n_features, context):
        # should we handle batch_first=True?
        super(Lookahead, self).__init__()
        self.n_features = n_features
        self.weight = Parameter(torch.Tensor(n_features, context + 1))
        assert context > 0
        self.context = context
        self.register_parameter('bias', None)
        self.init_parameters()

    def init_parameters(self):  # what's a better way initialiase this layer?
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        """
        Args:
            input: size [seq_len, batch_size, num_features]

        Returns:
            output shape - same as input
        """

        seq_len = input.size(0)
        # pad the 0th dimension (T/sequence) with zeroes whose number = context
        # Once pytorch's padding functions have settled, should move to those.
        padding = torch.zeros(self.context, *(input.size()[1:])) \
                       .type_as(input.data)
        x = torch.cat((input, Variable(padding)), 0)

        # add lookahead windows (with context+1 width) as a fourth dimension
        # for each seq-batch-feature combination

        # TxLxNxH - sequence, context, batch, feature
        x = [x[i:i + self.context + 1] for i in range(seq_len)]
        x = torch.stack(x)

        # TxNxHxL - sequence, batch, feature, context
        x = x.permute(0, 2, 3, 1)

        x = torch.mul(x, self.weight).sum(dim=3)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'n_features=' + str(self.n_features) \
               + ', context=' + str(self.context) + ')'
