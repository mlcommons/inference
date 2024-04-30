# Copyright (c) 2019, Myrtle Software Limited. All rights reserved.
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import os
from typing import Optional, Tuple


def rnn(rnn, input_size, hidden_size, num_layers, norm=None,
        forget_gate_bias=1.0, dropout=0.0, **kwargs):
    """TODO"""
    if rnn != "lstm":
        raise ValueError(f"Unknown rnn={rnn}")
    if norm not in [None]:
        raise ValueError(f"unknown norm={norm}")

    if rnn == "lstm":
        return LstmDrop(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            forget_gate_bias=forget_gate_bias,
            **kwargs
        )


class LstmDrop(torch.nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout, forget_gate_bias,
                 **kwargs):
        """Returns an LSTM with forget gate bias init to `forget_gate_bias`.

        Args:
            input_size: See `torch.nn.LSTM`.
            hidden_size: See `torch.nn.LSTM`.
            num_layers: See `torch.nn.LSTM`.
            dropout: See `torch.nn.LSTM`.
            forget_gate_bias: For each layer and each direction, the total value of
                to initialise the forget gate bias to.

        Returns:
            A `torch.nn.LSTM`.
        """
        super(LstmDrop, self).__init__()
        self.dev = torch.device("cuda:0") if torch.cuda.is_available() and os.environ.get("USE_GPU", "").lower() not in  [ "no", "false" ]  else torch.device("cpu")

        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        if forget_gate_bias is not None:
            for name, v in self.lstm.named_parameters():
                if "bias_ih" in name:
                    bias = getattr(self.lstm, name)
                    bias.data[hidden_size:2 * hidden_size].fill_(forget_gate_bias)
                if "bias_hh" in name:
                    bias = getattr(self.lstm, name)
                    bias.data[hidden_size:2 * hidden_size].fill_(0)

        if dropout:
            self.inplace_dropout = torch.nn.Dropout(dropout, inplace=True)
        else:
            self.inplace_droput = None

    def forward(self, x: torch.Tensor,
                h: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        x, h = self.lstm(x, h)

        if self.inplace_dropout is not None:
            self.inplace_dropout(x.data)

        return x, h


class StackTime(torch.nn.Module):

    __constants__ = ["factor"]

    def __init__(self, factor):
        super().__init__()
        self.factor = int(factor)
        self.dev = torch.device("cuda:0") if torch.cuda.is_available() and os.environ.get("USE_GPU", "").lower() not in  [ "no", "false" ]  else torch.device("cpu")


    def forward(self, x, x_lens):
        # T, B, U
        r = torch.transpose(x, 0, 1).to(self.dev)
        s = r.shape
        zeros = torch.zeros(
            s[0], (-s[1]) % self.factor, s[2], dtype=r.dtype, device=r.device)
        r = torch.cat([r, zeros], 1)
        s = r.shape
        rs = [s[0], s[1] // self.factor, s[2] * self.factor]
        r = torch.reshape(r, rs)
        rt = torch.transpose(r, 0, 1)
        x_lens = torch.ceil(x_lens.float() / self.factor).int()
        return rt, x_lens
