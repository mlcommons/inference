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

import torch.nn.functional as F
from model_separable_rnnt import label_collate
from typing import List, Optional, Tuple


class TransducerDecoder:
    """Decoder base class.

    Args:
        alphabet: An Alphabet object.
        blank_symbol: The symbol in `alphabet` to use as the blank during CTC
            decoding.
        model: Model to use for prediction.
    """

    def __init__(self, blank_index, model):
        model.eval()
        self._model = torch.jit.script(model)
        self._SOS = -1   # start of sequence
        self._blank_id = blank_index

    def _pred_step(self, label, hidden, device):
        if label == self._SOS:
            return self._model.prediction(None, hidden)
        if label > self._blank_id:
            label -= 1
        label = label_collate([[label]]).to(device)
        return self._model.prediction(label, hidden)

    def _joint_step(self, enc, pred, log_normalize: bool=False):
        logits = self._model.joint(enc, pred)[:, 0, 0, :]
        if not log_normalize:
            return logits

        probs = F.log_softmax(logits, dim=len(logits.shape) - 1)

        return probs

    def _get_last_symb(self, labels: List) -> int:
        return self._SOS if labels == [] else labels[-1]


class RNNTGreedyDecoder(TransducerDecoder):
    """A greedy transducer decoder.

    Args:
        blank_symbol: See `Decoder`.
        model: Model to use for prediction.
        max_symbols_per_step: The maximum number of symbols that can be added
            to a sequence in a single time step; if set to None then there is
            no limit.
        cutoff_prob: Skip to next step in search if current highest character
            probability is less than this.
    """

    def __init__(self, blank_index, model, max_symbols_per_step=30):
        super().__init__(blank_index, model)
        assert max_symbols_per_step is None or max_symbols_per_step > 0
        self.max_symbols = max_symbols_per_step

    # Hard to create a TorchScript Module here, because onnx outputs
    # must be Tensors, I believe. Maybe I can make a list of integers
    # and then convert to a Tensor, though.
    def decode(self, x, out_lens):
        """Returns a list of sentences given an input batch.

        Args:
            x: A tensor of size (batch, channels, features, seq_len)
                TODO was (seq_len, batch, in_features).
            out_lens: list of int representing the length of each sequence
                output sequence.

        Returns:
            list containing batch number of sentences (strings).
        """
        # Apply optional preprocessing

        logits, out_lens = self._model.encoder(x, out_lens)

        output = []
        for batch_idx in range(logits.size(0)):
            inseq = logits[batch_idx, :, :].unsqueeze(1)
            logitlen = out_lens[batch_idx]
            sentence = self._greedy_decode(inseq, logitlen)
            output.append(sentence)

        return logits, out_lens, output

    def _greedy_decode(self, x, out_len):
        training_state = self._model.training

        device = x.device

        hidden = None
        label = []
        for time_idx in range(out_len):
            f = x[time_idx, :, :].unsqueeze(0)

            not_blank = True
            symbols_added = 0

            while not_blank and (self.max_symbols is None or symbols_added < self.max_symbols):
                g, hidden_prime = self._pred_step(
                    self._get_last_symb(label),
                    hidden,
                    device
                )
                logp = self._joint_step(f, g, log_normalize=False)[0, :]

                # get index k, of max prob
                v, k = logp.max(0)
                k = k.item()

                if k == self._blank_id:
                    not_blank = False
                else:
                    label.append(k)
                    hidden = hidden_prime
                symbols_added += 1

        self._model.train(training_state)
        return label


class ScriptGreedyDecoder(torch.nn.Module):
    """A greedy transducer decoder.

    Args:
        blank_symbol: See `Decoder`.
        model: Model to use for prediction.
        max_symbols_per_step: The maximum number of symbols that can be added
            to a sequence in a single time step; if set to None then there is
            no limit.
        cutoff_prob: Skip to next step in search if current highest character
            probability is less than this.
    """

    def __init__(self, blank_index, model, max_symbols_per_step=30):
        super().__init__()
        assert isinstance(model, torch.jit.ScriptModule)
        # assert not model.training
        self.eval()
        self._model = model
        self._blank_id = blank_index
        self._SOS = -1
        assert max_symbols_per_step > 0
        self._max_symbols_per_step = max_symbols_per_step

    @torch.jit.export
    def forward(self, x: torch.Tensor, out_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]]]:
        """Returns a list of sentences given an input batch.

        Args:
            x: A tensor of size (batch, channels, features, seq_len)
                TODO was (seq_len, batch, in_features).
            out_lens: list of int representing the length of each sequence
                output sequence.

        Returns:
            list containing batch number of sentences (strings).
        """
        # Apply optional preprocessing

        logits, out_lens = self._model.encoder(x, out_lens)

        output: List[List[int]] = []
        for batch_idx in range(logits.size(0)):
            inseq = logits[batch_idx, :, :].unsqueeze(1)
            # inseq: TxBxF
            logitlen = out_lens[batch_idx]
            sentence = self._greedy_decode(inseq, logitlen)
            output.append(sentence)

        return logits, out_lens, output

    def _greedy_decode(self, x: torch.Tensor, out_len: torch.Tensor) -> List[int]:
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        label: List[int] = []
        for time_idx in range(int(out_len.item())):
            f = x[time_idx, :, :].unsqueeze(0)

            not_blank = True
            symbols_added = 0

            while not_blank and symbols_added < self._max_symbols_per_step:
                g, hidden_prime = self._pred_step(
                    self._get_last_symb(label),
                    hidden
                )
                logp = self._joint_step(f, g, log_normalize=False)[0, :]

                # get index k, of max prob
                v, k = logp.max(0)
                k = k.item()

                if k == self._blank_id:
                    not_blank = False
                else:
                    label.append(k)
                    hidden = hidden_prime
                symbols_added += 1

        return label

    def _pred_step(self, label: int, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if label == self._SOS:
            return self._model.prediction(None, hidden)
        if label > self._blank_id:
            label -= 1
        # Most likely to cause an error
        label = torch.tensor([[label]], dtype=torch.int64)
        # label = torch.nn.utils.rnn.pad_sequence([torch.Tensor(label, dtype=torch.int64)],
        #                                         batch_first=False)
        # label = label_collate([[label]])
        return self._model.prediction(label, hidden)

    def _joint_step(self, enc: torch.Tensor, pred: torch.Tensor, log_normalize: bool=False) -> torch.Tensor:
        logits = self._model.joint(enc, pred)[:, 0, 0, :]
        if not log_normalize:
            return logits

        probs = F.log_softmax(logits, dim=len(logits.shape) - 1)

        return probs

    def _get_last_symb(self, labels: List[int]) -> int:
        return self._SOS if len(labels) == 0 else labels[-1]
