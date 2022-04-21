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

from typing import List, Optional, Tuple

import torch

import torch.nn.functional as F
from model_separable_rnnt import label_collate

def _update_batch(dim, max_lens, max_symbols, _SOS, blankness, blank_vec, x, hidden_prime, hidden, label_col, label_row, label_tensor, symbols_added, time_idxs, f, k):
    symbols_added *= blankness.logical_not()
    tmp_blank_vec = blank_vec.logical_or(blankness)

    # If for sample blankid already encountered, then stop
    # update hidden values until input from next time step.
    # So we would mix value of hidden and hidden_prime together,
    # keep values in hidden where blank_vec[i] is true
    if hidden == None:
        hidden = [torch.zeros_like(hidden_prime[0]), torch.zeros_like(hidden_prime[1])]

    not_blank = tmp_blank_vec.eq(0)
    
    idx = (not_blank).nonzero(as_tuple=True)[0]

    hidden[0][:, idx, :] = hidden_prime[0][:, idx, :]
    hidden[1][:, idx, :] = hidden_prime[1][:, idx, :]

    label_col += not_blank
    label_tensor.index_put_([label_row, label_col], (k-_SOS)*not_blank, accumulate=True)

    symbols_added += not_blank

    need_add = symbols_added.ge(max_symbols)

    time_idxs += need_add
    blankness.logical_or_(need_add)
    symbols_added *= symbols_added.lt(max_symbols)

    # update f if necessary
    # if at least one id in blankness is blank them time_idx is updated
    # and we need to update f accordingly
    if blankness.nonzero().size(0) > 0:
        fetch_time_idxs = time_idxs.min(max_lens)
        # select tensor along second dim of x
        # implement something like --> f = x[:, fetch_time_idxs, :].unsqueeze(1)
        # for example, if all elements in fetch_time_idxs = n, then
        # this is equivelent to f = x[:, n, :].unsqueeze(1)
        f = x[list(range(x.size(0))), fetch_time_idxs, :].unsqueeze(1)

    return hidden, label_tensor, label_col, f, time_idxs, symbols_added

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

        logits, logits_lens = self._model.encoder(x, out_lens)

        output: List[List[int]] = []
        output = self._greedy_decode_batch(logits, logits_lens)
        return output

    def _greedy_decode_batch(self, x: torch.Tensor, out_lens: torch.Tensor) -> List[int]:
        batch_size = x.size(0)
        hidden = None
        max_len = out_lens.max()
        max_lens = torch.tensor([max_len-1 for i in range(batch_size)], dtype=torch.int64)
        # pos 0 of label_tensor is set to _SOS to simplify computation
        # real label start from pos 1
        label_tensor = torch.tensor([self._SOS]).repeat(batch_size, max_len*self._max_symbols_per_step)
        # (row, col) of current labels end
        label_row = torch.tensor([i for i in range(batch_size)])
        label_col = torch.tensor([0 for i in range(batch_size)])
        # this list will be used to return labels to caller
        label_copy = [0 for i in range(batch_size)]
        # initially time_idx is 0 for all input
        # then advance time_idx for each 'track' when needed and update f
        f = x[:, 0, :].unsqueeze(1)
        time_idxs = torch.tensor([0 for i in range(batch_size)], dtype=torch.int64)

        not_blank = True
        blank_vec = torch.tensor([False for i in range(batch_size)])
        symbols_added = torch.tensor([0 for i in range(batch_size)], dtype=torch.int64)

        while True:
            g, hidden_prime = self._pred_step_batch(
                label_tensor.gather(1, label_col.unsqueeze(1)),
                hidden,
            )
            logp = self._joint_step_batch(f, g, log_normalize=False)

            # get index k, of max prob
            v, k = logp.max(1)

            # if any of the output is blank, pull in the next time_idx for next f
            # tmp_blank_vec is the vect used to mix new hidden state with previous hidden state
            # blank_vec is the baseline of blank_vec, it turns to blank only when run out of time_idx
            blankness = k.eq(self._blank_id)
            time_idxs = time_idxs + blankness
            # it doesn't matter if blank_vec is update now or later,
            # tmp_blank_vec always get correct value for this round
            blank_vec = time_idxs.ge(out_lens)

            if blank_vec.nonzero().size(0) == batch_size:
                # all time_idxs processed, stop
                break
            else:
                hidden, label_tensor, label_col, f, time_idxs, symbols_added = _update_batch(
                    f.size()[2], 
                    max_lens,
                    self._max_symbols_per_step,
                    self._SOS,
                    blankness,
                    blank_vec,
                    x,
                    hidden_prime,
                    hidden,
                    label_col,
                    label_row,
                    label_tensor,
                    symbols_added,
                    time_idxs,
                    f,
                    k)

        for i in range(batch_size):
            label_copy[i]=label_tensor[i][1:label_col[i]+1].tolist()
        return label_copy

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
    
    def _pred_step_batch(self, label, hidden):
        # not really need this line, _blank_id is the last id of dict
        #label = label - label.gt(self._blank_id).int()
        result = self._model.prediction(label, hidden)
        return result

    def _pred_step(self, label: int, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if label == self._SOS:
            return self._model.prediction(None, hidden)
        if label > self._blank_id:
            label -= 1
        label = torch.tensor([[label]], dtype=torch.int64)
        return self._model.prediction(label, hidden)

    def _joint_step_batch(self, enc, pred, log_normalize=False):
        logits = self._model.joint(enc, pred)
        logits = logits[:, 0, 0, :]
        if not log_normalize:
            return logits

        probs = F.log_softmax(logits, dim=len(logits.shape) - 1)

        return probs

    def _joint_step(self, enc: torch.Tensor, pred: torch.Tensor, log_normalize: bool=False) -> torch.Tensor:
        logits = self._model.joint(enc, pred)[:, 0, 0, :]
        if not log_normalize:
            return logits

        probs = F.log_softmax(logits, dim=len(logits.shape) - 1)

        return probs

    def _get_last_symb(self, labels: List[int]) -> int:
        return self._SOS if len(labels) == 0 else labels[-1]
