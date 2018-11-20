import torch

import warpctc

from deepspeech.logging import LoggerMixin


class CTCLoss(torch.nn.Module, LoggerMixin):
    """Connectionist Temporal Classification (CTC) Loss.

    This computes the forward and backward pass in single-precision using a
    single call to an optimised kernel.

    Args:
        blank_index (int): The index of the blank symbol in logits.
        size_average (bool, optional): Normalise the loss by the batch size.
        length_average (bool, optional): Normalise the loss by the total number
            of input steps (i.e. `sum(logit_lens)`).
    """

    def __init__(self, blank_index, size_average=False, length_average=False):
        super().__init__()
        self._ctc_loss = warpctc.CTCLoss(reduce=True,
                                         size_average=size_average,
                                         length_average=length_average,
                                         blank_label=blank_index)

    def forward(self, logits, labels, logit_lens):
        """Returns the CTC loss.

        Args:
            logits: A torch.Tensor of logits with shape
                `(seq_len, batch, alphabet_len)`. These will be cast to
                `float32`.
            labels: A list of torch.IntTensor containing the target labels for
                each sample in the batch.
            logit_lens: A torch.IntTensor containing the length of the input
                sequences.
        """
        if logits.dtype != torch.float:
            self._logger.debug('casting logits to single-precision')
            logits = logits.float()

        if not logits.is_cuda:
            self._logger.debug('using cpu ctc loss')

        label_lens = torch.IntTensor([len(label) for label in labels])
        labels = torch.cat(labels)

        loss = self._ctc_loss(logits, logit_lens, labels, label_lens)

        return loss
