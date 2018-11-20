import torch
import torch.nn.functional as F
from ctcdecode import CTCBeamDecoder

from deepspeech.decoder.base import Decoder


class BeamCTCDecoder(Decoder):
    """A beam search decoder with an optional language model.

    Args:
        alphabet: See `Decoder`.
        blank_symbol: See `Decoder`.
        model_path: Path to KenLM LM. If None, LM score is not included.
        alpha: Language model weighting.
        beta: Word bonus weighting.
        cutoff_prob: Affects the list of symbols to consider when extending a
            prefix at each step. The symbols are sorted in descending order by
            probability mass. The first N symbols are considered such that
            their total probability mass is less than this value. N is also
            bounded by `cutoff_top_n`.
        cutoff_top_n: The top `cutoff_top_n` symbols with highest probability
            will be considered at each step. Note: `cutoff_prob` must be less
            then 1.0 for this to be considered else all symbols will be used.
        beam_width: Width of the beam search.
        num_processes: Number of threads for the beam search.
    """

    def __init__(self, alphabet, blank_symbol, model_path=None, alpha=1.0,
                 beta=1.0, cutoff_prob=1.0, cutoff_top_n=None, beam_width=128,
                 num_processes=4):
        super().__init__(alphabet, blank_symbol)

        cutoff_top_n = cutoff_top_n or len(alphabet)
        blank_id = alphabet.get_index(blank_symbol)

        if model_path is None:
            self._logger.warning('language model will not be used as '
                                 '`model_path` is None')
        if model_path is not None and alpha == 0.0:
            self._logger.warning("language model will not be used as it's "
                                 "weighting `alpha` is zero")

        self._decoder = CTCBeamDecoder(labels=alphabet,
                                       model_path=model_path,
                                       alpha=alpha,
                                       beta=beta,
                                       cutoff_top_n=cutoff_top_n,
                                       cutoff_prob=cutoff_prob,
                                       beam_width=beam_width,
                                       num_processes=num_processes,
                                       blank_id=blank_id)

    def decode(self, logits, logit_lens):
        """Returns a list of sentences given the logits for a batch.

        Args:
            logits: tensor of size (seq_len, batch, out_features).
            logit_lens: list of int representing the length of each sequence in
                logits.

        Returns:
            list containing batch number of sentences (strings).
        """
        if logits.dtype != torch.float:
            self._logger.debug('casting logits to single-precision')
            logits = logits.float()

        logit_lens = torch.IntTensor(logit_lens).cpu()

        probs = F.softmax(logits, dim=2)
        probs.transpose_(0, 1)   # decoder "expect[s] batch x seq x label_size"

        output, _, _, out_seq_len = self._decoder.decode(probs, logit_lens)

        batch_sentences = []
        for b, batch in enumerate(output):
            size = out_seq_len[b][0]
            indices = batch[0][:size]
            sentence = ''.join(self._alphabet.get_symbols(indices.tolist()))
            batch_sentences.append(sentence)

        return batch_sentences
