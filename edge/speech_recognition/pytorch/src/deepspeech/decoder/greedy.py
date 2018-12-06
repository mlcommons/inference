from deepspeech.decoder.base import Decoder


class GreedyCTCDecoder(Decoder):
    """Selects the symbol with highest logit value at each step.

    Args:
        alphabet: See `Decoder`.
        blank_symbol: See `Decoder`.
    """

    def __init__(self, alphabet, blank_symbol):
        super().__init__(alphabet, blank_symbol)

    def decode(self, logits, logit_lens):
        _, max_indices = logits.float().max(2)

        batch_sentences = []

        for i, indices in enumerate(max_indices.t()):
            # Ignore predictions past input sequence length.
            indices = indices[:logit_lens[i]]

            no_dups, prev = [], None
            for index in indices:
                if prev is None or index != prev:
                    no_dups.append(index.item())
                    prev = index

            symbols = self._alphabet.get_symbols(no_dups)

            no_blanks = [s for s in symbols if s != self._blank_symbol]

            batch_sentences.append(''.join(no_blanks))

        return batch_sentences
