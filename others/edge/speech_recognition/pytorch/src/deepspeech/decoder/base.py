from deepspeech.logging import LoggerMixin


class Decoder(LoggerMixin):
    """Decoder base class.

    Args:
        alphabet: An Alphabet object.
        blank_symbol: The symbol in `alphabet` to use as the blank during CTC
            decoding.
    """

    def __init__(self, alphabet, blank_symbol):
        self._alphabet = alphabet
        self._blank_symbol = blank_symbol

    def decode(self, logits, logit_lens):
        """Returns a list of sentences given the logits for a batch.

        Args:
            logits: tensor of size (seq_len, batch, out_features).
            logit_lens: list of int representing the length of each sequence in
                logits.

        Returns:
            list containing batch number of sentences (strings).
        """
        raise NotImplementedError
