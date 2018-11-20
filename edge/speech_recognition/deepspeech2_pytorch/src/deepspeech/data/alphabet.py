class Alphabet:
    """An alphabet for a language.

    Args:
        symbols (sequence of str): Sequence of symbols in the alphabet. Each
            symbol will be assigned, in iteration order, an index (int)
            starting from 0.

    Raises:
        ValueError: Duplicate symbol in symbols.

    Attributes:
        symbols: The original sequence of symbols.
    """

    def __init__(self, symbols):
        if len(set(symbols)) != len(symbols):
            raise ValueError('Duplicate symbol in symbols.')

        self.symbols = symbols
        self._index_map = dict(enumerate(symbols))
        self._symbol_map = {letter: i for i, letter in self._index_map.items()}

    def __repr__(self):
        return self.__class__.__name__ + ('(symbols=%r)' % self.symbols)

    def __len__(self):
        """Returns the number of symbols in the alphabet."""
        return len(self.symbols)

    def __getitem__(self, index):
        symbol = self.get_symbol(index)
        if symbol is None:
            raise IndexError('Index %d is out of range')
        return symbol

    def get_symbol(self, index):
        """Returns the symbol for an index or None if index has no symbol."""
        return self._index_map.get(index)

    def get_index(self, symbol):
        """Returns the index for a symbol or None if symbol not in Alphabet."""
        return self._symbol_map.get(symbol)

    def get_symbols(self, indices):
        """Maps each index in a sequence of indices to it's symbol.

        Args:
            indices: A sequence of indices (int).

        Returns:
            A list of symbols (str). Indices in the sequence that do not have a
            corresponding symbol will be ignored. This means len(returned list)
            may be shorted than len(indices).
        """
        symbols = [self.get_symbol(index) for index in indices]
        return list(filter(lambda x: x is not None, symbols))

    def get_indices(self, sentence):
        """Maps each symbol in a sentence to it's index.

        Args:
            sentence: A sequence of symbols.

        Returns:
            A list of indices (int). Symbols in the sentence that are not in
            the alphabet will be ignored. This means len(returned list) may be
            shorter than len(sentence).
        """
        indices = [self.get_index(symbol) for symbol in sentence]
        return list(filter(lambda x: x is not None, indices))
