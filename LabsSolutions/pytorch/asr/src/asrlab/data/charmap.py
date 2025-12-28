# coding: utf-8

# Standard imports
import logging
import sys

class CharMap(object):
    """
    Object in charge of performing the char <-> int conversion
    It holds the vocabulary and the functions required for performing
    the conversions in the two directions
    """

    _BLANK = 172  # Corresponds to '¬'
    _SOS = 182  # Corresponds to '¶', which will encode the start of string
    _EOS = 166  # Corresponds to '¦', which will encode the end of string

    def __init__(self):
        ord_chars = frozenset().union(
            range(97, 123),  # a-z
            range(48, 58),  # 0-9
            [32, 39, 44, 46],  # <space> <,> <.> <'>
            [self._SOS],  # <sos>¶
            [self._EOS],  # <eos>¦
            [10060],  # <unk> ❌
        )

        # The pad symbol is added first to guarantee it has idx 0
        self.idx2char = [chr(self._BLANK)] + [chr(i) for i in ord_chars]
        self.char2idx = {c: idx for (idx, c) in enumerate(self.idx2char)}

        self.equivalent_char = {}
        for i in range(224, 229):
            self.equivalent_char[chr(i)] = "a"
        for i in range(232, 236):
            self.equivalent_char[chr(i)] = "e"
        for i in range(236, 240):
            self.equivalent_char[chr(i)] = "i"
        for i in range(242, 247):
            self.equivalent_char[chr(i)] = "o"
        for i in range(249, 253):
            self.equivalent_char[chr(i)] = "u"
        # Remove the punctuation marks
        for c in ["!", "?", ";"]:
            self.equivalent_char[c] = "."
        for c in ["-", "…", ":"]:
            self.equivalent_char[c] = " "
        self.equivalent_char["—"] = ""
        # This 'œ' in self.equivalent_char returns False... why ?
        # self.equivalent_char['œ'] = 'oe'
        # self.equivalent_char['ç'] = 'c'
        self.equivalent_char["’"] = "'"

    @property
    def vocab_size(self):
        return len(self.idx2char)

    @property
    def eoschar(self):
        return chr(self._EOS)

    @property
    def eos(self):
        return self.char2idx[self.eoschar]

    @property
    def soschar(self):
        return chr(self._SOS)

    @property
    def blankid(self):
        return self.char2idx[chr(self._BLANK)]

    def encode(self, utterance):
        utterance = self.soschar + utterance.lower() + self.eoschar

        # Remove the accentuated characters
        utterance = [
            self.equivalent_char[c] if c in self.equivalent_char else c
            for c in utterance
        ]
        # Replace the unknown characters
        utterance = ["❌" if c not in self.char2idx else c for c in utterance]
        return [self.char2idx[c] for c in utterance]

    def decode(self, tokens):
        return "".join([self.idx2char[it] for it in tokens])

def test_charmap():
    charmap = CharMap()

    # Get the vocabulary size
    # @TEMPL@ vocab_size = 0
    # @SOL
    vocab_size = charmap.vocab_size
    # SOL@
    logging.info(f"The vocabulary contains {charmap.vocab_size} characters")

    # Experiment with the encoding of a sentence
    utterance = "Je vais m'éclater avec des RNNs !"
    # @TEMPL@ encoded = []
    encoded = charmap.encode(utterance) # @SOL@
    logging.info(f"The encoding of '{utterance}' is : {encoded}")

    # Decode the encoded sentence. Is it ok for you ?
    # @TEMPL@ decoded = ""
    decoded = charmap.decode(encoded) # @SOL@
    charmap.encode(utterance)
    logging.info(f"The decoding of the encoded sentence is '{decoded}'")

    # @SOL
    # What happens if a character is unknown ?
    utterance = "Nous sommes heureux de vous souhaiter nos meilleurs vœux pour 2021."
    encoded = charmap.encode(utterance) # @SOL@
    decoded = charmap.decode(encoded) # @SOL@
    logging.info(f"The decoding of '{utterance}' is '{decoded}'")
    # SOL@

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    test_charmap()
