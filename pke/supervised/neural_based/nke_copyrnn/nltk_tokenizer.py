from typing import List

from overrides import overrides

import nltk

from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.word_splitter import WordSplitter


@WordSplitter.register('nltk')
class NltkWordSplitter(WordSplitter):

    @overrides
    def split_words(self, sentence: str) -> List[Token]:
        """
        Splits ``sentence`` into a list of :class:`Token` objects.
        """
        return [Token(t) for t in nltk.word_tokenize(sentence, preserve_line=True)]
