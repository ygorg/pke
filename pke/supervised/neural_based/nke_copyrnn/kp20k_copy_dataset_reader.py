import logging
from typing import List, Dict

import re
import json
import string
import numpy as np
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ArrayField, MetadataField, NamespaceSwappingField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("kp20k_copy")
class CopyNetDatasetReader(DatasetReader):
    """
    Read a tsv file containing paired sequences, and create a dataset suitable for a
    ``CopyNet`` model, or any model with a matching API.

    The expected format for each input line is: <source_sequence_string><tab><target_sequence_string>.
    An instance produced by ``CopyNetDatasetReader`` will containing at least the following fields:

    - ``source_tokens``: a ``TextField`` containing the tokenized source sentence,
       including the ``START_SYMBOL`` and ``END_SYMBOL``.
       This will result in a tensor of shape ``(batch_size, source_length)``.

    - ``source_token_ids``: an ``ArrayField`` of size ``(batch_size, trimmed_source_length)``
      that contains an ID for each token in the source sentence. Tokens that
      match at the lowercase level will share the same ID. If ``target_tokens``
      is passed as well, these IDs will also correspond to the ``target_token_ids``
      field, i.e. any tokens that match at the lowercase level in both
      the source and target sentences will share the same ID. Note that these IDs
      have no correlation with the token indices from the corresponding
      vocabulary namespaces.

    - ``source_to_target``: a ``NamespaceSwappingField`` that keeps track of the index
      of the target token that matches each token in the source sentence.
      When there is no matching target token, the OOV index is used.
      This will result in a tensor of shape ``(batch_size, trimmed_source_length)``.

    - ``metadata``: a ``MetadataField`` which contains the source tokens and
      potentially target tokens as lists of strings.

    When ``target_string`` is passed, the instance will also contain these fields:

    - ``target_tokens``: a ``TextField`` containing the tokenized target sentence,
      including the ``START_SYMBOL`` and ``END_SYMBOL``. This will result in
      a tensor of shape ``(batch_size, target_length)``.

    - ``target_token_ids``: an ``ArrayField`` of size ``(batch_size, target_length)``.
      This is calculated in the same way as ``source_token_ids``.

    See the "Notes" section below for a description of how these fields are used.

    Parameters
    ----------
    target_namespace : ``str``, required
        The vocab namespace for the targets. This needs to be passed to the dataset reader
        in order to construct the NamespaceSwappingField.
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    target_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to ``tokenizer``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    target_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define output (target side) token representations. Defaults to
        ``token_indexers``.

    Notes
    -----
    By ``source_length`` we are referring to the number of tokens in the source
    sentence including the ``START_SYMBOL`` and ``END_SYMBOL``, while
    ``trimmed_source_length`` refers to the number of tokens in the source sentence
    *excluding* the ``START_SYMBOL`` and ``END_SYMBOL``, i.e.
    ``trimmed_source_length = source_length - 2``.

    On the other hand, ``target_length`` is the number of tokens in the target sentence
    *including* the ``START_SYMBOL`` and ``END_SYMBOL``.

    In the context where there is a ``batch_size`` dimension, the above refer
    to the maximum of their individual values across the batch.

    In regards to the fields in an ``Instance`` produced by this dataset reader,
    ``source_token_ids`` and ``target_token_ids`` are primarily used during training
    to determine whether a target token is copied from a source token (or multiple matching
    source tokens), while ``source_to_target`` is primarily used during prediction
    to combine the copy scores of source tokens with the generation scores for matching
    tokens in the target namespace.
    """

    def __init__(self,
                 target_namespace: str,
                 tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 add_start_token: bool = True,
                 ref_max_token: int = None,
                 #filter_reference: bool = False,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._target_namespace = target_namespace
        self._tokenizer = tokenizer or WordTokenizer()
        self._target_tokenizer = target_tokenizer or self._tokenizer
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        if "tokens" not in self._token_indexers or \
                not isinstance(self._token_indexers["tokens"], SingleIdTokenIndexer):
            raise ConfigurationError("CopyNetDatasetReader expects 'token_indexers' to contain "
                                     "a 'single_id' token indexer called 'tokens'.")
        self._target_token_indexers: Dict[str, TokenIndexer] = {
                "tokens": SingleIdTokenIndexer(namespace=self._target_namespace)
        }
        # Does this mean we can generate word only appearing in references ?

        self._add_start_token = add_start_token
        self._ref_max_token = ref_max_token

        #self._filter_reference = filter_reference
        #if self._filter_reference:
        #    punctuation = string.punctuation.replace('-', '')
        #    self.punct_reg = re.compile(r'[{}]'.format(punctuation))

    @overrides
    def _read(self, file_path):
        """
        Read a jsonl file and pass its content to _json_to_instance
        :param file_path: Path to dataset file
        :return: A generator of Instances
        """
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                line = line.strip()
                if not line:
                    continue
                paper_json = json.loads(line)
                yield from self._json_to_instance(paper_json)

    def _json_to_instance(self, json_dict):
        title = json_dict['title']
        abstract = json_dict['abstract']
        source_string = title + ' . ' + abstract

        metadata = {'id': json_dict['id']}

        keywords = json_dict['keyword']

        if keywords is None:
            yield self.text_to_instance(
                source_string, metadata=metadata)
        else:
            keywords = keywords.split(';')
            for kw in keywords:
                instance = self.text_to_instance(
                    source_string,
                    metadata=dict(metadata, ref=kw),
                    target_string=kw)
                if instance:
                    yield instance


    @staticmethod
    def replace_digits(tokens: List[Token], digit=Token('@@digit@@')):
        return [digit if t.text.isdigit() else t for t in tokens]

    @staticmethod
    def _tokens_to_ids(tokens: List[Token]) -> List[int]:
        ids: Dict[str, int] = {}
        out: List[int] = []
        for token in tokens:
            out.append(ids.setdefault(token.text.lower(), len(ids)))
        return out

    def valid_reference(self, tokens):
        tokens = [t.text for t in tokens]

        # Filter out references with punctuation
        # or with a lot of repeated words
        # or if the words are short words
        return not self.punct_reg.search(''.join(tokens)) or \
                len(set(tokens)) / len(tokens) <= 0.5 or \
                len(tokens) / sum(map(len, tokens)) >= 0.4

    def pre_treat_text(self, text: str, tokenizer: Tokenizer, filter: bool = False):
        tokenized = tokenizer.tokenize(text)

        if filter:
            if not tokenized:
                return None

            if self._ref_max_token and len(tokenized) > self._ref_max_token:
                return None
            #if self._filter_reference and self.valid_reference(tokenized):
            #    return None

        tokenized = self.replace_digits(tokenized)
        if self._add_start_token:
            tokenized.insert(0, Token(START_SYMBOL))
        tokenized.append(Token(END_SYMBOL))
        return tokenized


    @overrides
    def text_to_instance(self, source_string: str, target_string: str = None,
                         metadata: Dict[str, any] = None) -> Instance:  # type: ignore
        """
        Turn raw source string and target string into an ``Instance``.

        Parameters
        ----------
        source_string : ``str``, required
        target_string : ``str``, optional (default = None)

        Returns
        -------
        Instance
            See the above for a description of the fields that the instance will contain.
        """
        # pylint: disable=arguments-differ
        tokenized_source = self.pre_treat_text(source_string, self._tokenizer)
        source_field = TextField(tokenized_source, self._token_indexers)
        # For each token in the source sentence, we keep track of the matching token
        # in the target sentence (which will be the OOV symbol if there is no match).
        source_to_target_field = NamespaceSwappingField(tokenized_source[1:-1], self._target_namespace)

        fields_dict = {
                "source_tokens": source_field,
                "source_to_target": source_to_target_field,
        }

        meta_fields = dict(metadata, **{"source_tokens": [x.text for x in tokenized_source[1:-1]]})

        if target_string is not None:
            tokenized_target = self.pre_treat_text(target_string, self._target_tokenizer, filter=True)
            if tokenized_target is None:
                return None
            target_field = TextField(tokenized_target, self._target_token_indexers)
            fields_dict["target_tokens"] = target_field
            meta_fields["target_tokens"] = [y.text for y in tokenized_target[1:-1]]

            source_and_target_token_ids = self._tokens_to_ids(
                tokenized_source[1:-1] + tokenized_target)

            source_token_ids = source_and_target_token_ids[:len(tokenized_source)-2]
            fields_dict["source_token_ids"] = ArrayField(np.array(source_token_ids))

            target_token_ids = source_and_target_token_ids[len(tokenized_source)-2:]
            fields_dict["target_token_ids"] = ArrayField(np.array(target_token_ids))
        else:
            source_token_ids = self._tokens_to_ids(tokenized_source[1:-1])
            fields_dict["source_token_ids"] = ArrayField(np.array(source_token_ids))



        fields_dict["metadata"] = MetadataField(meta_fields)
        print(tokenized_source)
        return Instance(fields_dict)
