# -*- coding: utf-8 -*-
# Author: Florian Boudin
# Date: 11-11-2018

"""
Implementation of the CopyRNN model for automatic keyphrase extraction.
"""

from __future__ import absolute_import
from __future__ import print_function

import os
import logging

from nltk.stem.snowball import SnowballStemmer

from pke import LoadFile
from pke.base import ISO_to_language


class CopyRNN(LoadFile):

    _model_path = None
    _model_predictor = None

    def __init__(self, model_path=None, cuda=None, beam_size=None):
        """Constructor for CopyRNN model.

        :param model_path: Path to pretrained CopyRNN model, defaults to 'pke/models/copy_rnn.tar.gz'
        :type model_path: str, optional
        :param cuda: GPU id (usually 0 or 1, -1 for not using cuda) if available, defaults to -1
        :type cuda: int, optional
        :param beam_size: Max number of keyphrase to output, lower is faster but less accurate, defaults to None
        :type beam_size: int, optional
        :raises: ImportError, FileNotFoundError
        """
        try:
            # See https://github.com/allenai/allennlp
            from allennlp.predictors import Predictor
            from allennlp.models.archival import load_archive
        except ImportError as e:
            raise ImportError(
                'Module allennlp was not found. It is required by neural models. '\
                'Please install using : `pip install allennlp`')

        # Import to register allennlp objects
        import pke.supervised.neural_based.nke_copyrnn

        super(CopyRNN, self).__init__()

        # Deal with default path to model
        if model_path is None:
            model_name = 'copy_rnn.tar.gz'
            model_path = os.path.join(self._models, model_name)

        if cuda is None:
            cuda = -1

        # Check if path to model exists
        if not os.path.exists(model_path):
            logging.error('Could not find {}'.format(model_path))
            logging.error('Please download pretrained model from '
                          '.')  # TODO
            logging.error('And place it in {}.'.format(self._models))
            logging.error('Or provide a model path.')
            raise FileNotFoundError('No such file or directory: {}'.format(model_path))

        # Statically loading embedding path if necessary
        if CopyRNN._model_path is None or CopyRNN._model_path != model_path:
            logging.info('Loading allennlp model')
            archive = load_archive(
                archive_file=model_path, cuda_device=cuda, overrides={})
            CopyRNN._model_predictor = Predictor.from_archive(
                archive, predictor_name='kpextractor_predictor')
            CopyRNN._model_path = model_path
            logging.info('Done loading sent2vec model')

        # Retrieving loaded embedding model
        self._model_predictor = CopyRNN._model_predictor
        self._model_path = CopyRNN._model_path

    def load_document(self, inpu):
        self.doc = inpu
        super().load_document(inpu)
        doc = ''
        for s in self.sentences:
            for w, o in zip(s.words, s.meta['char_offsets']):
                doc += ' ' * (o[0] - len(doc)) + w
        return doc

    def candidate_selection(self):
        pass

    def candidate_weighting(self):
        # Recreate the document for inputting in the model.
        text = self.doc
        doc = ''
        for s in self.sentences:
            for w, o in s.words, s.meta['char_offsets']:
                doc += ' ' * (o[0] - len(doc)) + w

        assert text == doc

        if self.language == 'en':
            # create a new instance of a porter stemmer
            stemmer = SnowballStemmer("porter")
        else:
            # create a new instance of a porter stemmer
            stemmer = SnowballStemmer(ISO_to_language[self.language],
                                      ignore_stopwords=True)

        res = self._model_predictor.predict_json({'source_string': text, 'id': 0})
        for tokens, prob in zip(res['predicted_tokens'], res['predicted_log_probs']):
            keyphrase = ' '.join(tokens)
            if '@@' in keyphrase:
                continue
            stems = [stemmer.stem(w) for w in tokens]
            self.add_candidate(tokens, stems, None, None, None)
            stems = ' '.join(stems)
            self.weights[stems] = prob
