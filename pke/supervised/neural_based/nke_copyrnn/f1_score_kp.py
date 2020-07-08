from typing import Optional, List, Union

import torch

from allennlp.training.metrics.metric import Metric
from allennlp.data.tokenizers.word_stemmer import WordStemmer, PorterStemmer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter


@Metric.register("f1_kp")
class F1ScoreKP(Metric):
    """
    Computes Precision, Recall and F1 with respect to a given ``positive_label``.
    For example, for a BIO tagging scheme, you would pass the classification index of
    the tag you are interested in, resulting in the Precision, Recall and F1 score being
    calculated for this tag only.

    Adapted from allennlp.training.metrics.F1Measure
    """
    def __init__(self, rank: int = None, stemmer: WordStemmer = PorterStemmer()) -> None:
        self._rank = rank
        if stemmer:
            self._stemmer = stemmer.stem_word
            self._tokenizer = JustSpacesWordSplitter().split_words
        else:
            self._stemmer = None
        self._p = []
        self._r = []
        self._f = []
        self._nb_doc = 0

    def __call__(self,
                 predictions: List[List[str]],
                 gold_labels: List[List[str]],
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        # predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)
        self._nb_doc += len(predictions)

        for pred, ref in zip(predictions, gold_labels):
            # Keeping len_ref to ensure converting to set does not change the number of reference
            len_ref = len(ref)

            if self._rank and len(pred) > self._rank:
                pred = pred[:self._rank]

            if self._stemmer:
                pred = [' '.join([self._stemmer(t).text for t in self._tokenizer(p)]) for p in pred]
                ref = [' '.join([self._stemmer(t).text for t in self._tokenizer(r)]) for r in ref]

            nb_match = 0
            ref = set(map(str.lower, ref))
            for cand in pred:
                if cand.lower() in ref:
                    nb_match += 1

            p = nb_match / len(pred) if len(pred) > 0 else 0
            r = nb_match / len_ref if len_ref > 0 else 0
            f = (2 * p * r) / (p + r) if p + r > 0 else 0

            self._p.append(p)
            self._r.append(r)
            self._f.append(f)

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        precision : float
        recall : float
        f1-measure : float
        """
        precision = 0
        recall = 0
        f1_measure = 0
        if self._nb_doc > 0:
            precision = sum(self._p) / self._nb_doc * 100
            recall = sum(self._r) / self._nb_doc * 100
            f1_measure = sum(self._f) / self._nb_doc * 100
        if reset:
            self.reset()
        suffix = ''
        if self._rank:
            suffix = '@{}'.format(self._rank)
        return {'metrics/P' + suffix: precision,
                'metrics/R' + suffix: recall,
                'metrics/F' + suffix: f1_measure}

    def reset(self):
        self._p = []
        self._r = []
        self._f = []
        self._nb_doc = 0
