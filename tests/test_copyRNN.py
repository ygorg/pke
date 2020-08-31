#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import mock
import pytest

import pke

text = "A feedback vertex set of 2-degenerate graphs . A feedback vertex set of "\
       "a graph G is a set S  of its vertices such that the subgraph induced by "\
       "V(G)?S V ( G ) ? S is a forest. The cardinality of a minimum feedback "\
       "vertex set of G  is denoted by ?(G) ? ( G ) . A graph G is 2-degenerate"\
       "  if each subgraph G? G ? of G has a vertex v  such that dG?(v)?2 d G ? "\
       "( v ) ? 2 . In this paper, we prove that ?(G)?2n/5 ? ( G ) ? 2 n / 5 "\
       "for any 2-degenerate n-vertex graph G and moreover, we show that this "\
       "bound is tight. As a consequence, we derive a polynomial time algorithm"\
       ", which for a given 2-degenerate n -vertex graph returns its feedback "\
       "vertex set of cardinality at most 2n/5 2 n / 5 ."
expected_output = ["feedback vertex set", "2-degenerate graphs", "vertex set", "graph", "feedback"]

def test_CopyRNN_candidate_weighting():
    """Test SingleRank candidate weighting method."""
    extractor = pke.supervised.CopyRNN()
    extractor.load_document(input=text)
    extractor.candidate_selection()
    extractor.candidate_weighting()
    keyphrases = [k for k, s in extractor.get_n_best(n=5)]
    print(keyphrases)
    print(expected_output[:len(keyphrases)])
    assert keyphrases == expected_output[:len(keyphrases)]


def test_import_CopyRNN_noallennlp():
    # Without sent2vec this should not throw an error

    # Make allennlp unavailable
    with mock.patch.dict(sys.modules, {'allennlp.models.archival': None,
                                       'allennlp.predictors': None}):
        with pytest.raises(ImportError):
            extractor = pke.supervised.CopyRNN()
            extractor.load_document(input=text)
            extractor.candidate_selection()
            extractor.candidate_weighting()


test_CopyRNN_candidate_weighting()
test_import_CopyRNN_noallennlp()