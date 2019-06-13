# Copyright 2017 Google Inc. All Rights Reserved.
# Modifications copyright (C) 2019 MLPerf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Python implementation of BLEU, smooth-BLEU and Running BLEU.

@note The most common usage case is to invoke the function compute_bleu

This module provides a Python implementation of BLEU and smooth-BLEU.
Smooth BLEU is computed following the method outlined in the paper:
Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
evaluation metrics for machine translation. COLING 2004.
"""

import collections
import math

##
# @brief Class to compute running BLEU scores
# @detail BLEU scores can be computed in a non-linear way,
# or without having access to the full translated corpus in time.
class RunningBLEUScorer:

  def __init__(self, max_order=4, smooth=False):
    self.max_order = max_order
    self.smooth = smooth
    self.reset()

  ##
  # @brief Reset all variables (none of the previus sentences will be taken into account)
  def reset(self):
    self.matches_by_order = [0] * self.max_order
    self.possible_matches_by_order = [0] * self.max_order
    self.reference_length = 0
    self.translation_length = 0

  ##
  # @brief Add a single sentence
  # @param reference list of words for a reference sentence
  # @param translation list of words for its corresponding translated sentence
  # @post Updates internal structures to take this sentence's translation 
  # result into account in final BLEU score
  def add_sentence(self, reference, translation):
    self.add_sentence_with_multiple_refs([reference], translation)

  ##
  # @brief Add a single reference, with potentially multiple references
  # @param reference list of list of words for a reference sentence
  # @note That we could have multiple sentences serving as a reference
  # @param translation (single) list of words for its corresponding translated sentence
  # @post Updates internal structures to take this sentence's translation 
  # result into account in final BLEU score
  def add_sentence_with_multiple_refs(self, references, translation):
    self.reference_length += min(len(r) for r in references)
    self.translation_length += len(translation)

    merged_ref_ngram_counts = collections.Counter()
    for reference in references:
      merged_ref_ngram_counts |= self._get_ngrams(reference)

    translation_ngram_counts = self._get_ngrams(translation)
    
    new_matches_by_order, new_possible_matches_by_order = self._get_ngram_match_values(merged_ref_ngram_counts, translation_ngram_counts, len(translation))

    for i in range(self.max_order):
      self.matches_by_order[i] += new_matches_by_order[i]
      self.possible_matches_by_order[i] += new_possible_matches_by_order[i]

  ##
  # @brief Calculate final BLEU score
  def calc_BLEU_score(self):
    precisions = [0] * self.max_order
    for i in range(0, self.max_order):
      if self.smooth:
        precisions[i] = ((self.matches_by_order[i] + 1.) /
                         (self.possible_matches_by_order[i] + 1.))
      else:
        if self.possible_matches_by_order[i] > 0:
          precisions[i] = (float(self.matches_by_order[i]) /
                           self.possible_matches_by_order[i])
        else:
          precisions[i] = 0.0

    if min(precisions) > 0:
      p_log_sum = sum((1. / self.max_order) * math.log(p) for p in precisions)
      geo_mean = math.exp(p_log_sum)
    else:
      geo_mean = 0

    ratio = float(self.translation_length) / self.reference_length

    if ratio > 1.0:
      bp = 1.
    else:
      bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, self.translation_length, self.reference_length)

  ##
  # @brief Internal function to compute matching percentages for different order ngrams
  def _get_ngram_match_values(self, ref_ngram_counts, translation_ngram_counts, translation_length):
    new_matches_by_order = [0] * self.max_order
    new_possible_matches_by_order = [0] * self.max_order

    overlap = translation_ngram_counts & ref_ngram_counts
    for ngram in overlap:
      new_matches_by_order[len(ngram)-1] += overlap[ngram]
    for order in range(1, self.max_order+1):
      possible_matches = translation_length - order + 1
      new_possible_matches_by_order[order-1] = max(0, possible_matches)

    return (new_matches_by_order, new_possible_matches_by_order)

  def _get_ngrams(self, segment):
    """Internal function to extract all n-grams upto a given maximum order from an input segment.

    Args:
      segment: text segment from which n-grams will be extracted.

    Returns:
      The Counter containing all n-grams upto max_order in segment
      with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, self.max_order + 1):
      for i in range(0, len(segment) - order + 1):
        ngram = tuple(segment[i:i+order])
        ngram_counts[ngram] += 1
    return ngram_counts

def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
  """Computes BLEU score of translated segments against one or more references.
    This is the most common usage when calculating BLEU scores.

  Args:
    reference_corpus: list of lists of references for each translation. Each
        reference should be tokenized into a list of tokens.
        reference_corpus[i][j][k] represents the k'th word of the i'th sentence
        for the j'th reference text
    translation_corpus: list of translated sentences to score. Each sentence
        should be tokenized into a list of tokens.
        translation_corpus[i][j] represents the j'th word for the i'th sentence
    max_order: Maximum n-gram order to use when computing BLEU score.
    smooth: Whether or not to apply Lin et al. 2004 smoothing.

  Returns:
    3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
    precisions and brevity penalty.
  """
  runningBLEU = RunningBLEUScorer(max_order=max_order, smooth=smooth)


  for (references, translation) in zip(reference_corpus,
                                       translation_corpus):
    runningBLEU.add_sentence_with_multiple_refs(references, translation)
 
  return runningBLEU.calc_BLEU_score()