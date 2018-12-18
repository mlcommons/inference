
""" To calculate the verification accuracy of LFW dataset """
# MIT License
#
# Copyright (c) 2018 Jimmy Chiang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import pdb
import numpy as np


THRESHOLD = 0.41


def _distance(embeddings1, embeddings2):
    # Distance based on cosine similarity
    dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
    norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
    similarity = dot / norm
    dist = np.arccos(similarity) / math.pi
    return dist

def _calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    acc = float(tp + tn) / dist.size
    return acc

def _lfw_evaluate(embeddings1, embeddings2, actual_issame):
    if np.sum(np.isnan(embeddings1)) > 0 or np.sum(np.isnan(embeddings2)) > 0:
        return True, 0
    else:
        dist = _distance(embeddings1, embeddings2)
        accuracy = _calculate_accuracy(THRESHOLD, dist, actual_issame)
        return False, accuracy

def lfw_metric(embeddings1, embeddings2, actual_issame):
    isNan, accuracy = _lfw_evaluate(embeddings1, embeddings2, actual_issame)
    if isNan:
        return np.nan
    else:
        return accuracy
