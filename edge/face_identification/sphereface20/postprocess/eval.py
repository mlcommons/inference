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
