import numpy as np
import numba as nb


def ndcg(actual, predicted, k=10):
    """ for binary relavance """
    if len(predicted) > k:
        predicted = predicted[:k]
    actual = set(actual)

    dcg = 0.0
    idcg = 0.0
    for i, p in enumerate(predicted):
        if p in actual:
            dcg += 1.0 / np.log2(i + 2.0)
        if i < len(actual):
            idcg += 1.0 / np.log2(i + 2.0)

    if len(actual) == 0:
        return None

    return dcg / idcg
