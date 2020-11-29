import numpy as np
import numba as nb


@nb.njit(nb.f8(nb.i4[:], nb.i4[:], nb.i8), cache=True)
def ndcg(actual, predicted, k=10):
    """ for binary relavance 
    """
    m = len(actual)   
    if m == 0:
        return -1
    n = len(predicted)

    dcg = 0.0
    idcg = 0.0
    actual = set(actual)
    for i in range(k):
        p = predicted[i]
        x = np.log2(i + 2.)**-1
        if p in actual:
            dcg += x
        if i < m:
            idcg += x

    return dcg / idcg


@nb.njit(nb.f8(nb.i4[:], nb.i4[:], nb.i8), cache=True)
def precision(actual, predicted, k=10):
    """
    """
    m = len(actual)
    if m == 0:
        return -1
    
    actual = set(actual)
    hit = 0.
    for i in range(k):
        p = predicted[i]
        if p in actual:
            hit += 1
    
    return hit / k


@nb.njit(nb.f8(nb.i4[:], nb.i4[:], nb.i8), cache=True)
def recall(actual, predicted, k=10):
    """
    """
    m = len(actual)
    if m == 0:
        return -1
    
    actual = set(actual)
    hit = 0.
    for i in range(k):
        p = predicted[i]
        if p in actual:
            hit += 1
    
    return hit / m