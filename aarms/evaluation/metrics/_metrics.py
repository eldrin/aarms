import numpy as np
import numba as nb
from numba.typed import Dict


@nb.njit(
    [
        nb.f8(nb.i4[:], nb.f8[:], nb.i4[:], nb.i8, nb.b1),
        nb.f8(nb.i8[:], nb.f8[:], nb.i8[:], nb.i8, nb.b1),
    ],
    cache=True)
def ndcg(actual, relevance, predicted, k=10, pow2=False):
    """ normalized Discounted Cumulative Gain

    TODO: computational cost for this function can be improved significantly.

    Inputs:
        actual (array of int32): contains true integer indices
        relevance (array of float64): contains true relevance with same order of `actual`
        predicted (array of int32): contains predicted indces of items
        k (int64): cutoff for top-k metric computation
        pow2 (bool): flag whether relevance computed in power of 2 or linearly

    Outputs:
        float64: ndcg computed
    """
    m = len(actual)
    if m == 0:
        return -1
    idx = np.argsort(-relevance)
    actual = actual[idx]
    relevance = relevance[idx]

    dcg = 0.0
    idcg = 0.0
    for i in range(k):
        p = predicted[i]
        x = np.log2(i + 2.)**-1
        j = np.where(actual == p)[0]
        if len(j) > 0:
            rel = relevance[j[0]]
            if pow2:
                rel = 2.**rel - 1.
            dcg += rel * x
        if i < m:
            rel = relevance[i]
            if pow2:
                rel = 2.**rel - 1.
            idcg += rel * x

    return dcg / idcg


@nb.njit(
    [
        nb.f8(nb.i4[:], nb.i4[:], nb.i8),
        nb.f8(nb.i8[:], nb.i8[:], nb.i8)
    ],
    cache=True)
def ndcg_bin(actual, predicted, k=10):
    """ for binary relavance
    """
    m = len(actual)
    if m == 0:
        return -1

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


@nb.njit(
    [
        nb.f8(nb.i4[:], nb.i4[:], nb.i8),
        nb.f8(nb.i8[:], nb.i8[:], nb.i8)
    ],
    cache=True)
def apk(actual, predicted, k=10):
    """
    """
    m = len(actual)
    if m == 0:
        return -1

    actual = set(actual)
    ap = 0.
    hit = 0.
    Nk = min(m, k)
    for i in range(k):
        p = predicted[i]
        if p in actual:
            hit += 1.
            ap += hit / (i + 1.)
    return ap / Nk


@nb.njit(
    [
        nb.f8(nb.i4[:], nb.i4[:], nb.i8),
        nb.f8(nb.i8[:], nb.i8[:], nb.i8)
    ],
    cache=True)
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


@nb.njit(
    [
        nb.f8(nb.i4[:], nb.i4[:], nb.i8),
        nb.f8(nb.i8[:], nb.i8[:], nb.i8)
    ],
    cache=True)
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
