import numpy as np

from .base import PerUserMetric
from ._metrics import (ndcg, ndcg_bin, apk, precision, recall)


class NDCG(PerUserMetric):
    """
    """
    def __init__(self, topk, binarize=True, pow2=False):
        """
        """
        name = f'nDCG@{topk:d}'
        if binarize:
            name = 'binary_' + name

        super().__init__(name, topk)

        self.binarize = binarize
        self.pow2 = pow2

    def _score(self, true, pred, true_rel=None):
        """
        """
        if self.binarize:
            return ndcg_bin(true, pred, self.topk)
        else:
            if true_rel is None:
                raise ValueError('[ERROR] for relevance nDCG computation '
                                 '`true_rel` should be given!')
            # to make sure the dtype
            return ndcg(true, true_rel, pred, self.topk, self.pow2)


class Precision(PerUserMetric):
    """
    """
    def __init__(self, topk):
        """
        """
        super().__init__(f'Precision@{topk:d}', topk)

    def _score(self, true, pred, true_rel=None):
        """
        """
        return precision(true, pred, self.topk)


class Recall(PerUserMetric):
    """
    """
    def __init__(self, topk):
        """
        """
        super().__init__(f'Recall@{topk:d}', topk)

    def _score(self, true, pred, true_rel=None):
        """
        """
        return recall(true, pred, self.topk)


class AveragePrecision(PerUserMetric):
    """
    """
    def __init__(self, topk):
        """
        """
        super().__init__(f'AP@{topk:d}', topk)

    def _score(self, true, pred, true_rel=None):
        """
        """
        return apk(true, pred, self.topk)
