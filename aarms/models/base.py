import numpy as np
from ..utils import argpart_sort, slice_row_sparse
from ..metric import ndcg


class BaseRecommender:
    def __repr__(self):
        raise NotImplementedError()

    def fit(self, user_item, valid_user_item=None, verbose=False):
        """"""
        raise NotImplementedError()

    def predict(self, user, user_item=None, topk=100):
        """"""
        raise NotImplementedError()

    def validate(self, user_item, valid_user_item, n_tests=2000, topk=100):
        """"""
        scores = np.zeros((n_tests,))
        if n_tests >= user_item.shape[0]:
            targets = range(user_item.shape[0])
        else:
            targets = np.random.choice(user_item.shape[0], n_tests, False)

        for i, u in enumerate(targets):
            true, rel = slice_row_sparse(valid_user_item, u)
            if len(true) == 0: continue

            pred = self.predict(u, user_item, topk)
            scores[i] = ndcg(true, pred, topk)

        return np.mean(scores)


class BaseItemFeatRecommender(BaseRecommender):
    def __init__(self):
        super().__init__()

    def fit(self, user_item, item_feat, valid_user_item=None, verbose=False):
        """"""
        raise NotImplementedError()


class FactorizationMixin:
    def __init__(self, dtype, k, init=0.001):
        """"""
        if dtype == 'float32':
            self.f_dtype = np.float32
        elif dtype == 'float64':
            self.f_dtype = np.float64
        else:
            raise ValueError('Only float32/float64 are supported!')

        self.dtype = dtype
        self.k = k
        self.init = self.f_dtype(init)

    def _init_embeddings(self, **entity_sizes):
        """"""
        self.embeddings_ = dict()
        for name, size in entity_sizes.items():
            self.embeddings_[name] = (
                np.random.randn(size, self.k).astype(self.dtype) * self.init
            )

    def predict(self, user, user_item=None, topk=100):
        """"""
        s = self._get_score(user)
        if user_item is not None:
            u0, u1 = user_item.indptr[user], user_item.indptr[user+1]
            if u1 > u0:
                train = user_item.indices[u0:u1]
                s[train] = -np.inf

        pred = argpart_sort(s, topk, ascending=False)
        return pred

    def _get_score(self, user):
        """"""
        raise NotImplementedError()
