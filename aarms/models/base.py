import warnings
from collections.abc import Iterable

import numpy as np
import numba as nb
from ..utils import argpart_sort, slice_row_sparse, check_blas_config
from ..evaluation.metrics import ndcg


class BaseRecommender:
    def __repr__(self):
        raise NotImplementedError()

    def fit(self, user_item, valid_user_item=None, verbose=False):
        """"""
        raise NotImplementedError()

    def predict(self, user, user_item=None, topk=100):
        """"""
        raise NotImplementedError()

    def validate(self, user_item, valid_user_item,
                 targets=None, n_tests=2000, topk=100):
        """"""
        if targets is not None and isinstance(targets, Iterable):
            if len(targets) <= 0:
                raise ValueError('[ERROR] target length should be positive!')

        else:
            if n_tests >= user_item.shape[0]:
                targets = range(user_item.shape[0])
            else:
                targets = np.random.choice(user_item.shape[0], n_tests, False)

        scores = 0.
        count = 0.
        for i, u in enumerate(targets):
            true, rel = slice_row_sparse(valid_user_item, u)
            if len(true) == 0:
                continue

            pred = self.predict(u, user_item, topk)
            scores += ndcg(true, pred, topk)
            count += 1.

        return scores / count


class AARMSRecommender(BaseRecommender):
    """
    """
    def __init__(self):
        super().__init__()
    
    def _get_score(self, user):
        """
        """
        raise NotImplementedError()
    
    def _update_factor(self, target_entity, inputs, eps=1e-20):
        """
        """
        raise NotImplementedError()
        
    def _check_inputs(self, inputs):
        """
        """
        raise NotImplementedError()


class BaseItemFeatRecommender(BaseRecommender):
    def __init__(self):
        super().__init__()

    def fit(self, user_item, item_feat, valid_user_item=None, verbose=False):
        """"""
        raise NotImplementedError()


class FactorizationMixin:
    def __init__(self, dtype, k, init=0.001):
        """"""
        if dtype == "float32" or dtype == np.float32:
            self.f_dtype = np.float32
        elif dtype == "float64" or dtype == np.float64:
            self.f_dtype = np.float64
        else:
            raise ValueError("Only float32/float64 are supported!")

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
            u0, u1 = user_item.indptr[user], user_item.indptr[user + 1]
            if u1 > u0:
                train = user_item.indices[u0:u1]
                s[train] = -np.inf

        pred = argpart_sort(s, topk, ascending=False)
        return pred

    def _get_score(self, user):
        """"""
        raise NotImplementedError()


class MulticoreFactorizationMixin(FactorizationMixin):
    def __init__(self, dtype, k, init, n_jobs=-1):
        """
        """
        super().__init__(dtype, k, init)
        self._valid_n_jobs(n_jobs)
        
    def _valid_n_jobs(self, n_jobs):
        """
        """
        raise NotImplementedError()


class NumbaFactorizationMixin(MulticoreFactorizationMixin):
    """
    """
    def _valid_n_jobs(self, n_jobs):
        """
        """
        check_blas_config()
        if n_jobs > nb.config.NUMBA_NUM_THREADS:
            warnings.warn('n_jobs should be set lower than the number of cores! '
                          'setting it to the number...')
            self.n_jobs = nb.config.NUMBA_NUM_THREADS
        elif n_jobs == -1:
            self.n_jobs = nb.config.NUMBA_NUM_THREADS
        else:
            self.n_jobs = n_jobs