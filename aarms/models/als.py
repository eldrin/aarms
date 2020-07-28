import multiprocessing as mp
from functools import partial

import numpy as np
import numba as nb

from tqdm import tqdm

from ..utils import check_blas_config
from ._als import update_user_factor
from .base import (BaseRecommender, FactorizationMixin)
from .transform import (linear_confidence,
                        log_confidence,
                        sppmi)


class ALS(FactorizationMixin, BaseRecommender):
    def __init__(self, k, init=0.001, l2=0.0001, n_iters=15,
                 alpha=5, eps=0.5, kappa=1, transform=linear_confidence,
                 dtype='float32', solver='cg', cg_steps=3, n_jobs=-1):
        """"""
        BaseRecommender.__init__(self)
        FactorizationMixin.__init__(self, dtype, k, init)

        self.l2 = self.f_dtype(l2)
        self.alpha = self.f_dtype(alpha)
        self.eps = self.f_dtype(eps)
        self.kappa = self.f_dtype(kappa)
        self.n_iters = n_iters
        self.n_jobs = n_jobs
        self.solver = solver
        self.cg_steps = cg_steps

        check_blas_config()
        if n_jobs == -1:
            # nb.set_num_threads(mp.cpu_count())
            nb.config.NUMBA_NUM_THREADS = mp.cpu_count()
        else:
            # nb.set_num_threads(self.n_jobs)
            nb.config.NUMBA_NUM_THREADS = n_jobs

        # set transform function
        if transform == linear_confidence:
            self._transform = partial(transform, alpha=self.alpha)
        elif transform == log_confidence:
            self._transform = partial(transform, alpha=self.alpha, eps=self.eps)
        elif transform == sppmi:
            self._transform = partial(sppmi, k=self.kappa)
        else:
            raise ValueError('[ERROR] not supported transform given!')

    def __repr__(self):
        return "ALS@{:d}".format(self.k)

    def fit(self, user_item, valid_user_item=None, verbose=False):
        """"""
        n_users, n_items = user_item.shape
        self._init_embeddings(user=n_users, item=n_items)

        # preprocess data
        user_item = self._transform(user_item.astype(self.dtype))
        item_user = user_item.T.tocsr()

        # set the number of threads for training
        dsc_tmp = '[vacc={:.4f}]'
        with tqdm(total=self.n_iters, desc='[vacc=0.0000]',
                  disable=not verbose, ncols=80) as p:

            for n in range(self.n_iters):
                # update user factors
                update_user_factor(
                    user_item.data, user_item.indices, user_item.indptr,
                    self.embeddings_['user'], self.embeddings_['item'], self.l2,
                    solver=self.solver, cg_steps=self.cg_steps
                )

                # update item factors
                update_user_factor(
                    item_user.data, item_user.indices, item_user.indptr,
                    self.embeddings_['item'], self.embeddings_['user'], self.l2,
                    solver=self.solver, cg_steps=self.cg_steps
                )

                if valid_user_item is not None:
                    score = self.validate(user_item, valid_user_item)
                    p.set_description(dsc_tmp.format(score))
                p.update(1)

    def _get_score(self, user):
        """"""
        return self.embeddings_['user'][user] @ self.embeddings_['item'].T
