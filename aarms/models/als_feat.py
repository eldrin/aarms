import multiprocessing as mp
import numpy as np
import numba as nb
from tqdm import tqdm

from ..utils import check_blas_config
from ._als import (update_user_factor,
                   update_item_factor,
                   update_feat_factor)
from .base import (BaseItemFeatRecommender, FactorizationMixin)
from .transform import (linear_confidence,
                        log_confidence,
                        sppmi)


class ALSFeat(FactorizationMixin, BaseItemFeatRecommender):
    def __init__(self, k, init=0.001, lmbda=1, l2=0.0001, n_iters=15, dropout=0.,
                 alpha=5, eps=0.5, kappa=1, transform=linear_confidence,
                 dtype='float32', n_jobs=-1):
        """"""
        BaseRecommender.__init__(self)
        FactorizationMixin.__init__(self, dtype, k, init)

        self.lmbda = self.f_dtype(lmbda)
        self.l2 = self.f_dtype(l2)
        self.alpha = self.f_dtype(alpha)
        self.eps = self.f_dtype(eps)
        self.n_iters = n_iters
        self.dropout = dropout
        self.n_jobs = n_jobs

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
        return "ALSFeat@{:d}".format(self.k)

    def dropout_items(self, item_user):
        """"""
        if self.dropout > 0:
            n_items = item_user.shape[0]
            dropout_items = np.random.choice(
                n_items, int(n_items * self.dropout), False
            )
            for i in dropout_items:
                i0, i1 = item_user.indptr[i], item_user.indptr[i+1]
                item_user.data[i0:i1] = 0
            item_user.eliminate_zeros()
        return item_user

    def fit(self, user_item, item_feat, valid_user_item=None,
            verbose=False):
        """"""
        n_users, n_items = user_item.shape
        n_feats = item_feat.shape[1]
        self._init_embeddings(user=n_users, item=n_items, feat=n_feats)

        # preprocess data
        user_item = self._transform(user_item.astype(self.dtype))
        item_user = user_item.T.tocsr()
        item_feat = item_feat.astype(self.dtype)

        # pre-compute XX
        item_feat2 = item_feat.T @ item_feat

        # scale hyper-parameters
        lmbda = self.lmbda
        l2 = self.l2

        dsc_tmp = '[vacc={:.4f}]'
        with tqdm(total=self.n_iters, desc='[vacc=0.0000]',
                  disable=not verbose, ncols=80) as p:

            for n in range(self.n_iters):
                IU = self.dropout_items(item_user.copy())
                UI = IU.T.tocsr()

                # update user factors
                update_user_factor(
                    UI.data, UI.indices, UI.indptr,
                    self.embeddings_['user'], self.embeddings_['item'], l2
                )

                # update item factors
                update_item_factor(
                    IU.data, IU.indices, IU.indptr,
                    self.embeddings_['user'], self.embeddings_['item'],
                    item_feat, self.embeddings_['feat'], lmbda, l2
                )

                # update feat factors
                update_feat_factor(
                    self.embeddings_['item'], item_feat, item_feat2,
                    self.embeddings_['feat'], lmbda, l2
                )

                if valid_user_item is not None:
                    score = self.validate(user_item, item_feat, valid_user_item)
                    p.set_description(dsc_tmp.format(score))
                p.update(1)

    def _get_score(self, user):
        """"""
        return self.embeddings_['user'][user] @ self.embeddings_['item'].T
