import numpy as np
from tqdm import tqdm

from ..metric import ndcg
from ..utils import check_blas_config
from ._als import (update_user_factor, update_item_factor, update_feat_factor)


class ALSFeat:
    def __init__(self, k, init=0.001, lmbda=1, l2=0.0001, n_iters=15,
                 alpha=5, eps=0.5, dropout=0., dtype='float32'):

        if dtype == 'float32':
            self.f_dtype = np.float32
        elif dtype == 'float64':
            self.f_dtype = np.float64
        else:
            raise ValueError('Only float32/float64 are supported!')

        self.k = k
        self.init = self.f_dtype(init)
        self.lmbda = self.f_dtype(lmbda)
        self.l2 = self.f_dtype(l2)
        self.alpha = self.f_dtype(alpha)
        self.eps = self.f_dtype(eps)
        self.dtype = dtype
        self.n_iters = n_iters
        self.dropout = dropout

        check_blas_config()

    def __repr__(self):
        return "ALSFeat@{:d}".format(self.k)

    def _init_embeddings(self):
        for key, param in self.embeddings_.items():
            self.embeddings_[key] = param.astype(self.dtype) * self.init

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
        self.embeddings_ = {
            'user': np.random.randn(n_users, self.k),
            'item': np.random.randn(n_items, self.k),
            'feat': np.random.randn(n_feats, self.k)
        }
        self._init_embeddings()

        # preprocess data
        user_item = user_item.copy().astype(self.dtype)
        user_item.data = self.f_dtype(1) + user_item.data * self.alpha

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

    def validate(self, user_item, item_feat, valid_user_item,
                 n_tests=2000, topk=100):
        """"""
        scores = []
        if n_tests >= user_item.shape[0]:
            targets = range(user_item.shape[0])
        else:
            targets = np.random.choice(user_item.shape[0], n_tests, False)
        for u in targets:
            true = valid_user_item[u].indices
            if len(true) == 0:
                continue
            train = user_item[u].indices
            s = self.embeddings_['user'][u] @ self.embeddings_['item'].T
            s[train] = -np.inf
            idx = np.argpartition(-s, kth=topk)[:topk]
            pred = idx[np.argsort(-s[idx])]
            scores.append(ndcg(true, pred, topk))
        return np.mean(scores)

