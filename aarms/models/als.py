import numpy as np

from tqdm import tqdm

from ..metric import ndcg
from ._als import update_user_factor


class ALS:
    def __init__(self, k, init=0.001, l2=0.0001, n_iters=15,
                 alpha=5, eps=0.5, dtype='float32'):

        if dtype == 'float32':
            self.f_dtype = np.float32
        elif dtype == 'float64':
            self.f_dtype = np.float64
        else:
            raise ValueError('Only float32/float64 are supported!')

        self.k = k
        self.init = self.f_dtype(init)
        self.l2 = self.f_dtype(l2)
        self.alpha = self.f_dtype(alpha)
        self.eps = self.f_dtype(eps)
        self.dtype = dtype
        self.n_iters = n_iters

        check_blas_config()

    def __repr__(self):
        return "ALS@{:d}".format(self.k)

    def _init_embeddings(self):
        for key, param in self.embeddings_.items():
            self.embeddings_[key] = param.astype(self.dtype) * self.init

    def fit(self, user_item, valid_user_item=None, verbose=False):
        """"""
        n_users, n_items = user_item.shape
        self.embeddings_ = {
            'user': np.random.randn(n_users, self.k),
            'item': np.random.randn(n_items, self.k),
        }
        self._init_embeddings()

        # preprocess data
        user_item = user_item.copy().astype(self.dtype)
        user_item.data = self.f_dtype(1) + user_item.data * self.alpha
        item_user = user_item.T.tocsr()

        dsc_tmp = '[vacc={:.4f}]'
        with tqdm(total=self.n_iters, desc='[vacc=0.0000]',
                  disable=not verbose, ncols=80) as p:

            for n in range(self.n_iters):
                # update user factors
                update_user_factor(
                    user_item.data, user_item.indices, user_item.indptr,
                    self.embeddings_['user'], self.embeddings_['item'], self.l2
                )

                # update item factors
                update_user_factor(
                    item_user.data, item_user.indices, item_user.indptr,
                    self.embeddings_['item'], self.embeddings_['user'], self.l2
                )

                if valid_user_item is not None:
                    score = self.validate(user_item, valid_user_item)
                    p.set_description(dsc_tmp.format(score))
                p.update(1)

    def validate(self, user_item, valid_user_item, n_tests=2000, topk=100):
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
