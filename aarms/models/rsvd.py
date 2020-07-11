import numpy as np
from sklearn.decomposition import TruncatedSVD

from ._als import update_user_factor
from .base import (BaseRecommender, FactorizationMixin)
from .transform import sppmi


class RSVD(FactorizationMixin, BaseRecommender):
    def __init__(self, k, kappa=1, dtype='float32'):
        """
        k: the number of eigen vector dimensions to keep
        kappa: equivalent to the number of negative samples to consider.
               In original literature, it was introduced as `k`, but we choose
               `kappa` to avoid the notation we use for the latent dimensionality
        """
        BaseRecommender.__init__(self)
        # since the randomized svd doesn't use the `init`
        # we pass some dummy value
        FactorizationMixin.__init__(self, dtype=dtype, k=k, init=1)
        self.kappa = kappa

    def __repr__(self):
        return "RSVD@{:d}".format(self.k)

    def fit(self, user_item, valid_user_item=None, verbose=False):
        """"""
        n_users, n_items = user_item.shape
        self._init_embeddings(user=n_users, item=n_items)

        # preprocess data
        user_item_sppmi = sppmi(user_item, self.kappa)

        # fit the rSVD
        svd = TruncatedSVD(self.k)
        self.embeddings_['user'] = (
            svd.fit_transform(user_item_sppmi).astype(self.f_dtype)
        )
        self.embeddings_['item'] = np.asarray(
            svd.components_.T, dtype=self.f_dtype, order='C'
        )

    def _get_score(self, user):
        """"""
        return self.embeddings_['user'][user] @ self.embeddings_['item'].T
