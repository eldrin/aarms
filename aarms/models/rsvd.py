import numpy as np

from .base import BaseRecommender, FactorizationMixin
from ._rsvd import rsvd
from .transform import sppmi


class RSVD(FactorizationMixin, BaseRecommender):
    def __init__(self, k, power_iteration=0, over_sampling=5, dtype="float32"):
        """
        k: the number of eigen vector dimensions to keep
           (Truncated SVD does not accept the full rank k (should be smaller)
        """
        BaseRecommender.__init__(self)
        # since the randomized svd doesn't use the `init`
        # we pass some dummy value
        FactorizationMixin.__init__(self, k=k, dtype=dtype, init=1)
        self.power_iteration = power_iteration
        self.over_sampling = over_sampling

    def __repr__(self):
        return "RSVD@{:d}".format(self.k)

    def fit(self, user_item, valid_user_item=None, verbose=False):
        """"""
        n_users, n_items = user_item.shape
        self._init_embeddings(user=n_users, item=n_items)

        # fit the rSVD
        U, s, VT = rsvd(user_item, self.k, self.power_iteration, self.over_sampling)
        self.embeddings_["user"] = U @ (np.diag(s) ** 0.5)
        self.embeddings_["item"] = ((np.diag(s) ** 0.5) @ VT).T

    def _get_score(self, user):
        """"""
        return self.embeddings_["user"][user] @ self.embeddings_["item"].T


class RSVDSPPMI(FactorizationMixin, BaseRecommender):
    def __init__(self, k, kappa=1, power_iteration=0, over_sampling=5, dtype="float32"):
        """
        k: the number of eigen vector dimensions to keep
           (Truncated SVD does not accept the full rank k (should be smaller)
        kappa: equivalent to the number of negative samples to consider.
               In original literature, it was introduced as `k`, but we choose
               `kappa` to avoid the notation we use for the latent dimensionality
        """
        BaseRecommender.__init__(self)
        # since the randomized svd doesn't use the `init`
        # we pass some dummy value
        FactorizationMixin.__init__(self, dtype=dtype, k=k, init=1)
        self.kappa = kappa
        self.power_iteration = power_iteration
        self.over_sampling = over_sampling

    def __repr__(self):
        return "RSVDSPPMI@{:d}".format(self.k)

    def fit(self, user_item, valid_user_item=None, verbose=False):
        """"""
        n_users, n_items = user_item.shape
        self._init_embeddings(user=n_users, item=n_items)

        # preprocess data
        user_item_sppmi = sppmi(user_item, self.kappa)

        # fit the rSVD
        U, s, VT = rsvd(
            user_item_sppmi, self.k, self.power_iteration, self.over_sampling
        )
        self.embeddings_["user"] = U @ (np.diag(s) ** 0.5)
        self.embeddings_["item"] = ((np.diag(s) ** 0.5) @ VT).T

    def _get_score(self, user):
        """"""
        return self.embeddings_["user"][user] @ self.embeddings_["item"].T
