import multiprocessing as mp
from functools import partial

import numpy as np
import numba as nb
from scipy import sparse as sp

from tqdm import tqdm

from ..utils import check_blas_config, check_spmat, check_densemat
from ._als import update_side as update_side
from ._als import update_user as update_user
from ._als import update_weight_dense
from .base import BaseRecommender, FactorizationMixin
from .transform import linear_confidence, log_confidence, sppmi


class ALS(FactorizationMixin, BaseRecommender):
    def __init__(
        self,
        k,
        init=0.001,
        l2=0.0001,
        n_iters=15,
        alpha=5,
        eps=0.5,
        kappa=1,
        transform=linear_confidence,
        dtype="float32",
        solver="cg",
        cg_steps=5,
        n_jobs=-1,
    ):
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
            raise ValueError("[ERROR] not supported transform given!")

    def __repr__(self):
        return "ALS@{:d}".format(self.k)

    def fit(
        self,
        user_item,
        user_user=None,
        user_other=None,
        user_dense_feature=None,
        user_sparse_feature=None,
        item_item=None,
        item_other=None,
        item_dense_feature=None,
        item_sparse_feature=None,
        lmbda_user_user=-1,
        lmbda_user_other=-1,
        lmbda_user_dense_feature=-1,
        lmbda_user_sparse_feature=-1,
        lmbda_item_item=-1,
        lmbda_item_other=-1,
        lmbda_item_dense_feature=-1,
        lmbda_item_sparse_feature=-1,
        valid_user_item=None,
        verbose=False,
    ):
        """ factorize given sparse user-item matrix

        Inputs:
            user_item (sp.csr_matrix): sparse user-item matrix

            user_user (sp.csr_matrix): sparse matrix for relationship of user to user
            user_other (sp.csr_matrix): sparse matrix for relationship of user to other entity
            user_dense_feature (np.ndarray): dense feature per row related to user
            user_sparse_feature (sp.csr_matrix): sparse feature per row for user

            item_other (sp.csr_matrix): sparse matrix for relationship of item to other entity
            item_item (sp.csr_matrix): sparse matrix for relationship of item to item
            user_dense_feature (np.ndarray): dense feature per row related to item
            user_sparse_feature (sp.csr_matrix): sparse feature per row for items

            lmbda_user_user (float): loss term weight for user-user information
            lmbda_user_other (float): loss term weight for user-other information
            lmbda_user_dense_feature (float): loss term weight for user feature (dense)
            lmbda_user_sparse_feature (float): loss term weight for user feature (sparse)

            lmbda_item_item (float): loss term weight for item-item information
            lmbda_item_other (float): loss term weight for item-other information
            lmbda_item_dense_feature (float): loss term weight for item feature (dense)
            lmbda_item_sparse_feature (float): loss term weight for item feature (sparse)

            valid_user_item (sp.csr_matrix or None): validation set
            verbose (bool): verbosity
        """
        # put lambdas to trunc
        lmbdas = {
            "user_user": lmbda_user_user,
            "user_other": lmbda_user_other,
            "user_dense_feature": lmbda_user_dense_feature,
            "user_sparse_feature": lmbda_user_sparse_feature,
            "item_item": lmbda_item_item,
            "item_other": lmbda_item_other,
            "item_dense_feature": lmbda_item_dense_feature,
            "item_sparse_feature": lmbda_item_sparse_feature,
        }

        # check items
        (
            user_item,
            valid_user_item,
            user_user,
            user_other,
            user_sparse_feature,
            user_dense_feature,
            item_item,
            item_other,
            item_sparse_feature,
            item_dense_feature,
        ) = self._check_inputs(
            lmbdas,
            user_item,
            user_user,
            user_other,
            user_dense_feature,
            user_sparse_feature,
            item_item,
            item_other,
            item_dense_feature,
            item_sparse_feature,
            valid_user_item,
        )

        # initialize embeddings
        self._init_embeddings(
            user_item,
            user_other,
            user_dense_feature,
            user_sparse_feature,
            item_other,
            item_dense_feature,
            item_sparse_feature,
        )

        # preprocess data
        user_item = self._transform(user_item.astype(self.dtype))

        # fit model
        self._fit(
            user_item,
            valid_user_item,
            user_user,
            user_other,
            user_sparse_feature,
            user_dense_feature,
            item_item,
            item_other,
            item_sparse_feature,
            item_dense_feature,
            lmbdas,
            verbose,
        )

    def _fit(
        self,
        user_item,
        valid_user_item,
        user_user,
        user_other,
        user_sparse_feature,
        user_dense_feature,
        item_item,
        item_other,
        item_sparse_feature,
        item_dense_feature,
        lmbdas,
        verbose,
    ):
        """
        """
        item_user = user_item.T.tocsr()
        user_other_t = user_other.T.tocsr()
        item_other_t = item_other.T.tocsr()

        # set the number of threads for training
        dsc_tmp = "[vacc={:.4f}]"
        with tqdm(
            total=self.n_iters, desc="[vacc=0.0000]", disable=not verbose, ncols=80
        ) as p:

            for n in range(self.n_iters):
                self._update_factor(
                    "user",
                    user_item,
                    user_user,
                    user_other,
                    user_other_t,
                    user_dense_feature,
                    user_sparse_feature,
                    lmbdas,
                )
                self._update_factor(
                    "item",
                    item_user,
                    item_item,
                    item_other,
                    item_other_t,
                    item_dense_feature,
                    item_sparse_feature,
                    lmbdas,
                )

                if valid_user_item.size > 0:
                    score = self.validate(user_item, valid_user_item)
                    p.set_description(dsc_tmp.format(score))
                p.update(1)

        # finalize embeddings
        self.embeddings_ = {
            name: fac for name, fac in self.embeddings_.items() if fac.size > 0
        }

    def _update_factor(
        self,
        target_entity,
        user_item,
        user_user,
        user_other,
        user_other_t,
        user_dense_feature,
        user_sparse_feature,
        lmbdas,
    ):
        """
        """
        opposite_entity = "item" if target_entity == "user" else "user"

        update_user(
            user_item.data,
            user_item.indices,
            user_item.indptr,
            user_user.data,
            user_user.indices,
            user_user.indptr,
            user_other.data,
            user_other.indices,
            user_other.indptr,
            user_sparse_feature.data,
            user_sparse_feature.indices,
            user_sparse_feature.indptr,
            self.embeddings_[f"{target_entity}"],
            self.embeddings_[f"{opposite_entity}"],
            self.embeddings_[f"{target_entity}_other"],
            self.embeddings_[f"{target_entity}_dense_feature"],
            self.embeddings_[f"{target_entity}_sparse_feature"],
            user_dense_feature,
            lmbdas[f"{target_entity}_{target_entity}"],
            lmbdas[f"{target_entity}_other"],
            lmbdas[f"{target_entity}_dense_feature"],
            lmbdas[f"{target_entity}_sparse_feature"],
            self.l2,
            self.solver,
            self.cg_steps,
            self.eps,
        )
        update_side(
            user_other_t.data,
            user_other_t.indices,
            user_other_t.indptr,
            self.embeddings_[f"{target_entity}_other"],
            self.embeddings_[f"{target_entity}"],
            lmbdas[f"{target_entity}_other"],
            self.l2,
            self.solver,
            self.cg_steps,
            self.eps,
        )
        update_weight_dense(
            self.embeddings_[f"{target_entity}"],
            user_dense_feature,
            self.embeddings_[f"{target_entity}_dense_feature"],
            lmbdas[f"{target_entity}_dense_feature"],
            self.l2,
            self.eps,
        )

    def _get_score(self, user):
        """"""
        return self.embeddings_["user"][user] @ self.embeddings_["item"].T

    def _check_inputs(
        self,
        lmbdas,
        user_item,
        user_user=None,
        user_other=None,
        user_dense_feature=None,
        user_sparse_feature=None,
        item_item=None,
        item_other=None,
        item_dense_feature=None,
        item_sparse_feature=None,
        valid_user_item=None,
    ):
        """
        """
        # prepare empty csr matrix for placeholder
        dummy = sp.csr_matrix((0, 0), dtype=self.dtype)
        dummy_dense = np.array([[]], dtype=self.dtype)

        # building checking target
        to_check = [
            (user_item, "user_item"),
            (valid_user_item, "valid_user_item"),
            (user_user, "user_user"),
            (user_other, "user_other"),
            (user_sparse_feature, "user_sparse_feature"),
            (item_item, "item_item"),
            (item_other, "item_other"),
            (item_sparse_feature, "item_sparse_feature"),
            (user_dense_feature, "user_dense_feature"),
            (item_dense_feature, "item_dense_feature"),
        ]

        # check lambdas
        for mat, name in to_check:
            if mat is None:
                lmbdas[name] = -1

        # check sparse features
        (
            user_item,
            valid_user_item,
            user_user,
            user_other,
            user_sparse_feature,
            item_item,
            item_other,
            item_sparse_feature,
        ) = [
            check_spmat(mat, name, dtype=self.dtype)
            if mat is not None
            else dummy.copy()  # fill the not existing members with dummy
            for mat, name in [e for e in to_check if "dense" not in e[1]]
        ]

        # check dense features
        user_dense_feature = (
            check_densemat(user_dense_feature, dtype=self.dtype)
            if user_dense_feature is not None
            else dummy_dense.copy()
        )
        item_dense_feature = (
            check_densemat(item_dense_feature, dtype=self.dtype)
            if item_dense_feature is not None
            else dummy_dense.copy()
        )

        # check size of the data
        if valid_user_item.size > 0:
            assert user_item.shape == valid_user_item.shape

        for mat in (user_user, item_item):
            if mat.size > 0:
                assert mat.shape[0] == mat.shape[1]

        for mat in (user_user, user_other, user_sparse_feature, user_dense_feature):
            if mat.size > 0:
                assert mat.shape[0] == user_item.shape[0]

        for mat in (item_item, item_other, item_sparse_feature, item_dense_feature):
            if mat.size > 0:
                assert mat.shape[0] == user_item.shape[1]

        return (
            user_item,
            valid_user_item,
            user_user,
            user_other,
            user_sparse_feature,
            user_dense_feature,
            item_item,
            item_other,
            item_sparse_feature,
            item_dense_feature,
        )

    def _init_embeddings(
        self,
        user_item,
        user_other=None,
        user_dense_feature=None,
        user_sparse_feature=None,
        item_other=None,
        item_dense_feature=None,
        item_sparse_feature=None,
    ):
        """ overriding embedding initialization method
        """
        dims = {"user": user_item.shape[0], "item": user_item.shape[1]}

        if user_other.size > 0:
            dims["user_other"] = user_other.shape[1]

        if user_dense_feature.size > 0:
            dims["user_dense_feature"] = user_dense_feature.shape[1]

        if user_sparse_feature.size > 0:
            dims["user_sparse_feature"] = user_sparse_feature.shape[1]

        if item_other.size > 0:
            dims["item_other"] = item_other.shape[1]

        if item_dense_feature.size > 0:
            dims["item_dense_feature"] = item_dense_feature.shape[1]

        if item_sparse_feature.size > 0:
            dims["item_sparse_feature"] = item_sparse_feature.shape[1]

        # actually prepare embeddings
        super()._init_embeddings(**dims)

        # park placeholders
        all_embs = [
            "user",
            "item",
            "user_other",
            "user_dense_feature",
            "user_sparse_feature",
            "item_other",
            "item_dense_feature",
            "item_sparse_feature",
        ]
        for name in all_embs:
            if name not in self.embeddings_:
                self.embeddings_[name] = np.array([[]], dtype=self.dtype)
