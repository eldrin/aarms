import multiprocessing as mp
import warnings
from functools import partial

import numpy as np
import numba as nb
from scipy import sparse as sp

from tqdm import tqdm

from ..matrix import InteractionMatrix, SparseFeatureMatrix, DenseFeatureMatrix
from ..utils import check_blas_config, check_spmat, check_densemat
from ._als import update_side as update_side
from ._als import update_user as update_user
from ._als import update_weight_dense
from .base import BaseRecommender, FactorizationMixin


class ALS(FactorizationMixin, BaseRecommender):
    def __init__(
        self,
        k,
        init=0.001,
        l2=0.0001,
        n_iters=15,
        dtype="float32",
        solver="cg",
        cg_steps=5,
        n_jobs=-1
    ):
        """"""
        BaseRecommender.__init__(self)
        FactorizationMixin.__init__(self, dtype, k, init)

        self.l2 = self.f_dtype(l2)
        self.n_iters = n_iters
        self.n_jobs = n_jobs
        self.solver = solver
        self.cg_steps = cg_steps

        check_blas_config()
        if n_jobs > nb.config.NUMBA_NUM_THREADS:
            warnings.warn('n_jobs should be set lower than the number of cores! '
                          'setting it to the number...')
            self.n_jobs = nb.config.NUMBA_NUM_THREADS
        elif n_jobs == -1:
            self.n_jobs = nb.config.NUMBA_NUM_THREADS
        else:
            self.n_jobs = n_jobs

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
        # put inputs to trunc
        inputs = {
            "user_item": {'data': user_item, 'lambda': self.l2},
            "valid_user_item": {'data': valid_user_item, 'lmbda': self.l2},
            "user_user": {'data': user_user, 'lambda': lmbda_user_user},
            "user_other": {'data': user_other, 'lambda': lmbda_user_other},
            "user_dense_feature": {'data': user_dense_feature,
                                   'lambda': lmbda_user_dense_feature},
            "user_sparse_feature": {'data': user_sparse_feature,
                                    'lambda': lmbda_user_sparse_feature},
            "item_item": {'data': item_item, 'lambda': lmbda_item_item},
            "item_other": {'data': item_other, 'lambda': lmbda_item_other},
            "item_dense_feature": {'data': item_dense_feature,
                                   'lambda': lmbda_item_dense_feature},
            "item_sparse_feature": {'data': item_sparse_feature,
                                    'lambda': lmbda_item_sparse_feature}
        }

        # check items
        inputs = self._check_inputs(inputs)

        # initialize embeddings
        self._init_embeddings(
            inputs['user_item']['data'],
            inputs['user_other']['data'],
            inputs['user_dense_feature']['data'],
            inputs['user_sparse_feature']['data'],
            inputs['item_other']['data'],
            inputs['item_dense_feature']['data'],
            inputs['item_sparse_feature']['data'],
        )

        # compute some transposes
        inputs['item_user'] = {}
        inputs['user_other_t'] = {}
        inputs['item_other_t'] = {}
        inputs['item_user']['data'] = inputs['user_item']['data'].transpose()
        inputs['user_other_t']['data'] = inputs['user_other']['data'].transpose()
        inputs['item_other_t']['data'] = inputs['item_other']['data'].transpose()

        # fit model
        self._fit(inputs, valid_user_item, verbose)

    def _fit(self, inputs, valid_user_item, verbose):
        """
        """
        # set threading
        if self.n_jobs >= 1:
            nb.set_num_threads(self.n_jobs)

        # set the number of threads for training
        dsc_tmp = "[vacc={:.4f}]"
        with tqdm(
            total=self.n_iters, desc="[vacc=0.0000]", disable=not verbose, ncols=80
        ) as p:

            for n in range(self.n_iters):
                self._update_factor("user", inputs)
                self._update_factor("item", inputs)

                if inputs['valid_user_item']['data'].size > 0:
                    score = self.validate(inputs['user_item']['data']._data,
                                          inputs['valid_user_item']['data']._data)
                    p.set_description(dsc_tmp.format(score))
                p.update(1)

        # finalize embeddings
        self.embeddings_ = {
            name: fac for name, fac in self.embeddings_.items() if fac.size > 0
        }

        # set the number of threads to the default
        if self.n_jobs >= 1:
            nb.set_num_threads(nb.config.NUMBA_NUM_THREADS)

    def _update_factor(self, target_entity, inputs, eps=1e-20):
        """
        """
        eps = self.f_dtype(eps)
        opposite_entity = "item" if target_entity == "user" else "user"
        X = inputs[f"{target_entity}_{opposite_entity}"]
        Y = inputs[f"{target_entity}_{target_entity}"]
        G = inputs[f'{target_entity}_other']
        Gt = inputs[f'{target_entity}_other_t']
        A = inputs[f'{target_entity}_dense_feature']
        S = inputs[f'{target_entity}_sparse_feature']

        update_user(
            X['data']._data.data,
            X['data']._data.indices,
            X['data']._data.indptr,
            Y['data']._data.data,
            Y['data']._data.indices,
            Y['data']._data.indptr,
            G['data']._data.data,
            G['data']._data.indices,
            G['data']._data.indptr,
            S['data']._data.data,
            S['data']._data.indices,
            S['data']._data.indptr,
            self.embeddings_[f"{target_entity}"],
            self.embeddings_[f"{opposite_entity}"],
            self.embeddings_[f"{target_entity}_other"],
            self.embeddings_[f"{target_entity}_dense_feature"],
            self.embeddings_[f"{target_entity}_sparse_feature"],
            A['data']._data,
            Y['lambda'],
            G['lambda'],
            A['lambda'],
            S['lambda'],
            self.l2,
            X['data'].is_sampled_explicit,
            Y['data'].is_sampled_explicit,
            G['data'].is_sampled_explicit,
            self.solver,
            self.cg_steps,
            eps,
        )
        update_side(
            Gt['data']._data.data,
            Gt['data']._data.indices,
            Gt['data']._data.indptr,
            self.embeddings_[f"{target_entity}_other"],
            self.embeddings_[f"{target_entity}"],
            G['lambda'],
            self.l2,
            G['data'].is_sampled_explicit,
            self.solver,
            self.cg_steps,
            eps,
        )
        update_weight_dense(
            self.embeddings_[f"{target_entity}"],
            A['data']._data,
            self.embeddings_[f"{target_entity}_dense_feature"],
            A['lambda'],
            self.l2,
            eps,
        )

    def _get_score(self, user):
        """"""
        return self.embeddings_["user"][user] @ self.embeddings_["item"].T

    def _check_inputs(self, inputs):
        """
        """
        # prepare empty csr matrix for placeholder
        for name, term_data in inputs.items():
            data = term_data['data']

            if 'dense' not in name:
                # either interaction or sparse feature
                if 'sparse' in name:
                    # sparse feature
                    if not isinstance(data, SparseFeatureMatrix):
                        data = SparseFeatureMatrix(data, self.dtype)
                else:
                    # interaction matrix
                    if not isinstance(data, InteractionMatrix):
                        data = InteractionMatrix(data, dtype=self.dtype)
            else:
                # dense feature
                if not isinstance(data, DenseFeatureMatrix):
                    data = DenseFeatureMatrix(data, dtype=self.dtype)

            # update data
            term_data.update({'data': data})

        # check size of the data
        for name, term_data in inputs.items():
            data = term_data['data']

            if name == 'valid_user_item' and data.size > 0:
                assert inputs['user_item']['data'].shape == data.shape

            if name in {'user_user', 'item_item'}:
                if data.size > 0:
                    assert data.shape[0] == data.shape[1]

            if name in {'user_user', 'user_other',
                        'user_sparse_feature', 'user_dense_feature'}:
                if data.size > 0:
                    assert inputs['user_item']['data'].shape[0] == data.shape[0]

            if name in {'item_item', 'item_other',
                        'item_sparse_feature', 'item_dense_feature'}:
                if data.size > 0:
                    assert inputs['user_item']['data'].shape[1] == data.shape[0]

        return inputs

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
