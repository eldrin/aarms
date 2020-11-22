import multiprocessing as mp
import warnings
from functools import partial

import numpy as np
import numba as nb
from scipy import sparse as sp

from tqdm import tqdm

from ..matrix import InteractionMatrix, SparseFeatureMatrix, DenseFeatureMatrix
from ..utils import check_blas_config, check_spmat, check_densemat
from ._symals import update_entity as update_entity
from ._als import update_side as update_side
from ._als import update_weight_dense
from .base import AARMSRecommender, NumbaFactorizationMixin


class SymALS(NumbaFactorizationMixin, AARMSRecommender):
    def __init__(self, k, init=0.001, l2=0.0001, n_iters=15, dtype="float32",
                 solver="cg", cg_steps=5, n_jobs=-1):
        """
        """
        AARMSRecommender.__init__(self)
        NumbaFactorizationMixin.__init__(self, dtype, k, init, n_jobs)

        self.l2 = self.f_dtype(l2)
        self.n_iters = n_iters
        self.solver = solver
        self.cg_steps = cg_steps

    def __repr__(self):
        return "SymALS@{:d}".format(self.k)

    def fit(
        self,
        entity_entity,
        entity_other=None,
        entity_dense_feature=None,
        entity_sparse_feature=None,
        lmbda_entity_other=-1,
        lmbda_entity_dense_feature=-1,
        lmbda_entity_sparse_feature=-1,
        valid_entity_entity=None,
        verbose=False
    ):
        """ factorize given sparse user-item matrix

        Inputs:
            entity_entity (sp.csr_matrix): sparse entity-entity relevance matrix

            entity_other (sp.csr_matrix): sparse matrix for relationship of entity to other entity
            entity_dense_feature (np.ndarray): dense feature per row related to entity
            entity_sparse_feature (sp.csr_matrix): sparse feature per row for entity

            lmbda_entity_other (float): loss term weight for entity-other information
            lmbda_entity_dense_feature (float): loss term weight for entity feature (dense)
            lmbda_entity_sparse_feature (float): loss term weight for entity feature (sparse)

            valid_entity_entity: (sp.csr_matrix or None): validation set
            verbose (bool): verbosity
        """
        # put inputs to trunc
        inputs = {
            "entity_entity": {'data': entity_entity, 'lambda': self.l2},
            "valid_entity_entity": {'data': valid_entity_entity, 'lmbda': self.l2},
            "entity_other": {'data': entity_other, 'lambda': lmbda_entity_other},
            "entity_dense_feature": {'data': entity_dense_feature,
                                     'lambda': lmbda_entity_dense_feature},
            "entity_sparse_feature": {'data': entity_sparse_feature,
                                      'lambda': lmbda_entity_sparse_feature},
        }

        # check items
        inputs = self._check_inputs(inputs)

        # initialize embeddings
        self._init_embeddings(
            inputs['entity_entity']['data'],
            inputs['entity_other']['data'],
            inputs['entity_dense_feature']['data'],
            inputs['entity_sparse_feature']['data']
        )

        # compute some transposes
        inputs['entity_entity_t'] = {}
        inputs['entity_other_t'] = {}
        inputs['entity_entity_t']['data'] = inputs['entity_entity']['data'].transpose()
        inputs['entity_other_t']['data'] = inputs['entity_other']['data'].transpose()

        # fit model
        self._fit(inputs, verbose)

    def _fit(self, inputs, verbose):
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
                self._update_factor("entity", inputs)
                self._update_factor("entity_t", inputs)

                if inputs['valid_entity_entity']['data'].size > 0:
                    score = self.validate(inputs['entity_entity']['data']._data,
                                          inputs['valid_entity_entity']['data']._data)
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
        if target_entity == 'entity':
            X = inputs['entity_entity']
        elif target_entity == 'entity_t':
            X = inputs['entity_entity_t']

        G = inputs['entity_other']
        Gt = inputs['entity_other_t']
        A = inputs['entity_dense_feature']
        S = inputs['entity_sparse_feature']

        update_entity(
            X['data']._data.data,
            X['data']._data.indices,
            X['data']._data.indptr,
            G['data']._data.data,
            G['data']._data.indices,
            G['data']._data.indptr,
            S['data']._data.data,
            S['data']._data.indices,
            S['data']._data.indptr,
            self.embeddings_['entity'],
            self.embeddings_["entity_other"],
            self.embeddings_["entity_dense_feature"],
            self.embeddings_["entity_sparse_feature"],
            A['data']._data,
            G['lambda'],
            A['lambda'],
            S['lambda'],
            self.l2,
            X['data'].is_sampled_explicit,
            G['data'].is_sampled_explicit,
            self.solver,
            self.cg_steps,
            eps,
        )
        update_side(
            Gt['data']._data.data,
            Gt['data']._data.indices,
            Gt['data']._data.indptr,
            self.embeddings_["entity_other"],
            self.embeddings_["entity"],
            G['lambda'],
            self.l2,
            G['data'].is_sampled_explicit,
            self.solver,
            self.cg_steps,
            eps,
        )
        update_weight_dense(
            self.embeddings_["entity"],
            A['data']._data,
            self.embeddings_["entity_dense_feature"],
            A['lambda'],
            self.l2,
            eps,
        )

    def _get_score(self, node):
        """
        """
        return self.embeddings_["entity"][node] @ self.embeddings_["entity"].T

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

            if name == 'valid_entity_entity' and data.size > 0:
                assert inputs['entity_entity']['data'].shape == data.shape

            if name in {'entity_other', 'entity_sparse_feature',
                        'entity_dense_feature'}:
                if data.size > 0:
                    assert inputs['entity_entity']['data'].shape[0] == data.shape[0]
                    
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
        dims = {"entity": user_item.shape[0]}

        if user_other.size > 0:
            dims["entity_other"] = user_other.shape[1]

        if user_dense_feature.size > 0:
            dims["entity_dense_feature"] = user_dense_feature.shape[1]

        if user_sparse_feature.size > 0:
            dims["entity_sparse_feature"] = user_sparse_feature.shape[1]

        # actually prepare embeddings
        super()._init_embeddings(**dims)

        # park placeholders
        all_embs = [
            "entity",
            "entity_other",
            "entity_dense_feature",
            "entity_sparse_feature",
        ]
        for name in all_embs:
            if name not in self.embeddings_:
                self.embeddings_[name] = np.array([[]], dtype=self.dtype)