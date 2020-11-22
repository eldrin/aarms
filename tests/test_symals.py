import unittest

import os
os.environ['NUMBA_NUM_THREADS'] = '1'

import numpy as np
from scipy import sparse as sp
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score

from aarms.models.symals import SymALS
from aarms.models.transform import (linear_confidence,
                                    log_confidence, sppmi)
from aarms.matrix import InteractionMatrix
from base_test import TestAARMS


class TestSymALS(TestAARMS):
    """
    """
    def test_vanilla_factorize(self, n_trials=1, thresh=0.8):
        """
        This test function refers a lot from::
            https://github.com/benfred/implicit/blob/master/tests/als_test.py
        """
        X = self._gen_symmetric_data()

        cases = [
            (solver, dtype, transform)
            for dtype in (np.float32, np.float64)
            for solver in ('cg', 'lu')
            for transform in (linear_confidence, log_confidence, sppmi)
        ]

        for solver, dtype, transform in cases:

            # transform input data
            X_ = InteractionMatrix(X, transform_fn=transform, dtype=dtype)
            X_.transform()

            counter = 0
            for _ in range(n_trials):
                try:
                    als = SymALS(k = 7,
                                 l2 = 0.1,
                                 n_iters = 35,
                                 cg_steps = 3,
                                 dtype = dtype)
                    als.fit(X_)

                except Exception as e:
                    self.fail(msg = "failed for basic user-item factorization: "
                                    f"{e}, solver={solver}, dtype={dtype}, "
                                    f"transform={transform}")

                Xhat = als.embeddings_['entity'] @ als.embeddings_['entity'].T
                try:
                    auc = roc_auc_score(X.A.ravel(), Xhat.ravel())
                    self.assertTrue(auc > thresh)
                except Exception as e:
                    counter += 1
                    if counter <= n_trials:
                        continue
                    else:
                        self.fail(msg=e)

    def test_factorize_sideinfo(self, lmbda=.3, thres=.7, n_trials=2):
        """ this is an extension of the above test

        it check all the combination of possible scenarios
        where various side information data are given
        """
        _, X, _, G, H, S, _, A, _ = self._gen_data()

        cases = [
            (
                solver, dtype, transform,
                user_other, user_sparse_feature, user_dense_feature
            )
            for dtype in (np.float32, np.float64)
            for solver in ('cg', 'lu')
            for transform in (linear_confidence, log_confidence, sppmi)
            for user_other in (None, G)
            for user_sparse_feature in (None, S)
            for user_dense_feature in (None, A)
        ]

        for case in cases:
            (
                solver, dtype, transform,
                user_other, user_sparse_feature, user_dense_feature
            ) = case

            # trasnform data
            X_ = InteractionMatrix(X, transform_fn=transform, dtype=dtype)
            X_.transform()

            counter = 0
            for _ in range(n_trials):
                try:
                    als = SymALS(k = 7,
                                 l2 = 0.,
                                 n_iters = 35,
                                 cg_steps = 3,
                                 dtype = dtype)
                    als.fit(
                        X_,
                        entity_other=user_other,
                        entity_sparse_feature=user_sparse_feature,
                        entity_dense_feature=user_dense_feature,
                        lmbda_entity_other=-1 if user_other is None else lmbda,
                        lmbda_entity_dense_feature=(-1
                                                    if user_dense_feature is None
                                                    else lmbda),
                        lmbda_entity_sparse_feature=(-1
                                                     if user_sparse_feature is None
                                                     else lmbda),
                    )
                except Exception as e:
                    self.fail(msg = "failed for basic user-item factorization: "
                                    f"{e}, solver={solver}, dtype={dtype}, "
                                    f"transform={transform}")


                try:
                    # evaluate simply
                    Xhat = als.embeddings_['entity'] @ als.embeddings_['entity'].T
                    taus = np.empty((Xhat.shape[0],))
                    for i in range(Xhat.shape[0]):
                        taus[i] = kendalltau(Xhat[i], X[i].A).correlation
                    res = np.mean(taus)
                    if res > thres:
                        break
                    self.assertTrue(res > thres)

                except Exception as e:
                    counter += 1
                    if counter <= n_trials:
                        continue
                    else:
                        self.fail(msg=e)

    def test_vanilla_sampled_explicit(self, n_trials=2, rand_state=1234):
        """
        """
        R = self._gen_symmetric_data(n=7, d=1, is_explicit=True, rand_state=rand_state)

        # gen data
        cases = [
            (solver, dtype)
            for dtype in (np.float32, np.float64)
            for solver in ('cg', 'lu')
        ]

        for solver, dtype in cases:

            # transform input data
            R_ = InteractionMatrix(R, is_implicit=False, dtype=dtype)

            counter = 0
            for _ in range(n_trials):
                try:
                    als = SymALS(k = 3,
                                 l2 = 0.1,
                                 n_iters = 35,
                                 cg_steps = 3,
                                 dtype = dtype)
                    als.fit(R_)

                except Exception as e:
                    self.fail(msg = "failed for basic user-item factorization: "
                                    f"{e}, solver={solver}, dtype={dtype}, "
                                    f"transform={transform}")

                Rhat = als.embeddings_['entity'] @ als.embeddings_['entity'].T
                try:
                    # compute rmse
                    M = R.tocoo()
                    rmse = 0.
                    for i, j in zip(M.row, M.col):
                        v = R[i, j]  # masked value
                        vhat = Rhat[i, j]  # predicted value
                        rmse += (v - vhat)**2
                    rmse /= M.nnz
                    # It's not as accurate as solving the problem
                    # with two sets of factors.
                    self.assertAlmostEqual(0, rmse, places=2)
                except Exception as e:
                    counter += 1
                    if counter < n_trials:
                        continue
                    else:
                        self.fail(msg=e)


if __name__ == "__main__":
    unittest.main()
