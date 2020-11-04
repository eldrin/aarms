import unittest

import os
os.environ['NUMBA_NUM_THREADS'] = '1'

import numpy as np
from scipy import sparse as sp
from scipy.stats import kendalltau

from aarms.models.als import ALS
from aarms.models.transform import (linear_confidence,
                                    log_confidence, sppmi)
from base_test import TestAARMS


class TestALS(TestAARMS):
    """
    """
    def test_vanilla_factorize(self):
        """
        This test function refers a lot from::
            https://github.com/benfred/implicit/blob/master/tests/als_test.py
        """
        X = sp.csr_matrix([[1, 1, 0, 1, 0, 0],
                           [0, 1, 1, 1, 0, 0],
                           [1, 0, 1, 0, 0, 0],
                           [1, 1, 0, 0, 0, 0],
                           [0, 0, 1, 1, 0, 1],
                           [0, 1, 0, 0, 0, 1],
                           [0, 0, 0, 0, 1, 1]])

        cases = [
            (solver, dtype, transform)
            for dtype in (np.float32, np.float64)
            for solver in ('cg', 'lu')
            for transform in (linear_confidence, log_confidence, sppmi)
        ]

        for solver, dtype, transform in cases:
            try:
                als = ALS(k = 6,
                          l2 = 0.,
                          n_iters = 30,
                          cg_steps = 5,
                          eps = 1e-20,
                          transform = transform,
                          dtype = dtype)
                als.fit(X)

            except Exception as e:
                self.fail(msg = "failed for basic user-item factorization: "
                                f"{e}, solver={solver}, dtype={dtype}, "
                                f"transform={transform}")

            Xhat = als.embeddings_['user'] @ als.embeddings_['item'].T
            self._compare_recon(X, Xhat, thresh=1e-3,
                                **{'solver': solver, 'dtype': dtype,
                                   'transform': transform})

    def test_factorize_sideinfo(self, lmbda=.3, thres=.6, n_trials=3):
        """ this is an extension of the above test

        it check all the combination of possible scenarios
        where various side information data are given
        """
        # user-item
        X = sp.csr_matrix([[1, 1, 0, 1, 0, 0],
                           [0, 1, 1, 1, 0, 0],
                           [1, 0, 1, 0, 0, 0],
                           [1, 1, 0, 0, 0, 0],
                           [0, 0, 1, 1, 0, 1],
                           [0, 1, 0, 0, 0, 1],
                           [0, 0, 0, 0, 1, 1]])
        # user-user
        Y = sp.csr_matrix([[1, 1, 1, 0, 0, 0, 1],
                           [0, 0, 1, 1, 1, 0, 0],
                           [1, 0, 1, 0, 1, 0, 1],
                           [0, 1, 0, 1, 0, 1, 0],
                           [0, 1, 0, 1, 1, 0, 0],
                           [1, 1, 0, 0, 1, 0, 0],
                           [1, 0, 0, 0, 1, 0, 1]])
        # item-item
        Z = sp.csr_matrix([[0, 1, 0, 0, 1, 0],
                           [1, 0, 0, 1, 1, 1],
                           [0, 0, 1, 0, 0, 1],
                           [0, 1, 1, 1, 0, 0],
                           [1, 1, 0, 0, 0, 0],
                           [1, 0, 1, 1, 1, 1]])

        # user-other
        G = sp.csr_matrix([[0, 1, 0, 1],
                           [1, 1, 0, 0],
                           [0, 1, 1, 1],
                           [1, 0, 0, 1],
                           [0, 0, 1, 1],
                           [1, 0, 1, 0],
                           [0, 1, 0, 1]])

        # item-other
        H = sp.csr_matrix([[0, 1, 1, 0],
                           [1, 0, 1, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1],
                           [1, 1, 0, 0],
                           [0, 0, 1, 1]])

        # user-sparse-feature
        S = sp.csr_matrix([[0, 1, 1],
                           [1, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0],
                           [0, 0, 1],
                           [1, 1, 0],
                           [1, 0, 0]])

        # item-sparse-feature
        R = sp.csr_matrix([[1, 1, 0, 1],
                           [1, 0, 0, 1],
                           [1, 0, 0, 0],
                           [0, 0, 1, 0],
                           [1, 0, 0, 1],
                           [0, 0, 1, 0]])

        # user-dense-feature
        A = np.array([[0, 1, 1],
                      [1, 0, 1],
                      [0, 0, 1],
                      [1, 1, 0],
                      [0, 1, 0],
                      [1, 0, 1],
                      [0, 0, 0]])

        # item-dense-feature
        B = np.array([[0, 1],
                      [0, 1],
                      [1, 0],
                      [0, 0],
                      [1, 0],
                      [1, 1]])

        cases = [
            (
                solver, dtype, transform,
                user_user, item_item, user_other, item_other,
                user_sparse_feature, item_sparse_feature,
                user_dense_feature, item_dense_feature
            )
            for dtype in (np.float32, np.float64)
            for solver in ('cg', 'lu')
            for transform in (linear_confidence, log_confidence, sppmi)
            for user_user in (None, Y)
            for item_item in (None, Z)
            for user_other in (None, G)
            for item_other in (None, H)
            for user_sparse_feature in (None, S)
            for item_sparse_feature in (None, R)
            for user_dense_feature in (None, A)
            for item_dense_feature in (None, B)
        ]

        for case in cases:
            (
                solver, dtype, transform,
                user_user, item_item, user_other, item_other,
                user_sparse_feature, item_sparse_feature,
                user_dense_feature, item_dense_feature
            ) = case
            try:
                als = ALS(k = 6,
                          l2 = 0.,
                          n_iters = 30,
                          cg_steps = 5,
                          eps = 1e-20,
                          transform = transform,
                          dtype = dtype)

                for _ in range(n_trials):
                    als.fit(
                        X,
                        user_user=user_user,
                        item_item=item_item,
                        user_other=user_other,
                        item_other=item_other,
                        user_sparse_feature=user_sparse_feature,
                        item_sparse_feature=item_sparse_feature,
                        user_dense_feature=user_dense_feature,
                        item_dense_feature=item_dense_feature,
                        lmbda_user_user=-1 if user_user is None else lmbda,
                        lmbda_user_other=-1 if user_other is None else lmbda,
                        lmbda_user_dense_feature=(-1
                                                  if user_dense_feature is None
                                                  else lmbda),
                        lmbda_user_sparse_feature=(-1
                                                   if user_sparse_feature is None
                                                   else lmbda),
                        lmbda_item_item=-1 if item_item is None else lmbda,
                        lmbda_item_other=-1 if item_other is None else lmbda,
                        lmbda_item_dense_feature=(-1
                                                  if item_dense_feature is None
                                                  else lmbda),
                        lmbda_item_sparse_feature=(-1
                                                   if item_sparse_feature is None
                                                   else lmbda)
                    )

                    # evaluate simply
                    Xhat = als.embeddings_['user'] @ als.embeddings_['item'].T
                    taus = np.empty((Xhat.shape[0],))
                    for i in range(Xhat.shape[0]):
                        taus[i] = kendalltau(Xhat[i], X[i].A).correlation
                    res = np.mean(taus)
                    if res > thres:
                        break
                self.assertTrue(res > thres)

            except Exception as e:
                self.fail(msg = "failed for basic user-item factorization: "
                                f"{e}, solver={solver}, dtype={dtype}, "
                                f"transform={transform}")


if __name__ == "__main__":
    unittest.main()
