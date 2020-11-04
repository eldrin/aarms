import unittest

import os
os.environ['NUMBA_NUM_THREADS'] = '1'

import numpy as np
from scipy import sparse as sp

from aarms.models.rsvd import RSVD, RSVDSPPMI
from aarms.models.transform import sppmi

from base_test import TestAARMS


class TestRSVD(TestAARMS):
    """
    """
    def test_rsvd_factorize(self):
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

        cases = [dtype for dtype in (np.float32, np.float64)]

        for dtype in cases:
            try:
                # Truncated SVD does not accept the full rank k (should be smaller)
                svd = RSVD(k = 6, dtype = dtype)
                svd.fit(X)

            except Exception as e:
                self.fail(msg = "failed for basic user-item factorization: "
                                f"{e}, dtype={dtype}, ")

            Xhat = svd.embeddings_['user'] @ svd.embeddings_['item'].T
            self._compare_recon(X, Xhat, thresh=3e-1, **{'dtype': dtype})

    def test_rsvdsppmi_factorize(self):
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

        cases = [dtype for dtype in (np.float32, np.float64)]

        for dtype in cases:
            try:
                svd = RSVDSPPMI(k = 6, dtype = dtype)
                svd.fit(X)

            except Exception as e:
                self.fail(msg = "failed for basic user-item factorization: "
                                f"{e}, dtype={dtype}, ")

            Xhat = svd.embeddings_['user'] @ svd.embeddings_['item'].T
            user_item_sppmi = sppmi(X, svd.kappa)
            self._compare_recon(user_item_sppmi, Xhat,
                                thresh=1e-3, **{'dtype': dtype})

if __name__ == "__main__":
    unittest.main()
