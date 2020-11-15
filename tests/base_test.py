import unittest

import numpy as np
from scipy import sparse as sp


class TestAARMS(unittest.TestCase):
    """
    """
    def _gen_data(self, value='binary'):
        """

        Inputs:
            value (str): select data value mode {binary, random}
                        it is useful flag to test explicit/implicit data
        """
        if value not in {'binary', 'random'}:
            raise ValueError('[ERROR] value should be either '
                             '"binary" or "random"')

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

        if value == 'binary':
            return X, Y, Z, G, H, S, R, A, B
        elif value == 'random':
            matrices = []
            for mat in [X, Y, Z, G, H, S, R, A, B]:
                if sp.issparse(mat):
                    mat.data = np.random.rand(mat.nnz)
                matrices.append(mat)
            return matrices

    def _gen_sampled_explicit_data(self, n=6, m=7, d=1, rand_state=1345):
        """
        """
        rng = np.random.RandomState(rand_state)
        U = rng.randn(n, d)
        V = rng.randn(m, d)
        X = U @ V.T  # get rank d matrix from factors

        # build mask
        M = sp.rand(X.shape[0], X.shape[1], density=0.1)

        # preparing actual target
        R = X.copy()
        R[(M.row, M.col)] = 0.  # mask
        R = sp.csr_matrix(R)

        return R

    def _compare_recon(self, X, Xhat, thresh=1e-3, **case_arguments):
        """
        """
        msg_string = ", ".join(
            "{}={}".format(k, v)
            for k, v in case_arguments.items()
        )
        for i in range(Xhat.shape[0]):
            for j in range(Xhat.shape[1]):
                self.assertAlmostEqual(
                    X[i, j], Xhat[i, j], delta=thresh,
                    msg="failed for basic user-item factorization: " +
                        "row={:d}, col={:d}, ".format(i, j) +
                        "val={:.6f}, ".format(Xhat[i, j]) + msg_string
                )
