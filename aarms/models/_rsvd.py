from scipy import sparse as sp
import numpy as np
import numba as nb


def rsvd(X, r, q, p):
    """
    """
    n, m = X.shape

    # over-sampling with amount of p
    P = np.random.randn(m, r + p)

    Z = X @ P
    # power iteration
    for _ in range(q):
        Z = X @ (X.T @ Z)

    Q, R = np.linalg.qr(Z)
    Y = Q.T @ X
    UY, s, VT = np.linalg.svd(Y, full_matrices=False)
    U = Q @ UY
    return U[:, :r], s[:r], VT[:r]
