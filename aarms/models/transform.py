import numpy as np
from scipy import sparse as sp


def check_sparse(X, force_type="csr"):
    """"""
    if sp.issparse(X):
        if force_type is not None:
            if force_type == "csr":
                X = X.tocsr()
            elif force_type == "csc":
                X = X.tocsc()
            elif force_type == "coo":
                X = X.tocoo()
            else:
                raise ValueError('[ERROR] Not support "{}" type!'.format(force_type))
        return X
    else:
        raise ValueError("[ERROR] Input is not a sparse matrix!")


def linear_confidence(X, alpha):
    """"""
    X = check_sparse(X)
    Y = X.copy()
    Y.data = X.dtype.type(1.0) + alpha * X.data
    return Y


def log_confidence(X, alpha, eps):
    """"""
    X = check_sparse(X)
    Y = X.copy()
    one = X.dtype.type(1.0)
    Y.data = one + alpha * np.log(X.data / eps + one)
    return Y


def sppmi(X, k):
    """"""
    X = check_sparse(X, force_type=None)

    # setup some variables
    zero = X.dtype.type(0.0)
    k = X.dtype.type(k)
    D = X.dtype.type(X.nnz)
    n_w = X.sum(1).A.ravel().astype(X.dtype)
    n_c = X.sum(0).A.ravel().astype(X.dtype)
    sppmi = X.copy().tocoo()

    # prepare word counts
    w = n_w[sppmi.row]
    c = n_c[sppmi.col]
    wc = sppmi.data

    # process
    sppmi.data = np.maximum(np.log(wc / (w * c) * D / k), zero)

    # make sure the zeroed entry considered in sparse_matrix scheme
    sppmi.eliminate_zeros()
    return sppmi.tocsr()
