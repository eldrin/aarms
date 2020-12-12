import numpy as np
from scipy import sparse as sp

from ..matrix import SparseMatrix


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


def linear_confidence(X, alpha=5):
    """"""
    X = check_sparse(X)
    Y = X.copy()
    Y.data = X.dtype.type(1.0) + alpha * X.data
    return Y


def log_confidence(X, alpha=5, eps=1):
    """"""
    X = check_sparse(X)
    Y = X.copy()
    one = X.dtype.type(1.0)
    Y.data = one + alpha * np.log(X.data / eps + one)
    return Y


def sppmi(X, k=1):
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


def _sparse_elm_mul(spmat_csr, col):
    """
    spmat (n, m)
    col (n,)
    """
    for i in range(spmat_csr.shape[0]):
        i0, i1 = spmat_csr.indptr[i], spmat_csr.indptr[i+1]
        if i1 == i0:
            continue
        spmat_csr.data[i0:i1] *= col[i]
    return spmat_csr


def _normalize_tfidf(csr, norm=2):
    """
    """
    for i in range(csr.shape[0]):
        i0, i1 = csr.indptr[i], csr.indptr[i+1]
        if i1 == i0:
            continue
        csr.data[i0:i1] /= np.linalg.norm(csr.data[i0:i1], ord=norm)
    return csr


def tfidf(X, norm='l2', use_idf=True, smooth_idf=True,
          sublinear_tf=False, dtype=np.float32):
    """ TF-IDF transformation

    It follows scikit-learn's API
    """
    assert norm in {'l1', 'l2'}
    X = check_sparse(X, force_type=None).astype(dtype)

    one = X.dtype.type(0.)
    n, _ = X.shape  # (num_docs, num_terms)

    if sublinear_tf:
        X.data = np.log(one + X.data)

    if use_idf:
        # get document frequency
        df = np.ediff1d(X.T.tocsr().indptr)

        # get idf
        if smooth_idf:
            idf = np.log((n + 1) / (df + 1)) + 1
        else:
            idf = np.log(n / df) + 1

    # compute tf-idf
    X = _sparse_elm_mul(X.T.tocsr(), idf).T.tocsr()
    
    # normalize if needed
    if norm is not None:
        norm_ord = 2 if norm == 'l2' else 1
        X = _normalize_tfidf(X, norm_ord)
    
    return X.tocsr()