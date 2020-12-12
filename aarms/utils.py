import os
import logging
import numpy as np
from scipy import sparse as sp


def slice_row_sparse(csr, i):
    slc = slice(csr.indptr[i], csr.indptr[i + 1])
    return csr.indices[slc], csr.data[slc]


def argpart_sort(s, k, ascending=True):
    if ascending:
        p = s
    else:
        p = -s
    idx = np.argpartition(p, kth=k)[:k]
    return idx[np.argsort(p[idx])]


def argpart_sort_2d(s, k, ascending=True):
    if ascending:
        p = s
    else:
        p = -s
    n = p.shape[0]
    rng = np.arange(n)[:, None]
    idx = np.argpartition(p, kth=k, axis=1)[:, :k]
    inner_idx = np.argsort(p[rng, idx], axis=1)
    rec = idx[rng, inner_idx]
    return rec


def check_blas_config():
    """ checks if using OpenBlas/Intel MKL
        This function directly adopted from
        https://github.com/benfred/implicit/blob/master/implicit/utils.py
    """
    pkg_dict = {"OPENBLAS": "openblas", "MKL": "blas_mkl"}
    for pkg, name in pkg_dict.items():
        if (
            np.__config__.get_info("{}_info".format(name))
            and os.environ.get("{}_NUM_THREADS".format(pkg)) != "1"
        ):
            logging.warning(
                "{} detected, but using more than 1 thread. Its recommended "
                "to set it 'export {}_NUM_THREADS=1' to internal multithreading".format(
                    name, pkg.upper()
                )
            )


def check_spmat(mat, name="input", force_csr=True, dtype=None):
    """ check input matrix is sparse or not. otherwise, raise value error
    """
    if mat is None:
        return None

    if not sp.issparse(mat):
        raise ValueError(f"[ERROR] {name} matrix should be a" " (CSR) sparse matrix")

    if force_csr:
        mat = mat.tocsr()

    if dtype is not None:
        return mat.astype(dtype)
    else:
        return mat


def check_densemat(mat, name="input", dtype=None):
    """ check input matrix is dense
    """
    if sp.issparse(mat):
        raise ValueError(f"[ERROR] {name} matrix should be a" " (CSR) sparse matrix")

    if dtype is not None:
        return mat.astype(dtype)
    else:
        return mat