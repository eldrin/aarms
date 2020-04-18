import os
import logging
import numpy as np
from scipy import sparse as sp


def split_recsys_data(X, train_ratio=0.8, valid_ratio=0.1):
    """ Split given user-item matrix into train/test.
    This split is to check typical internal ranking accracy.
    (not checking cold-start problem)
    Inputs:
        X (scipy.sparse.csr_matrix): user-item matrix
        train_ratio (float): ratio of training records per user
        test_ratio (float): ratio of validation records per user
    Returns:
        scipy.sparse.csr_matrix: training matrix
        scipy.sparse.csr_matrix: validation matrix
        scipy.sparse.csr_matrix: testing matrix
    """
    def _store_data(cur_i, container, indices, data, rnd_idx, start, end):
        n_records = end - start
        if n_records == 0:
            return
        container['I'].extend(np.full((end - start,), cur_i).tolist())
        container['J'].extend(indices[rnd_idx[start:end]].tolist())
        container['V'].extend(data[rnd_idx[start:end]].tolist())

    def _build_mat(container, shape):
        return sp.coo_matrix(
            (container['V'], (container['I'], container['J'])),
            shape=shape
        ).tocsr()

    # prepare empty containers
    train = {'V': [], 'I': [], 'J': []}
    valid = {'V': [], 'I': [], 'J': []}
    test = {'V': [], 'I': [], 'J': []}
    for i in range(X.shape[0]):
        idx, dat = slice_row_sparse(X, i)
        rnd_idx = np.random.permutation(len(idx))
        n = len(idx)
        train_bound = int(train_ratio * n)
        if np.random.rand() > 0.5:
            valid_bound = int(valid_ratio * n) + train_bound
        else:
            valid_bound = int(valid_ratio * n) + train_bound + 1

        _store_data(i, train, idx, dat, rnd_idx, 0, train_bound)
        _store_data(i, valid, idx, dat, rnd_idx, train_bound, valid_bound)
        _store_data(i, test, idx, dat, rnd_idx, valid_bound, n)

    return tuple(
        _build_mat(container, X.shape)
        for container in [train, valid, test]
    )


def slice_row_sparse(csr, i):
    slc = slice(csr.indptr[i], csr.indptr[i+1])
    return csr.indices[slc], csr.data[slc]


def argpart_sort(s, k, ascending=True):
    if ascending: p = s
    else:         p = -s
    idx = np.argpartition(p, kth=k)[:k]
    return idx[np.argsort(p[idx])]


def argpart_sort_2d(s, k, ascending=True):
    if ascending: p = s
    else:         p = -s
    n = p.shape[0]
    rng = np.arange(n)[:, None]
    idx = np.argpartition(p, kth=k, axis=1)[:, :k]
    inner_idx = np.argsort(p[rng, idx], axis=1)
    rec = idx[rng, inner_idx]
    return rec


def densify(ui_csr, users, items, item_feat=None, thresh=5, user_sample=0.3):
    """ Densify the User-Item interactio matrix
    """
    def _filt_entity(csr, entities, thresh):
        filt_targs = np.where(np.ediff1d(csr.indptr) >= thresh)[0]
        return csr[filt_targs], entities[filt_targs], filt_targs

    n_users, n_items = ui_csr.shape
    users = np.asarray(users)
    items = np.asarray(items)

    if user_sample > 0:
        assert user_sample < 1
        p = user_sample
        uid = np.random.choice(n_users, int(n_users * p), False)
        ui_csr = ui_csr[uid]
        users = users[uid]

    diff = 1
    while diff > 0:
        prev_nnz = ui_csr.nnz
        iu_csr, items, filt_idx = _filt_entity(ui_csr.T.tocsr(), items, thresh)
        if item_feat is not None:
            item_feat = item_feat[filt_idx]
        ui_csr, users, filt_idx = _filt_entity(iu_csr.T.tocsr(), users, thresh)
        diff = prev_nnz - ui_csr.nnz
    return ui_csr, users, items, item_feat


def check_blas_config():
    """ checks if using OpenBlas/Intel MKL
        This function directly adopted from
        https://github.com/benfred/implicit/blob/master/implicit/utils.py
    """
    pkg_dict = {'OPENBLAS':'openblas', 'MKL':'blas_mkl'}
    for pkg, name in pkg_dict.items():
        if (np.__config__.get_info('{}_info'.format(name))
            and
            os.environ.get('{}_NUM_THREADS'.format(pkg)) != '1'):
            logging.warning(
                "{} detected, but using more than 1 thread. Its recommended "
                "to set it 'export {}_NUM_THREADS=1' to internal multithreading"
                .format(name, pkg)
            )
