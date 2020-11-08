import numpy as np
from scipy import sparse as sp

from ..utils import slice_row_sparse


def _store_data(cur_i, container, indices, data, rnd_idx, start, end):
    """
    """
    n_records = end - start
    if n_records == 0:
        return
    container["I"].extend(np.full((end - start,), cur_i).tolist())
    container["J"].extend(indices[rnd_idx[start:end]].tolist())
    container["V"].extend(data[rnd_idx[start:end]].tolist())


def _build_mat(container, shape):
    """
    """
    return sp.coo_matrix(
        (container["V"], (container["I"], container["J"])), shape=shape
    ).tocsr()


def user_ratio_shuffle_split(X, train_ratio=0.8, valid_ratio=0.5, rand_state=None):
    """ Split given each user records into subset

    This split is to check typical internal ranking accracy.
    (not checking cold-start problem)

    Inputs:
        X (scipy.sparse.csr_matrix): user-item matrix
        train_ratio (float): ratio of training records per user
        valid_ratio (float): ratio of validation records per user
                             out of non-training records
    Returns:
        scipy.sparse.csr_matrix: training matrix
        scipy.sparse.csr_matrix: validation matrix
        scipy.sparse.csr_matrix: testing matrix
    """
    csr = X.tocsr()
    rng = np.random.RandomState(rand_state)

    # check arguments
    if train_ratio >= 1:
        raise ValueError('[ERROR] train_ratio should be smaller than 1!')
    rem = 1 - train_ratio
    valid_ratio_ = rem * valid_ratio

    # prepare empty containers
    train = {"V": [], "I": [], "J": []}
    valid = {"V": [], "I": [], "J": []}
    test = {"V": [], "I": [], "J": []}
    for i in range(X.shape[0]):
        idx, dat = slice_row_sparse(csr, i)
        rnd_idx = rng.permutation(len(idx))
        if len(idx) <= 1:
            _store_data(i, train, idx, dat, rnd_idx, 0, len(idx))

        n = len(idx)
        train_bound = int(train_ratio * n)
        if rng.rand() > 0.5:
            valid_bound = int(valid_ratio_ * n) + train_bound
        else:
            valid_bound = int(valid_ratio_ * n) + train_bound + 1

        _store_data(i, train, idx, dat, rnd_idx, 0, train_bound)
        _store_data(i, valid, idx, dat, rnd_idx, train_bound, valid_bound)
        _store_data(i, test, idx, dat, rnd_idx, valid_bound, n)

    return tuple(
        _build_mat(container, X.shape)
        for container in [train, valid, test]
    )


def nonzero_shuffle_split(X, train_ratio=0.8, valid_ratio=0.5, rand_state=None):
    """ Split given non-zero entries (observations)

    Unconditioned split often used for matrix-completion task formulation

    Inputs:
        X (scipy.sparse.csr_matrix): user-item matrix
        train_ratio (float): ratio of training records per user
        valid_ratio (float): ratio of validation records per user
                             out of non-training records
        rand_state (None or int): random state to fix it
    Returns:
        scipy.sparse.csr_matrix: training matrix
        scipy.sparse.csr_matrix: validation matrix
        scipy.sparse.csr_matrix: testing matrix
    """
    # get random generator
    rng = np.random.RandomState(rand_state)

    # check arguments
    if train_ratio >= 1:
        raise ValueError('[ERROR] train_ratio should be smaller than 1!')
    rem = 1  - train_ratio
    valid_ratio_ = rem * valid_ratio

    coo = X.tocoo()  # it's easier to handle
    n = X.nnz
    rnd_idx = rng.permutation(n)

    train_bound = int(train_ratio * n)
    if np.random.rand() > 0.5:
        valid_bound = int(valid_ratio_ * n) + train_bound
    else:
        valid_bound = int(valid_ratio_ * n) + train_bound + 1

    ranges = [
        (0, train_bound),
        (train_bound, valid_bound),
        (valid_bound, None)
    ]

    split = []
    for start, end in ranges:
        x = sp.coo_matrix(
            (coo.data[rnd_idx[start:end]],
             (coo.row[rnd_idx[start:end]],
              coo.col[rnd_idx[start:end]])),
            shape=X.shape
        )

        split.append(x)

    return tuple(split)


def nonzero_kfold_split(X, k=5, rand_state=None):
    """ Generate k folds containing mutually exclusive records

    Training data is composed of records not selected for valid or test set.

    Inputs:
        X (scipy.sparse.coo_matrix): user-item matrix
        k (int): desired number of folds
        rand_state (None or int): random state to fix it
    Returns:
        scipy.sparse.csr_matrix: training matrix
        scipy.sparse.csr_matrix: validation matrix
        scipy.sparse.csr_matrix: testing matrix
    """
    n_folds = k
    rng = np.random.RandomState(rand_state)
    rnd_idx = rng.permutation(X.nnz)
    bags = np.array_split(rnd_idx, k)
    coo = X.tocoo()  # it's easier to handle
    
    folds = []
    for i in range(n_folds):
        test_bag = i
        valid_bag = np.mod(i+1, n_folds)  # next to test
        split_bags = [
            bags[test_bag],
            bags[valid_bag],
            np.hstack([
                bags[j] for j in range(n_folds)
                if j != test_bag and j != valid_bag
            ])  # rest
        ]

        fold_split = []
        for idx in split_bags:
            x = sp.coo_matrix(
                (coo.data[rnd_idx[idx]],
                 (coo.row[rnd_idx[idx]],
                  coo.col[rnd_idx[idx]])),
                shape=X.shape
            )
            fold_split.append(x)

        folds.append(tuple(fold_split))

    return folds


def gen_ksplit(X, k=5, method='user',
               train_ratio=0.8, valid_ratio=0.5, rand_state=None):
    """ get k-splits with specified method

    three methods can be selected:
        1. "user": per-user shuffle split
        2. "nonzero": shuffle batch split for non-zero records
        3. "kfold": mutually exclusive k-folds of non-zero records

    Inputs:
        X (scipy.sparse.csr_matrix): user-item matrix
        k (int): desired number of folds
        method (string): explained above. {"user", "nonzero", "kfold"}
        train_ratio (float): training set ratio (NOT USED FOR "kfold")
        valid_ratio (float): validation proportaion for non-training records
                             (NOT USED FOR "kfold")
        rand_state (None or int): random state to fix it
    Returns:
        list of tuple of sp.csr_matrix: splits
    """
    if method not in {'user', 'nonzero', 'kfold'}:
        raise ValueError(f'[ERROR] {method} is not supported! '
                         '{"user", "nonzero", "kfold"}')

    if k <= 3:
        raise ValueError('[ERROR] K should be larger than 3')

    if method == 'kfold':
        return nonzero_kfold_split(X, k, rand_state)

    else:

        folds = []
        for _ in range(k):
            if method == 'user':
                fold = user_ratio_shuffle_split(X, train_ratio, valid_ratio,
                                                rand_state)
            else:
                # nonzero shuffle
                fold = nonzero_shuffle_split(X, train_ratio, valid_ratio,
                                             rand_state)

            folds.append(fold)

        return folds
