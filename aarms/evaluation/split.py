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


def user_ratio_shuffle_split(X, train_ratio=0.8, valid_ratio=0.5,
                             minimum_interaction=3, rand_state=None):
    """ Split given each user records into subsets

    This split is to check typical internal ranking accracy.
    (not checking cold-start problem)

    Inputs:
        X (scipy.sparse.csr_matrix): user-item matrix
        train_ratio (float): ratio of training records per user
        valid_ratio (float): ratio of validation records per user
                             out of non-training records
        minimum_interaction (int): minimum interaction of user to be considered.
                                   if it's smaller than this,
                                   put all records to the training set
        rand_state (bool or int): random state seed number or None
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
    for i in range(csr.shape[0]):
        idx, dat = slice_row_sparse(csr, i)
        n = len(idx)
        rnd_idx = rng.permutation(n)

        # not to make cornercase
        if n <= minimum_interaction:
            _store_data(i, train, idx, dat, rnd_idx, 0, n)
            continue

        # compute each bounds for valid/train
        train_bound = int(train_ratio * n)
        if rng.rand() > 0.5:
            valid_bound = int(valid_ratio_ * n) + train_bound
        else:
            valid_bound = int(valid_ratio_ * n) + train_bound + 1

        # store them to the containers
        _store_data(i, train, idx, dat, rnd_idx, 0, train_bound)
        _store_data(i, valid, idx, dat, rnd_idx, train_bound, valid_bound)
        _store_data(i, test, idx, dat, rnd_idx, valid_bound, n)

    return tuple(
        _build_mat(container, csr.shape)
        for container in [train, valid, test]
    )


def user_ratio_shuffle_split_with_targets(X,
                                          train_ratio=0.8,
                                          n_valid_users=1000,
                                          n_test_users=1000,
                                          minimum_interaction=3,
                                          rand_state=None):
    """ Split given test / valid user records into subsets
    
    User records are splitted proportionally per user
    as same as `user_ratio_shuffle_split`.
    However, split is only made for randomly selected test / valid user population.
    
    Inputs:
        X (scipy.sparse.csr_matrix): user-item matrix
        train_ratio (float): ratio of training records per user
        n_valid_users (int): number of validation users
        n_test_users (int): number of testing users
        minimum_interaction (int): minimum interaction of user to be considered.
                                   if it's smaller than this,
                                   put all records to the training set
        rand_state (bool or int): random state seed number or None
    Returns:
        scipy.sparse.csr_matrix: training matrix
        scipy.sparse.csr_matrix: validation matrix
        scipy.sparse.csr_matrix: testing matrix
    """
    # first draw valid / test users
    rnd_idx = np.random.permutation(X.shape[0])

    valid_users = rnd_idx[:n_valid_users]
    test_users = rnd_idx[n_valid_users:n_valid_users + n_test_users]
    train_users = rnd_idx[n_valid_users + n_test_users:]
    
    # split records for valid / test users
    Xvl, Xvl_vl, Xvl_ts = user_ratio_shuffle_split(X[valid_users],
                                                   train_ratio,
                                                   0.5,  # valid_ratio
                                                   minimum_interaction,
                                                   rand_state)
    
    # merge them, as this scheme does not need within user validation set
    Xvl_ts = Xvl_vl + Xvl_ts
    
    Xts, Xts_vl, Xts_ts = user_ratio_shuffle_split(X[test_users],
                                                   train_ratio,
                                                   0.5, # valid ratio
                                                   minimum_interaction,
                                                   rand_state)
    Xts_ts = Xts_vl + Xts_ts  # merge
    
    # assign them back to the original data
    Xtr = X[train_users]
    Xtr_ = sp.vstack([Xvl, Xts, Xtr])
    Xts_ = sp.vstack([Xvl_ts, Xts_ts, Xtr])
    
    # un-shuffle
    reverse_idx = {j:i for i, j in enumerate(rnd_idx)}
    reverse_idx = [reverse_idx[i] for i in range(X.shape[0])]

    Xtr_ = Xtr_[reverse_idx]
    Xts_ = Xts_[reverse_idx]
    
    return Xtr_, Xts_, (train_users, valid_users, test_users)


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


def identity_check_sparse(X1, X2):
    """REALLY SIMPLE TEST FOR SPLITTING
    
    It does not verify the ratio / etc, but matching the entire entries
    reason for using `!=` rather than `==` is that `==` evaluate all the zero entry,
    which will not be benefitted from the sparsity.
    
    reference::
        https://stackoverflow.com/questions/30685024/check-if-two-scipy-sparse-csr-matrix-are-equal
    """
    return (X1 != X2).nnz == 0


def test_user_ratio_split_integrity(X, Xtr, Xvl, Xts):
    """ For testing purpose
    """
    Xhat = Xtr + Xvl + Xts
    return identity_check_sparse(X, Xhat)


def test_user_ratio_split_with_target_integrity(X, Xtr, Xts,
                                                train_users,
                                                valid_users,
                                                test_users):
    """ For testing purpose
    """
    train_check = identity_check_sparse(Xtr[train_users], X[train_users])
    valid_check = identity_check_sparse(Xtr[valid_users] + Xts[valid_users],
                                        X[valid_users])
    test_check = identity_check_sparse(Xtr[test_users] + Xts[test_users],
                                       X[test_users])
    
    return all([train_check, valid_check, test_check])