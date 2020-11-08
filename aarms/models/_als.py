from scipy import sparse as sp

import numpy as np
import numba as nb


# Atomic operations
# =============================================================================


def _compute_terms_wals_npy(A_, b_, val, ind, factors, covar, lmbda):
    """
    """
    if lmbda <= 0:
        return

    # prep data
    c = val + factors.dtype.type(0.0)
    vv = factors[ind].copy()

    # compute "b" in "Ax=b"
    b_ += lmbda * (vv.T @ c)

    # compute "A" in "Ax=b"
    A_ += lmbda * (covar + vv.T @ np.diag(c - 1) @ vv)


@nb.njit(
    [
        "void(f4[:, ::1], f4[::1], f4[::1], i4[::1], f4[:,::1], f4[:,::1], f4)",
        "void(f8[:, ::1], f8[::1], f8[::1], i4[::1], f8[:,::1], f8[:,::1], f8)",
    ],
    cache=True,
)
def _compute_terms_wals(A_, b_, val, ind, factors, covar, lmbda):
    """
    """
    if lmbda <= 0:
        return

    # prep data
    d = A_.shape[0]
    c = val + A_.dtype.type(0.0)
    vv = factors[ind].copy()

    # compute 'b'
    for f in range(d):
        for j in range(len(c)):
            b_[f] += lmbda * vv[j, f] * c[j]

    # compute 'A'
    for f in range(d):
        for q in range(f, d):
            A_[f, q] += lmbda * covar[f, q]
            for j in range(len(c)):
                A_[f, q] += lmbda * vv[j, f] * (c[j] - 1) * vv[j, q]

    for f in range(d):
        for q in range(f + 1, d):
            A_[q, f] = A_[f, q]


def _compute_terms_wals_cg_Ap_npy(Ap, val, ind, p, factors, covar, lmbda):
    """
    """
    if lmbda <= 0:
        return

    # prep data
    d = factors.shape[1]
    c = val + factors.dtype.type(0.0)
    vv = factors[ind].copy()

    Ap[:] += lmbda * ((covar @ p) + vv.T @ np.diag(c - 1) @ (vv @ p))


@nb.njit(
    [
        "void(f4[::1], f4[::1], i4[::1], f4[::1], f4[:,::1], f4[:,::1], f4)",
        "void(f8[::1], f8[::1], i4[::1], f8[::1], f8[:,::1], f8[:,::1], f8)",
    ],
    cache=True,
)
def _compute_terms_wals_cg_Ap(Ap, val, ind, p, factors, covar, lmbda):
    """
    """
    if lmbda <= 0:
        return

    # prep data
    d = factors.shape[1]
    c = val + factors.dtype.type(0.0)
    vv = factors[ind].copy()

    # compute 'Ap'
    for f in range(d):
        for q in range(d):
            Ap[f] += lmbda * covar[f, q] * p[q]

    for j in range(len(c)):
        vp = p.dtype.type(0.0)
        for f in range(d):
            vp += vv[j, f] * p[f]

        for f in range(d):
            Ap[f] += lmbda * vv[j, f] * (c[j] - 1) * vp


def _compute_terms_wals_cg_b_npy(b_, val, ind, factors, lmbda):
    """
    """
    if lmbda <= 0:
        return

    # prep data
    d = factors.shape[1]
    c = val + factors.dtype.type(0.0)
    vv = factors[ind].copy()

    b_[:] += lmbda * (vv.T @ c)


@nb.njit(
    [
        "void(f4[::1], f4[::1], i4[::1], f4[:, ::1], f4)",
        "void(f8[::1], f8[::1], i4[::1], f8[:, ::1], f8)",
    ],
    cache=True,
)
def _compute_terms_wals_cg_b(b_, val, ind, factors, lmbda):
    """
    """
    if lmbda <= 0:
        return

    # prep data
    d = factors.shape[1]
    c = val + factors.dtype.type(0.0)
    vv = factors[ind].copy()

    for j in range(len(c)):
        for f in range(d):
            b_[f] += lmbda * vv[j, f] * c[j]


def _compute_terms_dense_feat_cg_Ap_npy(Ap, p, lmbda):
    """
    """
    if lmbda <= 0:
        return

    Ap[:] += lmbda * p


@nb.njit(["void(f4[::1], f4[::1], f4)", "void(f8[::1], f8[::1], f8)"], cache=True)
def _compute_terms_dense_feat_cg_Ap(Ap, p, lmbda):
    """
    """
    if lmbda <= 0:
        return

    Ap[:] += lmbda * p


def _compute_terms_dense_feat_cg_b_npy(b_, a, weight, lmbda):
    """
    """
    if lmbda <= 0:
        return

    b_[:] += lmbda * (weight.T @ a)


@nb.njit(
    [
        "void(f4[::1], f4[::1], f4[:, ::1], f4)",
        "void(f8[::1], f8[::1], f8[:, ::1], f8)",
    ],
    cache=True,
)
def _compute_terms_dense_feat_cg_b(b_, a, weight, lmbda):
    """
    """
    if lmbda <= 0:
        return

    b_[:] += lmbda * (weight.T @ a)


def _compute_terms_sparse_feat_cg_Ap_npy(Ap, p, lmbda):
    """
    """
    _compute_terms_dense_feat_cg_Ap_npy(Ap, p, lmbda)


@nb.njit(["void(f4[::1], f4[::1], f4)", "void(f8[::1], f8[::1], f8)"], cache=True)
def _compute_terms_sparse_feat_cg_Ap(Ap, p, lmbda):
    """
    """
    _compute_terms_dense_feat_cg_Ap(Ap, p, lmbda)


def _compute_terms_sparse_feat_cg_b_npy(b_, val, ind, weight, lmbda):
    """
    """
    if lmbda <= 0:
        return

    # prep data
    s = val + weight.dtype.type(0.0)
    ww = weight[ind].copy()

    # compute
    b_ += lmbda * (ww.T @ s)


@nb.njit(
    [
        "void(f4[::1], f4[::1], i4[::1], f4[:, ::1], f4)",
        "void(f8[::1], f8[::1], i4[::1], f8[:, ::1], f8)",
    ],
    cache=True,
)
def _compute_terms_sparse_feat_cg_b(b_, val, ind, weight, lmbda):
    """
    """
    if lmbda <= 0:
        return

    # prep data
    s = val + weight.dtype.type(0.0)
    ww = weight[ind].copy()

    # compute
    b_ += lmbda * (ww.T @ s)


def _compute_terms_dense_feat_npy(A_, b_, a, weight, lmbda):
    """
    """
    if lmbda <= 0:
        return

    d = A_.shape[0]
    A_ += lmbda * np.eye(d, dtype=A_.dtype)
    b_ += lmbda * (weight.T @ a)


@nb.njit(
    [
        "void(f4[:, ::1], f4[::1], f4[::1], f4[:, ::1], f4)",
        "void(f8[:, ::1], f8[::1], f8[::1], f8[:, ::1], f8)",
    ],
    cache=True,
)
def _compute_terms_dense_feat(A_, b_, a, weight, lmbda):
    """
    """
    if lmbda <= 0:
        return

    d = A_.shape[0]
    A_ += lmbda * np.eye(d, dtype=A_.dtype)
    b_ += lmbda * (weight.T @ a)


def _compute_terms_sparse_feat_npy(A_, b_, val, ind, weight, lmbda):
    """
    """
    if lmbda <= 0:
        return

    d = A_.shape[0]

    # prep data
    s = val + weight.dtype.type(0.0)
    ww = weight[ind].copy()

    # compute
    A_ += lmbda * np.eye(d, dtype=A_.dtype)
    b_ += lmbda * (ww.T @ s)


@nb.njit(
    [
        "void(f4[:, ::1], f4[::1], f4[::1], i4[::1], f4[:, ::1], f4)",
        "void(f8[:, ::1], f8[::1], f8[::1], i4[::1], f8[:, ::1], f8)",
    ],
    cache=True,
)
def _compute_terms_sparse_feat(A_, b_, val, ind, weight, lmbda):
    """
    """
    if lmbda <= 0:
        return

    d = A_.shape[0]

    # prep data
    s = val + weight.dtype.type(0.0)
    ww = weight[ind].copy()

    # compute
    A_ += lmbda * np.eye(d, dtype=A_.dtype)
    b_ += lmbda * (ww.T @ s)


@nb.njit(
    [
        "Tuple((f4[::1], f4[::1], f4[::1], f4))("
        "i8, f4[::1], f4[::1], f4[::1], f4[::1], f4, f8"
        ")",
        "Tuple((f8[::1], f8[::1], f8[::1], f8))("
        "i8, f8[::1], f8[::1], f8[::1], f8[::1], f8, f8"
        ")",
    ],
    cache=True,
)
def _cg_update(d, x, Ap, p, r, rsold, eps):
    """
    """
    # standard CG update
    pAp = r.dtype.type(0.0)
    for f in range(d):
        pAp += p[f] * Ap[f]
    # alpha = rsold / pAp
    alpha = rsold / max(pAp, eps)
    for f in range(d):
        x[f] += alpha * p[f]
        r[f] -= alpha * Ap[f]

    rsnew = r.dtype.type(0.0)
    for f in range(d):
        rsnew += r[f] ** 2
    p = r + (rsnew / rsold) * p
    rsold = rsnew

    return x, p, r, rsold


def _cg_Ap_aarms_npy(
    Ap,
    p,
    val_x,
    ind_x,
    val_y,
    ind_y,
    val_g,
    ind_g,
    V,
    U_tmp,
    P,
    VV,
    UU,
    PP,
    lmbda_y,
    lmbda_g,
    lmbda_s,
    lmbda_a,
):
    """
    """
    _compute_terms_wals_cg_Ap_npy(Ap, val_x, ind_x, p, V, VV, 1)
    _compute_terms_wals_cg_Ap_npy(Ap, val_y, ind_y, p, U_tmp, UU, lmbda_y)
    _compute_terms_wals_cg_Ap_npy(Ap, val_g, ind_g, p, P, PP, lmbda_g)
    _compute_terms_sparse_feat_cg_Ap_npy(Ap, p, lmbda_s)
    _compute_terms_dense_feat_cg_Ap_npy(Ap, p, lmbda_a)


@nb.njit(
    [
        "void("
        "f4[::1], f4[::1], "
        "f4[::1], i4[::1], f4[::1], i4[::1], f4[::1], i4[::1], "
        "f4[:,::1], f4[:,::1], f4[:, ::1], f4[:,::1], f4[:,::1], f4[:,::1], "
        "f4, f4, f4, f4"
        ")",
        "void("
        "f8[::1], f8[::1], "
        "f8[::1], i4[::1], f8[::1], i4[::1], f8[::1], i4[::1], "
        "f8[:,::1], f8[:,::1], f8[:, ::1], f8[:,::1], f8[:,::1], f8[:,::1], "
        "f8, f8, f8, f8"
        ")",
    ],
    cache=True,
)
def _cg_Ap_aarms(
    Ap,
    p,
    val_x,
    ind_x,
    val_y,
    ind_y,
    val_g,
    ind_g,
    V,
    U_tmp,
    P,
    VV,
    UU,
    PP,
    lmbda_y,
    lmbda_g,
    lmbda_s,
    lmbda_a,
):
    """
    """
    _compute_terms_wals_cg_Ap(Ap, val_x, ind_x, p, V, VV, 1)
    _compute_terms_wals_cg_Ap(Ap, val_y, ind_y, p, U_tmp, UU, lmbda_y)
    _compute_terms_wals_cg_Ap(Ap, val_g, ind_g, p, P, PP, lmbda_g)
    _compute_terms_sparse_feat_cg_Ap(Ap, p, lmbda_s)
    _compute_terms_dense_feat_cg_Ap(Ap, p, lmbda_a)


def _cg_b_aarms_npy(
    b_,
    a,
    val_x,
    ind_x,
    val_y,
    ind_y,
    val_g,
    ind_g,
    val_s,
    ind_s,
    V,
    U_tmp,
    P,
    W_a,
    W_s,
    VV,
    UU,
    PP,
    lmbda_y,
    lmbda_g,
    lmbda_s,
    lmbda_a,
):
    """
    """
    _compute_terms_wals_cg_b_npy(b_, val_x, ind_x, V, 1)
    _compute_terms_wals_cg_b_npy(b_, val_y, ind_y, U_tmp, lmbda_y)
    _compute_terms_wals_cg_b_npy(b_, val_g, ind_g, P, lmbda_g)
    _compute_terms_sparse_feat_cg_b_npy(b_, val_s, ind_s, W_s, lmbda_s)
    _compute_terms_dense_feat_cg_b_npy(b_, a, W_a, lmbda_a)


@nb.njit(
    [
        "void("
        "f4[::1], f4[::1], "
        "f4[::1], i4[::1], f4[::1], i4[::1], f4[::1], i4[::1], f4[::1], i4[::1], "
        "f4[:,::1], f4[:,::1], f4[:, ::1], f4[:,::1], "
        "f4[:,::1], f4[:,::1], f4[:, ::1], f4[:, ::1], "
        "f4, f4, f4, f4"
        ")",
        "void("
        "f8[::1], f8[::1], "
        "f8[::1], i4[::1], f8[::1], i4[::1], f8[::1], i4[::1], f8[::1], i4[::1], "
        "f8[:,::1], f8[:,::1], f8[:, ::1], f8[:,::1], "
        "f8[:,::1], f8[:,::1], f8[:, ::1], f8[:, ::1], "
        "f8, f8, f8, f8"
        ")",
    ],
    cache=True,
)
def _cg_b_aarms(
    b_,
    a,
    val_x,
    ind_x,
    val_y,
    ind_y,
    val_g,
    ind_g,
    val_s,
    ind_s,
    V,
    U_tmp,
    P,
    W_a,
    W_s,
    VV,
    UU,
    PP,
    lmbda_y,
    lmbda_g,
    lmbda_s,
    lmbda_a,
):
    """
    """
    _compute_terms_wals_cg_b(b_, val_x, ind_x, V, 1)
    _compute_terms_wals_cg_b(b_, val_y, ind_y, U_tmp, lmbda_y)
    _compute_terms_wals_cg_b(b_, val_g, ind_g, P, lmbda_g)
    _compute_terms_sparse_feat_cg_b(b_, val_s, ind_s, W_s, lmbda_s)
    _compute_terms_dense_feat_cg_b(b_, a, W_a, lmbda_a)


# Some utilities
# =============================================================================


@nb.njit(
    [
        "Tuple((f4[::1], i4[::1], i8))(i8, f4[::1], i4[::1], i4[::1], f4)",
        "Tuple((f8[::1], i4[::1], i8))(i8, f8[::1], i4[::1], i4[::1], f8)",
    ],
    cache=True,
)
def fetch(u, data, indices, indptr, lmbda):
    """
    """
    is_skip = 1
    if lmbda > 0:
        u0, u1 = indptr[u], indptr[u + 1]
        val, ind = data[u0:u1], indices[u0:u1]
        if u1 > u0:
            is_skip = 0
    else:
        val = np.empty(0, dtype=data.dtype)
        ind = np.empty(0, dtype=np.int32)

    return val, ind, is_skip


# Solvers: mid-level opererations for partial update per entity
# =============================================================================


def partial_wals_vanilla_npy(u, val, ind, U, V, VV, lmbda):
    """
    """
    d = U.shape[1]
    A_ = lmbda * np.eye(d, dtype=U.dtype)  # already add ridge term
    b_ = np.zeros((d,), dtype=U.dtype)
    _compute_terms_wals(A_, b_, val, ind, V, VV, 1)

    # solve system to update factor
    U[u] = np.linalg.solve(A_, b_)


@nb.njit(
    [
        "void(i8, f4[::1], i4[::1], f4[:,::1], f4[:,::1], f4[:,::1], f4)",
        "void(i8, f8[::1], i4[::1], f8[:,::1], f8[:,::1], f8[:,::1], f8)",
    ],
    cache=True,
)
def partial_wals_vanilla(u, val, ind, U, V, VV, lmbda):
    """
    """
    d = U.shape[1]
    A_ = lmbda * np.eye(d, dtype=U.dtype)
    b_ = np.zeros((d,), dtype=U.dtype)
    _compute_terms_wals(A_, b_, val, ind, V, VV, 1)
    # solve
    U[u] = np.linalg.solve(A_, b_)


def partial_wals_vanilla_cg_npy(u, val, ind, U, V, VV, lmbda, cg_steps=5, eps=1e-20):
    """
    """
    d = U.shape[1]
    u_ = U[u].copy()  # initial value p0

    # compute residual
    # first compute "Ap"
    r_ = lmbda * u_.copy()  # residual (b - Ap)
    _compute_terms_wals_cg_Ap_npy(r_, val, ind, u_, V, VV, 1)

    # flip the sign
    r_ *= -1

    # add "b"
    _compute_terms_wals_cg_b_npy(r_, val, ind, V, 1)

    p = r_.copy()
    rsold = r_.dtype.type(0.0)
    for f in range(d):
        rsold += r_[f] ** 2
    if rsold ** 0.5 < eps:
        return

    for it in range(cg_steps):
        Ap = lmbda * p.copy()
        _compute_terms_wals_cg_Ap_npy(Ap, val, ind, p, V, VV, 1)

        # update
        u_, p, r_, rsold = _cg_update(d, u_, Ap, p, r_, rsold, eps)

        # check convergence
        if rsold ** 0.5 < eps:
            break

    # update
    U[u] = u_


@nb.njit(
    [
        "void(i8, f4[::1], i4[::1], f4[:,::1], f4[:,::1], f4[:,::1], f4, i8, f8)",
        "void(i8, f8[::1], i4[::1], f8[:,::1], f8[:,::1], f8[:,::1], f8, i8, f8)",
    ],
    cache=True,
)
def partial_wals_vanilla_cg(u, val, ind, U, V, VV, lmbda, cg_steps=5, eps=1e-20):
    """
    """
    d = U.shape[1]
    u_ = U[u].copy()  # initial value p0

    # compute residual
    # first compute "Ap"
    r_ = lmbda * u_.copy()  # residual (b - Ap)
    _compute_terms_wals_cg_Ap(r_, val, ind, u_, V, VV, 1)

    # flip the sign
    r_ *= -1

    # add "b"
    _compute_terms_wals_cg_b(r_, val, ind, V, 1)

    p = r_.copy()
    rsold = r_.dtype.type(0.0)
    for f in range(d):
        rsold += r_[f] ** 2
    if rsold ** 0.5 < eps:
        return

    for it in range(cg_steps):
        Ap = lmbda * p.copy()
        _compute_terms_wals_cg_Ap(Ap, val, ind, p, V, VV, 1)

        # update
        u_, p, r_, rsold = _cg_update(d, u_, Ap, p, r_, rsold, eps)

        # check convergence
        if rsold ** 0.5 < eps:
            break

    # update
    U[u] = u_


def partial_wals_npy(
    u,  # target index
    val_x,
    ind_x,  # data slices
    val_y,
    ind_y,  # data slices
    val_g,
    ind_g,  # data slices
    val_s,
    ind_s,  # data (sparse feat) slices
    U,
    V,
    U_tmp,
    P,
    W_a,
    W_s,
    A,  # factors & dense feature
    VV,
    UU,
    PP,  # pre-computed covariances
    lmbda_y,
    lmbda_g,
    lmbda_a,
    lmbda_s,  # loss weights
    lmbda,
):  # ridge coefficient
    """
    """
    d = U.shape[1]
    if A.size == 0:
        a = U[u].copy()  # it's dummy and will not be used
    else:
        a = A[u].copy()
    A_ = lmbda * np.eye(d, dtype=U.dtype)  # already add ridge term
    b_ = np.zeros((d,), dtype=U.dtype)

    # add terms to A_ and b_
    _compute_terms_wals_npy(A_, b_, val_x, ind_x, V, VV, 1)
    _compute_terms_wals_npy(A_, b_, val_y, ind_y, U_tmp, UU, lmbda_y)
    _compute_terms_wals_npy(A_, b_, val_g, ind_g, P, PP, lmbda_g)
    _compute_terms_sparse_feat_npy(A_, b_, val_s, ind_s, W_s, lmbda_s)
    _compute_terms_dense_feat_npy(A_, b_, a, W_a, lmbda_a)

    # solve system to update factor
    U[u] = np.linalg.solve(A_, b_)


@nb.njit(
    [
        "void("
        "i8, "
        "f4[::1], i4[::1], f4[::1], i4[::1], f4[::1], i4[::1], f4[::1], i4[::1], "
        "f4[:,::1], f4[:,::1], f4[:,::1], f4[:,::1], f4[:,::1], "
        "f4[:,::1], f4[:,::1], f4[:,::1], f4[:,::1], f4[:,::1], "
        "f4, f4, f4, f4, f4"
        ")",
        "void("
        "i8, "
        "f8[::1], i4[::1], f8[::1], i4[::1], f8[::1], i4[::1], f8[::1], i4[::1], "
        "f8[:,::1], f8[:,::1], f8[:,::1], f8[:,::1], f8[:,::1], "
        "f8[:,::1], f8[:,::1], f8[:,::1], f8[:,::1], f8[:,::1], "
        "f8, f8, f8, f8, f8"
        ")",
    ],
    cache=True,
)
def partial_wals(
    u,
    val_x,
    ind_x,
    val_y,
    ind_y,
    val_g,
    ind_g,
    val_s,
    ind_s,
    U,
    V,
    U_tmp,
    P,
    W_a,
    W_s,
    A,
    VV,
    UU,
    PP,
    lmbda_y,
    lmbda_g,
    lmbda_a,
    lmbda_s,
    lmbda,
):
    """
    """
    d = U.shape[1]
    if A.size == 0:
        a = U[u].copy()  # it's dummy and will not be used
    else:
        a = A[u].copy()
    A_ = lmbda * np.eye(d, dtype=U.dtype)
    b_ = np.zeros((d,), dtype=U.dtype)

    # add terms to A_ and b_
    _compute_terms_wals(A_, b_, val_x, ind_x, V, VV, 1)
    _compute_terms_wals(A_, b_, val_y, ind_y, U_tmp, UU, lmbda_y)
    _compute_terms_wals(A_, b_, val_g, ind_g, P, PP, lmbda_g)
    _compute_terms_sparse_feat(A_, b_, val_s, ind_s, W_s, lmbda_s)
    _compute_terms_dense_feat(A_, b_, a, W_a, lmbda_a)

    # solve system to update factor
    U[u] = np.linalg.solve(A_, b_)


def partial_wals_cg_npy(
    u,
    val_x,
    ind_x,
    val_y,
    ind_y,
    val_g,
    ind_g,
    val_s,
    ind_s,
    U,
    V,
    U_tmp,
    P,
    W_a,
    W_s,
    A,
    VV,
    UU,
    PP,
    lmbda_y,
    lmbda_g,
    lmbda_a,
    lmbda_s,
    lmbda,
    cg_steps=5,
    eps=1e-20,
):
    """
    """
    d = U.shape[1]
    if A.size == 0:
        a = U[u].copy()  # it's dummy and will not be used
    else:
        a = A[u].copy()
    u_ = U[u].copy()  # initial value p0

    # compute residual
    # first compute "Ap"
    r_ = lmbda * u_.copy()  # residual (b - Ap)
    _cg_Ap_aarms_npy(
        r_,
        u_,
        val_x,
        ind_x,
        val_y,
        ind_y,
        val_g,
        ind_g,
        V,
        U_tmp,
        P,
        VV,
        UU,
        PP,
        lmbda_y,
        lmbda_g,
        lmbda_s,
        lmbda_a,
    )

    # flip the sign
    r_ *= -1

    # add "b"
    _cg_b_aarms_npy(
        r_,
        a,
        val_x,
        ind_x,
        val_y,
        ind_y,
        val_g,
        ind_g,
        val_s,
        ind_s,
        V,
        U_tmp,
        P,
        W_a,
        W_s,
        VV,
        UU,
        PP,
        lmbda_y,
        lmbda_g,
        lmbda_s,
        lmbda_a,
    )

    p = r_.copy()
    rsold = r_.dtype.type(0.0)
    for f in range(d):
        rsold += r_[f] ** 2
    if rsold ** 0.5 < eps:
        return

    for it in range(cg_steps):
        Ap = lmbda * p.copy()
        _cg_Ap_aarms_npy(
            Ap,
            p,
            val_x,
            ind_x,
            val_y,
            ind_y,
            val_g,
            ind_g,
            V,
            U_tmp,
            P,
            VV,
            UU,
            PP,
            lmbda_y,
            lmbda_g,
            lmbda_s,
            lmbda_a,
        )

        # update
        u_, p, r_, rsold = _cg_update(d, u_, Ap, p, r_, rsold, eps)

        # check convergence
        if rsold ** 0.5 < eps:
            break

    # update
    U[u] = u_


@nb.njit(
    [
        "void("
        "i8, "
        "f4[::1], i4[::1], f4[::1], i4[::1], f4[::1], i4[::1], f4[::1], i4[::1], "
        "f4[:,::1], f4[:,::1], f4[:,::1], f4[:,::1], f4[:,::1], "
        "f4[:,::1], f4[:,::1], f4[:,::1], f4[:,::1], f4[:,::1], "
        "f4, f4, f4, f4, f4, "
        "i8, f8"
        ")",
        "void("
        "i8, "
        "f8[::1], i4[::1], f8[::1], i4[::1], f8[::1], i4[::1], f8[::1], i4[::1], "
        "f8[:,::1], f8[:,::1], f8[:,::1], f8[:,::1], f8[:,::1], "
        "f8[:,::1], f8[:,::1], f8[:,::1], f8[:,::1], f8[:,::1], "
        "f8, f8, f8, f8, f8, "
        "i8, f8"
        ")",
    ],
    cache=True,
)
def partial_wals_cg(
    u,
    val_x,
    ind_x,
    val_y,
    ind_y,
    val_g,
    ind_g,
    val_s,
    ind_s,
    U,
    V,
    U_tmp,
    P,
    W_a,
    W_s,
    A,
    VV,
    UU,
    PP,
    lmbda_y,
    lmbda_g,
    lmbda_a,
    lmbda_s,
    lmbda,
    cg_steps=5,
    eps=1e-20,
):
    """
    """
    d = U.shape[1]
    if A.size == 0:
        a = U[u].copy()  # it's dummy and will not be used
    else:
        a = A[u].copy()
    u_ = U[u].copy()  # initial value p0

    # compute residual
    # first compute "Ap"
    r_ = lmbda * u_.copy()  # residual (b - Ap)
    _cg_Ap_aarms(
        r_,
        u_,
        val_x,
        ind_x,
        val_y,
        ind_y,
        val_g,
        ind_g,
        V,
        U_tmp,
        P,
        VV,
        UU,
        PP,
        lmbda_y,
        lmbda_g,
        lmbda_s,
        lmbda_a,
    )

    # flip the sign
    r_ *= -1

    # add "b"
    _cg_b_aarms(
        r_,
        a,
        val_x,
        ind_x,
        val_y,
        ind_y,
        val_g,
        ind_g,
        val_s,
        ind_s,
        V,
        U_tmp,
        P,
        W_a,
        W_s,
        VV,
        UU,
        PP,
        lmbda_y,
        lmbda_g,
        lmbda_s,
        lmbda_a,
    )

    p = r_.copy()
    rsold = r_.dtype.type(0.0)
    for f in range(d):
        rsold += r_[f] ** 2
    if rsold ** 0.5 < eps:
        return

    for it in range(cg_steps):
        Ap = lmbda * p.copy()
        _cg_Ap_aarms(
            Ap,
            p,
            val_x,
            ind_x,
            val_y,
            ind_y,
            val_g,
            ind_g,
            V,
            U_tmp,
            P,
            VV,
            UU,
            PP,
            lmbda_y,
            lmbda_g,
            lmbda_s,
            lmbda_a,
        )

        # update
        u_, p, r_, rsold = _cg_update(d, u_, Ap, p, r_, rsold, eps)

        # check convergence
        if rsold ** 0.5 < eps:
            break

    # update
    U[u] = u_


# Highest level operations for entire factor update
# =============================================================================


def update_user_npy(
    data_x,
    indices_x,
    indptr_x,
    data_y,
    indices_y,
    indptr_y,
    data_g,
    indices_g,
    indptr_g,
    data_s,
    indices_s,
    indptr_s,
    U,
    V,
    P,
    W_a,
    W_s,
    A,
    lmbda_y,
    lmbda_g,
    lmbda_a,
    lmbda_s,
    lmbda,
    solver="cg",
    cg_steps=5,
    eps=1e-20,
):
    """
    """
    # setup some vars and pre-computation
    N = U.shape[0]
    U_tmp = U.copy()
    VV = V.T @ V
    UU = U.T @ U
    PP = P.T @ P
    rnd_idx = np.random.permutation(N)

    # run!
    for n in range(N):
        u = rnd_idx[n]
        val_x, ind_x, skip_x = fetch(u, data_x, indices_x, indptr_x, 1)
        val_y, ind_y, skip_y = fetch(u, data_y, indices_y, indptr_y, lmbda_y)
        val_g, ind_g, skip_g = fetch(u, data_g, indices_g, indptr_g, lmbda_g)
        val_s, ind_s, skip_s = fetch(u, data_s, indices_s, indptr_s, lmbda_s)
        if skip_x * skip_y * skip_g * skip_s == 1:
            continue

        if solver == "lu":
            partial_wals_npy(
                u,
                val_x,
                ind_x,
                val_y,
                ind_y,
                val_g,
                ind_g,
                val_s,
                ind_s,
                U,
                V,
                U_tmp,
                P,
                W_a,
                W_s,
                A,
                VV,
                UU,
                PP,
                lmbda_y,
                lmbda_g,
                lmbda_a,
                lmbda_s,
                lmbda,
            )
        elif solver == "cg":
            partial_wals_cg_npy(
                u,
                val_x,
                ind_x,
                val_y,
                ind_y,
                val_g,
                ind_g,
                val_s,
                ind_s,
                U,
                V,
                U_tmp,
                P,
                W_a,
                W_s,
                A,
                VV,
                UU,
                PP,
                lmbda_y,
                lmbda_g,
                lmbda_a,
                lmbda_s,
                lmbda,
                cg_steps=cg_steps,
                eps=eps,
            )


@nb.njit(
    [
        nb.void(
            nb.f4[::1],
            nb.i4[::1],
            nb.i4[::1],
            nb.f4[::1],
            nb.i4[::1],
            nb.i4[::1],
            nb.f4[::1],
            nb.i4[::1],
            nb.i4[::1],
            nb.f4[::1],
            nb.i4[::1],
            nb.i4[::1],
            nb.f4[:, ::1],
            nb.f4[:, ::1],
            nb.f4[:, ::1],
            nb.f4[:, ::1],
            nb.f4[:, ::1],
            nb.f4[:, ::1],
            nb.f4,
            nb.f4,
            nb.f4,
            nb.f4,
            nb.f4,
            nb.types.unicode_type,
            nb.i8,
            nb.f8,
        ),
        nb.void(
            nb.f8[::1],
            nb.i4[::1],
            nb.i4[::1],
            nb.f8[::1],
            nb.i4[::1],
            nb.i4[::1],
            nb.f8[::1],
            nb.i4[::1],
            nb.i4[::1],
            nb.f8[::1],
            nb.i4[::1],
            nb.i4[::1],
            nb.f8[:, ::1],
            nb.f8[:, ::1],
            nb.f8[:, ::1],
            nb.f8[:, ::1],
            nb.f8[:, ::1],
            nb.f8[:, ::1],
            nb.f8,
            nb.f8,
            nb.f8,
            nb.f8,
            nb.f8,
            nb.types.unicode_type,
            nb.i8,
            nb.f8,
        ),
    ],
    nogil=True,
    parallel=True,
    cache=True,
)
def update_user(
    data_x,
    indices_x,
    indptr_x,
    data_y,
    indices_y,
    indptr_y,
    data_g,
    indices_g,
    indptr_g,
    data_s,
    indices_s,
    indptr_s,
    U,
    V,
    P,
    W_a,
    W_s,
    A,
    lmbda_y,
    lmbda_g,
    lmbda_a,
    lmbda_s,
    lmbda,
    solver="cg",
    cg_steps=5,
    eps=1e-20,
):
    """
    """
    # setup some vars and pre-computation
    N = U.shape[0]
    U_tmp = U.copy()
    VV = V.T @ V
    UU = U.T @ U
    PP = P.T @ P
    rnd_idx = np.random.permutation(N)

    # run!
    for n in nb.prange(N):
        u = rnd_idx[n]
        val_x, ind_x, skip_x = fetch(u, data_x, indices_x, indptr_x, 1)
        val_y, ind_y, skip_y = fetch(u, data_y, indices_y, indptr_y, lmbda_y)
        val_g, ind_g, skip_g = fetch(u, data_g, indices_g, indptr_g, lmbda_g)
        val_s, ind_s, skip_s = fetch(u, data_s, indices_s, indptr_s, lmbda_s)
        if skip_x * skip_y * skip_g * skip_s == 1:
            continue

        if solver == "lu":
            partial_wals(
                u,
                val_x,
                ind_x,
                val_y,
                ind_y,
                val_g,
                ind_g,
                val_s,
                ind_s,
                U,
                V,
                U_tmp,
                P,
                W_a,
                W_s,
                A,
                VV,
                UU,
                PP,
                lmbda_y,
                lmbda_g,
                lmbda_a,
                lmbda_s,
                lmbda,
            )

        elif solver == "cg":
            partial_wals_cg(
                u,
                val_x,
                ind_x,
                val_y,
                ind_y,
                val_g,
                ind_g,
                val_s,
                ind_s,
                U,
                V,
                U_tmp,
                P,
                W_a,
                W_s,
                A,
                VV,
                UU,
                PP,
                lmbda_y,
                lmbda_g,
                lmbda_a,
                lmbda_s,
                lmbda,
                cg_steps=cg_steps,
                eps=eps,
            )


def update_side_npy(
    data_g,
    indices_g,
    indptr_g,
    P,
    U,
    lmbda_g,
    lmbda,
    solver="lu",
    cg_steps=5,
    eps=1e-20,
):
    """
    """
    if lmbda_g <= 0:
        return

    # setup some vars and pre-computation
    L = P.shape[0]
    UU = U.T @ U
    new_lmbda = lmbda_g / max(lmbda, eps)
    rnd_idx = np.random.permutation(L)

    # run!
    for n in range(L):
        l = rnd_idx[n]
        val_g, ind_g, skip_g = fetch(l, data_g, indices_g, indptr_g, lmbda_g)
        if skip_g == 1:
            continue

        if solver == "lu":
            partial_wals_vanilla_npy(l, val_g, ind_g, P, U, UU, new_lmbda)

        elif solver == "cg":
            partial_wals_vanilla_cg_npy(
                l, val_g, ind_g, P, U, UU, new_lmbda, cg_steps=cg_steps, eps=eps
            )


# TODO: this is super uggly, so yeah specs sould be moved to somehwere else for sure
@nb.njit(
    [
        nb.void(
            nb.f4[::1],
            nb.i4[::1],
            nb.i4[::1],
            nb.f4[:, ::1],
            nb.f4[:, ::1],
            nb.f4,
            nb.f4,
            nb.types.unicode_type,
            nb.i8,
            nb.f8,
        ),
        nb.void(
            nb.f8[::1],
            nb.i4[::1],
            nb.i4[::1],
            nb.f8[:, ::1],
            nb.f8[:, ::1],
            nb.f8,
            nb.f8,
            nb.types.unicode_type,
            nb.i8,
            nb.f8,
        ),
    ],
    cache=True,
)
def update_side(
    data_g,
    indices_g,
    indptr_g,
    P,
    U,
    lmbda_g,
    lmbda,
    solver="cg",
    cg_steps=5,
    eps=1e-20,
):
    """
    """
    if lmbda_g <= 0:
        return

    # setup some vars and pre-computation
    L = P.shape[0]
    UU = U.T @ U
    new_lmbda = lmbda_g / max(lmbda, eps)
    rnd_idx = np.random.permutation(L)

    # run!
    for n in nb.prange(L):
        l = rnd_idx[n]
        val_g, ind_g, skip_g = fetch(l, data_g, indices_g, indptr_g, lmbda_g)
        if skip_g == 1:
            continue

        if solver == "lu":
            partial_wals_vanilla(l, val_g, ind_g, P, U, UU, new_lmbda)
        elif solver == "cg":
            partial_wals_vanilla_cg(
                l, val_g, ind_g, P, U, UU, new_lmbda, cg_steps=cg_steps, eps=eps
            )


@nb.njit(
    [
        "void(f4[:,::1], f4[:,::1], f4[:,::1], f4, f4, f4)",
        "void(f8[:,::1], f8[:,::1], f8[:,::1], f8, f8, f8)",
    ],
    cache=True,
)
def update_weight_dense(U, A, W_a, lmbda_a, lmbda, eps):
    """
    """
    if lmbda_a <= 0:
        return

    new_lmbda = lmbda_a / max(lmbda, eps)
    d = A.shape[1]
    A_ = A.T @ A + new_lmbda * np.eye(d, dtype=W_a.dtype)
    B_ = A.T @ U
    W_a[:, :] = np.linalg.solve(A_, B_)
