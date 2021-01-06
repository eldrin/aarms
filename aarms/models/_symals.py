from scipy import sparse as sp

import numpy as np
import numba as nb

from ._als import (fetch,
                   partial_wals_npy, partial_wals_cg_npy,
                   partial_wals, partial_wals_cg)


def update_entity_npy(
    data_x, indices_x, indptr_x,
    data_g, indices_g, indptr_g,
    data_s, indices_s, indptr_s,
    U, P, W_a, W_s, A,
    lmbda_g, lmbda_a, lmbda_s, lmbda,
    is_smp_exp=False, is_smp_exp_g=False,
    solver="cg", cg_steps=5, eps=1e-20,
):
    """
    """
    # setup some vars and pre-computation
    N = U.shape[0]
    U_tmp = U.copy()
    UU = U.T @ U
    PP = P.T @ P

    # prepare some dummies for Y term
    val_y = np.array([0], dtype=data_x.dtype)
    ind_y = np.array([0], dtype=indices_x.dtype)
    _ = np.array([[0]], dtype=U.dtype)  # dummy

    # randomize order
    rnd_idx = np.random.permutation(N)

    # run!
    for n in range(N):
        u = rnd_idx[n]
        val_x, ind_x, skip_x = fetch(u, data_x, indices_x, indptr_x, 1)
        val_g, ind_g, skip_g = fetch(u, data_g, indices_g, indptr_g, lmbda_g)
        val_s, ind_s, skip_s = fetch(u, data_s, indices_s, indptr_s, lmbda_s)
        if skip_x * skip_g * skip_s == 1:
            continue

        if solver == "lu":
            partial_wals_npy(
                u, val_x, ind_x, val_y, ind_y, val_g, ind_g, val_s, ind_s,
                U, U_tmp, _, P, W_a, W_s, A, UU, _, PP,
                -1, lmbda_g, lmbda_a, lmbda_s, lmbda,
                is_smp_exp, False, is_smp_exp_g
            )
        elif solver == "cg":
            partial_wals_cg_npy(
                u, val_x, ind_x, val_y, ind_y, val_g, ind_g, val_s, ind_s,
                U, U_tmp, _, P, W_a, W_s, A, UU, _, PP,
                -1, lmbda_g, lmbda_a, lmbda_s, lmbda,
                is_smp_exp, False, is_smp_exp_g,
                cg_steps=cg_steps, eps=eps,
            )


@nb.njit(
    [
        nb.void(
            nb.f4[::1], nb.i4[::1], nb.i4[::1],
            nb.f4[::1], nb.i4[::1], nb.i4[::1],
            nb.f4[::1], nb.i4[::1], nb.i4[::1],
            nb.f4[:,::1], nb.f4[:,::1], nb.f4[:,::1], nb.f4[:,::1], nb.f4[:,::1],
            nb.f4, nb.f4, nb.f4, nb.f4, nb.b1, nb.b1,
            nb.types.unicode_type, nb.i8, nb.f8,
        ),
        nb.void(
            nb.f8[::1], nb.i4[::1], nb.i4[::1],
            nb.f8[::1], nb.i4[::1], nb.i4[::1],
            nb.f8[::1], nb.i4[::1], nb.i4[::1],
            nb.f8[:,::1], nb.f8[:,::1], nb.f8[:,::1], nb.f8[:,::1], nb.f8[:,::1],
            nb.f8, nb.f8, nb.f8, nb.f8, nb.b1, nb.b1,
            nb.types.unicode_type, nb.i8, nb.f8,
        ),
    ],
    nogil=True, parallel=True, cache=True,
)
def update_entity(
    data_x, indices_x, indptr_x,
    data_g, indices_g, indptr_g,
    data_s, indices_s, indptr_s,
    U, P, W_a, W_s, A,
    lmbda_g, lmbda_a, lmbda_s, lmbda,
    is_smp_exp=False, is_smp_exp_g=False,
    solver="cg", cg_steps=5, eps=1e-20,
):
    """
    """
    # setup some vars and pre-computation
    N = U.shape[0]
    U_tmp = U.copy()
    UU = U.T @ U
    PP = P.T @ P

    # prepare some dummies for Y term
    val_y = np.array([0], dtype=data_x.dtype)
    ind_y = np.array([0], dtype=np.int32)
    _ = np.array([[0]], dtype=U.dtype)  # dummy

    # randomize order
    rnd_idx = np.random.permutation(N)

    # run!
    for n in nb.prange(N):
        u = rnd_idx[n]
        val_x, ind_x, skip_x = fetch(u, data_x, indices_x, indptr_x, 1)
        val_g, ind_g, skip_g = fetch(u, data_g, indices_g, indptr_g, lmbda_g)
        val_s, ind_s, skip_s = fetch(u, data_s, indices_s, indptr_s, lmbda_s)
        if skip_x * skip_g * skip_s == 1:
            continue

        if solver == "lu":
            partial_wals(
                u, val_x, ind_x, val_y, ind_y, val_g, ind_g, val_s, ind_s,
                U, U_tmp, _, P, W_a, W_s, A, UU, _, PP,
                -1, lmbda_g, lmbda_a, lmbda_s, lmbda,
                is_smp_exp, False, is_smp_exp_g
            )

        elif solver == "cg":
            partial_wals_cg(
                u, val_x, ind_x, val_y, ind_y, val_g, ind_g, val_s, ind_s,
                U, U_tmp, _, P, W_a, W_s, A, UU, _, PP,
                -1, lmbda_g, lmbda_a, lmbda_s, lmbda,
                is_smp_exp, False, is_smp_exp_g,
                cg_steps=cg_steps, eps=eps,
            )
