import numba as nb
import numpy as np


@nb.njit(nogil=True, parallel=True)
def update_user_factor(data, indices, indptr, U, V, lmbda):
    """"""
    VV = V.T @ V  # precompute
    d = V.shape[1]
    I = np.eye(d, dtype=VV.dtype)
    # randomize the order so that scheduling is more efficient
    rnd_idx = np.random.permutation(U.shape[0])

    # for n in range(U.shape[0]):
    for n in nb.prange(U.shape[0]):
        u = rnd_idx[n]
        u0, u1 = indptr[u], indptr[u + 1]
        if u1 - u0 == 0:
            continue
        ind = indices[u0:u1]
        val = data[u0:u1]
        U[u] = partial_ALS(val, ind, V, VV, lmbda)


@nb.njit(nogil=True, parallel=True)
def update_item_factor(data, indices, indptr, U, V, X, W, lmbda_x, lmbda):
    """"""
    UU = U.T @ U
    d = U.shape[1]
    h = X.shape[1]
    I = np.eye(d, dtype=UU.dtype)
    # randomize the order so that scheduling is more efficient
    rnd_idx = np.random.permutation(V.shape[0])

    for n in nb.prange(V.shape[0]):
    # for n in range(V.shape[0]):
        i = rnd_idx[n]
        i0, i1 = indptr[i], indptr[i+1]
        # if i1 - i0 == 0:
        #     continue
        ind = indices[i0:i1]
        val = data[i0:i1]
        V[i] = partial_ALS_feat(val, ind, U, UU, X[i], W, lmbda_x, lmbda)


@nb.njit
def update_feat_factor(V, X, XX, W, lmbda_x, lmbda):
    h = X.shape[1]
    I = np.eye(h, dtype=V.dtype)
    # d = V.shape[1]
    # A = np.zeros((h, h))
    # B = np.zeros((h, d))

    A = XX + (lmbda / lmbda_x) * I
    # for f in range(h):
    #     for q in range(f, h):
    #         if f == q:
    #             A[f, q] += lmbda / lmbda_x 
    #         for j in range(X.shape[0]):
    #             A[f, q] += X[j, f] * X[j, q]
    # A = A + A.T - np.diag(A)

    B = X.T @ V
    # for f in range(h):
    #     for r in range(d):
    #         for j in range(X.shape[0]):
    #             B[f, r] += X[j, f] * V[j, r]

    # update feature factors
    W = np.linalg.solve(A, B)


@nb.njit
def partial_ALS(data, indices, V, VV, lmbda):
    d = V.shape[1]
    b = np.zeros((d,))
    A = np.zeros((d, d))
    c = data + 0
    vv = V[indices].copy()
    # I = np.eye(d, dtype=VV.dtype)

    # b = np.dot(c, vv)
    for f in range(d):
        for j in range(len(c)):
            b[f] += c[j] * vv[j, f]

    # A = VV + vv.T @ np.diag(c - 1) @ vv + lmbda * I
    for f in range(d):
        for q in range(f, d):
            if q == f:
                A[f, q] += lmbda
            A[f, q] += VV[f, q]
            for j in range(len(c)):
                A[f, q] += vv[j, f] * (c[j] - 1) * vv[j, q]

    # copy the triu elements to the tril
    # A = A + A.T - np.diag(np.diag(A))
    for j in range(d):
        for k in range(j+1, d):
            A[k][j] = A[j][k]

    # update user factor
    return np.linalg.solve(A, b.ravel())


@nb.njit
def partial_ALS_feat(data, indices, U, UU, x, W, lmbda_x, lmbda):
    d = U.shape[1]
    b = np.zeros((d,))
    A = np.zeros((d, d))
    xw = np.zeros((d,))
    c = data + 0
    uu = U[indices].copy()
    # I = np.eye(d, dtype=UU.dtype)

    # xw = x @ W
    for f in range(d):
        for h in range(len(x)):
            xw[f] += x[h] * W[h, f]

    # b = np.dot(c, uu) + lmbda_x * xw
    for f in range(d):
        b[f] += lmbda_x * xw[f]
        for j in range(len(c)):
            b[f] += c[j] * uu[j, f]

    # A = UU + uu.T @ np.diag(c - 1) @ uu + (lmbda_x + lmbda) * I
    for f in range(d):
        for q in range(f, d):
            if q == f:
                A[f, q] += (lmbda + lmbda_x)
            A[f, q] += UU[f, q]
            for j in range(len(c)):
                A[f, q] += uu[j, f] * (c[j] - 1) * uu[j, q]

    # copy the triu elements to the tril
    # A = A + A.T - np.diag(np.diag(A))
    for j in range(d):
        for k in range(j+1, d):
            A[k][j] = A[j][k]

    return np.linalg.solve(A, b.ravel())
