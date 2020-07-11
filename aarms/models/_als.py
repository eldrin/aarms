import numba as nb
import numpy as np


@nb.njit(nogil=True, parallel=True, cache=True)
def update_user_factor(data, indices, indptr, U, V, lmbda, solver='cg'):
    """"""
    d = V.shape[1]
    VVpI = V.T @ V + lmbda * np.eye(d, dtype=V.dtype)
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
        if solver == 'cg':
            partial_ALS_cg(u, val, ind, U, V, VVpI, cg_steps=3, eps=1e-20)
        elif solver == 'cholskey':
            partial_ALS(u, val, ind, U, V, VVpI)


@nb.njit(nogil=True, parallel=True, cache=True)
def update_item_factor(data, indices, indptr, U, V, X, W, lmbda_x, lmbda, solver='cg'):
    """"""
    d = U.shape[1]
    h = X.shape[1]
    UUpI = U.T @ U + lmbda * np.eye(d, dtype=U.dtype)
    # randomize the order so that scheduling is more efficient
    rnd_idx = np.random.permutation(V.shape[0])

    for n in nb.prange(V.shape[0]):
        i = rnd_idx[n]
        i0, i1 = indptr[i], indptr[i+1]
        # if i1 - i0 == 0:
        #     continue
        ind = indices[i0:i1]
        val = data[i0:i1]

        if solver == 'cg':
            partial_ALS_feat_cg(i, val, ind, V, U, UUpI, X[i], W, lmbda_x,
                                cg_steps=3, eps=1e-20)
        elif solver == 'cholskey':
            partial_ALS_feat(i, val, ind, V, U, UUpI, X[i], W, lmbda_x)


@nb.njit(cache=True)
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


@nb.njit(cache=True)
def partial_ALS(u, data, indices, U, V, VVpI):
    d = V.shape[1]
    b = np.zeros((d,), dtype=V.dtype)
    # A = np.zeros((d, d))
    A = VVpI.copy()
    c = data + V.dtype.type(0.)
    vv = V[indices].copy()
    # I = np.eye(d, dtype=VV.dtype)

    # b = np.dot(c, vv)
    for f in range(d):
        for j in range(len(c)):
            b[f] += c[j] * vv[j, f]

    # A = VV + vv.T @ np.diag(c - 1) @ vv + lmbda * I
    for f in range(d):
        for q in range(f, d):
            # if q == f:
            #     A[f, q] += lmbda
            # A[f, q] += VV[f, q]
            for j in range(len(c)):
                A[f, q] += vv[j, f] * (c[j] - 1) * vv[j, q]

    # copy the triu elements to the tril
    # A = A + A.T - np.diag(np.diag(A))
    for j in range(d):
        for k in range(j+1, d):
            A[k][j] = A[j][k]

    # update user factor
    # U[u] = np.linalg.solve(A, b.ravel())
    U[u] = np.linalg.solve(A, b)


@nb.njit([
    "void(i8, f4[::1], i4[::1], f4[:, ::1], f4[:, ::1], f4[:, ::1], i8, f8)",
    "void(i8, f8[::1], i4[::1], f8[:, ::1], f8[:, ::1], f8[:, ::1], i8, f8)",
], cache=True)
def partial_ALS_cg(u, data, indices, U, V, VVpI, cg_steps=3, eps=1e-20):
    """"""
    d = V.shape[1]
    b = np.zeros((d,), dtype=V.dtype)
    c = data + V.dtype.type(0.) 
    vv = V[indices].copy()
    x = U[u].copy()
    r = np.zeros((d,), dtype=V.dtype)

    # [r = VCp - V(C - I)Vu - VVu]
    # ======================================
    # r = -VVpI.dot(x)
    for f in range(d):
        for q in range(d):
            r[f] -= VVpI[f, q] * x[q]
    # r += VCp - V(C - I)Vu
    # <=> r += V(Cp - (C - I)Vu)
    for j in range(len(c)):
        vx = r.dtype.type(0.)
        for f in range(d):
            vx += vv[j, f] * x[f]
        for f in range(d):
            r[f] += (c[j] - (c[j] - 1) * vx) * vv[j, f]

    p = r.copy()
    rsold = r.dtype.type(0.)
    for f in range(d):
        rsold += r[f]**2
    if rsold**.5 < eps:
        return

    for it in range(cg_steps):
        # calculate Ap = VCVp - without actually calculate VCV
        Ap = np.zeros((d,), dtype=V.dtype)
        for f in range(d):
            for q in range(d):
                Ap[f] += VVpI[f, q] * p[q]

        for j in range(len(c)):
            vp = r.dtype.type(0.)
            for f in range(d):
                vp += vv[j, f] * p[f]

            for f in range(d):
                Ap[f] += (c[j] - 1) * vp * vv[j, f]

        # standard CG update
        pAp = r.dtype.type(0.)
        for f in range(d):
            pAp += p[f] * Ap[f]
        alpha = rsold / pAp
        for f in range(d):
            x[f] += alpha * p[f]
            r[f] -= alpha * Ap[f]

        rsnew = r.dtype.type(0.)
        for f in range(d):
            rsnew += r[f]**2
        if rsnew**.5 < eps:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    U[u] = x


@nb.njit(cache=True)
def partial_ALS_feat(i, data, indices, V, U, UUpI, x, W, lmbda_x):
    d = U.shape[1]
    b = np.zeros((d,), dtype=V.dtype)
    A = UUpI.copy()
    xw = np.zeros((d,), dtype=V.dtype)
    c = data + U.dtype.type(0.)
    uu = U[indices].copy()

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
                A[f, q] += lmbda_x
            for j in range(len(c)):
                A[f, q] += uu[j, f] * (c[j] - 1) * uu[j, q]

    # copy the triu elements to the tril
    # A = A + A.T - np.diag(np.diag(A))
    for j in range(d):
        for k in range(j+1, d):
            A[k][j] = A[j][k]

    V[i] = np.linalg.solve(A, b.ravel())


@nb.njit([
    "void(i8, f4[::1], i4[::1], f4[:, ::1], f4[:, ::1], f4[:, ::1], f4[::1], f4[:, ::1], f8, i8, f8)",
    "void(i8, f8[::1], i4[::1], f8[:, ::1], f8[:, ::1], f8[:, ::1], f8[::1], f8[:, ::1], f8, i8, f8)",
], cache=True)
def partial_ALS_feat_cg(i, data, indices, V, U, UUpI, x, W, lmbda_x,
                        cg_steps=3, eps=1e-20):
    """"""
    d = V.shape[1]
    c = data + V.dtype.type(0.)
    uu = U[indices].copy()
    v = V[i].copy()
    r = np.zeros((d,), dtype=V.dtype)

    # compute residual
    # [r = b - Ap]
    #    => [r = (UCp + l_x * xW) - (U(C-I)Uv + UUv + (l_x + l) * Iv)]
    # =============================================
    # r = -(UUpI + l_x * I).dot(v)
    for f in range(d):
        for q in range(d):
            if f == q:
                r[f] -= lmbda_x * v[q]
            r[f] -= UUpI[f, q] * v[q]

    # r += UCp - U(C-I)Uv
    # <=> r += U(Cp - (C-I)Uv)
    for j in range(len(c)):
        uv = r.dtype.type(0.)
        for f in range(d):
            uv += uu[j, f] * v[f]

        for f in range(d):
            r[f] += (c[j] - (c[j] - 1) * uv) * uu[j, f]

    # r += l_x * xW
    for f in range(d):
        for h in range(len(x)):
            r[f] += lmbda_x * x[h] * W[h, f]

    p = r.copy()
    rsold = r.dtype.type(0.)
    for f in range(d):
        rsold += r[f]**2
    if rsold**.5 < eps:
        return

    for it in range(cg_steps):
        # calculate Ap = (UCU + (l_x + l)I)p without actually calculate UCU
        Ap = np.zeros((d,), dtype=V.dtype)
        for f in range(d):
            for q in range(d):
                if f == q:
                    Ap[f] += lmbda_x * p[q]
                Ap[f] += UUpI[f, q] * p[q]

        for j in range(len(c)):
            up = r.dtype.type(0.)
            for f in range(d):
                up += uu[j, f] * p[f]

            for f in range(d):
                Ap[f] += uu[j, f] * (c[j] - 1) * up

        # standard CG update
        pAp = r.dtype.type(0.)
        for f in range(d):
            pAp += p[f] * Ap[f]
        alpha = rsold / pAp
        for f in range(d):
            v[f] += alpha * p[f]
            r[f] -= alpha * Ap[f]

        rsnew = r.dtype.type(0.)
        for f in range(d):
            rsnew += r[f]**2
        if rsnew**.5 < eps:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    # update
    V[i] = v
