from numpy.linalg import norm, lstsq
import numpy as np


def fgmres(A, b, M, maxits=1000, tol=1e-6, x0=None):
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()
    r0 = b - A(x)
    b_norm = norm(b) or 1.0
    res_norm = norm(r0)
    res_hist = [res_norm / b_norm]

    if res_hist[-1] < tol:
        return x, 0, res_hist

    V = np.zeros((n, maxits + 1))
    H = np.zeros((maxits + 1, maxits))
    W_list = []

    V[:, 0] = r0 / res_norm
    g = np.zeros(maxits + 1)
    g[0] = res_norm

    for j in range(maxits):
        w_j = M(V[:, j])  # Apply preconditioner to V[:, j]
        u = A(w_j)  # Compute A*w_j

        for i in range(j + 1):  # Arnoldi Orthogonalization
            H[i, j] = np.dot(V[:, i], u)
            u -= H[i, j] * V[:, i]

        H[j + 1, j] = norm(u)

        if H[j + 1, j] < 1e-14:  # Check for happy breakdown
            W_list.append(w_j)
            break

        V[:, j + 1] = u / H[j + 1, j]

        W_list.append(w_j)

        # Solve the small least-squares problem
        y, *_ = lstsq(H[:j + 2, :j + 1], g[:j + 2], rcond=None)

        # Compute and store the true residual norm explicitly
        res_norm = norm(g[:j + 2] - H[:j + 2, :j + 1] @ y)
        res_hist.append(res_norm / b_norm)

        if res_hist[-1] < tol:
            break

    iters = len(W_list)

    # Reconstruct the final solution explicitly from scratch:
    for j in range(iters):
        x += y[j] * W_list[j]

    return x, iters, res_hist


def dcg(A, b, x0=None, W=None, AW=None, M=None, tol=1e-6, maxiter=1000, is_W_A_orthonormal=False):
    x = np.zeros_like(b) if x0 is None else x0.copy()
    r0 = b - A @ x

    if W is None:
        def P(x): return x
    else:
        if not is_W_A_orthonormal:
            AW = A @ W
            WTAW = W.T @ AW
            WTAW_inv = np.linalg.inv(WTAW)
            WTAW_inv_Wt_A = WTAW_inv @ AW.T
            def P(x): return x - W @ (WTAW_inv_Wt_A @ x)
            coarse_sol = W @ (WTAW_inv @ (W.T @ r0))
        else:
            def P(x): return x - W @ (AW.T @ x)
            coarse_sol = W @ (W.T @ r0)
        x += coarse_sol

    r = b - A @ x

    b_norm = np.linalg.norm(b)
    res = np.linalg.norm(r) / b_norm
    resvec = [res]

    if res < tol or maxiter == 0 or b_norm == 0:
        return x, 0, resvec, 0, [], [], [], None

    z = M(r)
    p = P(z)
    p_list = []
    Ap_list = []
    pnorm_list = []
    gamma_old = np.dot(r, z)
    tag = 3

    for j in range(1, maxiter + 1):
        s = A @ p
        p_A_norm = np.dot(s, p)
        p_list.append(p.copy() / np.sqrt(p_A_norm))
        Ap_list.append(s.copy() / np.sqrt(p_A_norm))
        pnorm_list.append(np.sqrt(p_A_norm))

        alpha = gamma_old / p_A_norm
        x = x + alpha * p
        r = r - alpha * s
        res = np.linalg.norm(r) / b_norm
        resvec.append(res)

        if res < tol:
            tag = 1
            break

        z = M(r)

        for p_old, ap_old in zip(p_list, Ap_list):
            z -= np.dot(ap_old, z) * p_old

        gamma_new = np.dot(r, z)
        # beta = gamma_new / gamma_old
        p = P(z)
        gamma_old = gamma_new

    return x, j, resvec, tag, p_list, Ap_list, pnorm_list, x - coarse_sol if W is not None else x
