from numpy.linalg import norm, solve
import numpy as np


def deflated_pcg(matmult, b, precond=None, maxits=1000, tol=1e-6, Z=None, x0=None):
    """
    Deflated Preconditioned Conjugate Gradient for SPD matrices.
    - matmult(v): computes A @ v
    - precond(v): computes M^{-1} @ v (preconditioner application), or identity if None
    - b: right-hand side vector
    - maxits: maximum iterations
    - tol: convergence tolerance on relative residual norm
    - Z: deflation basis matrix (n x k), columns span deflation subspace
    - x0: initial guess (default 0 vector)
    Returns: (x, iters, res_history)
    """
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()
    # If deflation basis is given, perform coarse solve for initial guess
    if Z is not None and Z.shape[1] > 0:
        # Compute A*Z (for coarse operator and projector)
        k = Z.shape[1]
        W = np.zeros((n, k))
        for j in range(k):
            W[:, j] = matmult(Z[:, j])
        E = Z.T.dot(W)                         # Galerkin matrix E = Z^T A Z (k x k)
        r = b - matmult(x)                     # initial residual
        # Solve coarse system: E * y = Z^T r
        try:
            y_coarse = np.linalg.solve(E, Z.T.dot(r))
        except np.linalg.LinAlgError:
            y_coarse, *_ = np.linalg.lstsq(E, Z.T.dot(r), rcond=None)
        x += Z.dot(y_coarse)                  # add coarse correction
        r = b - matmult(x)                     # updated residual
        # Project residual onto complement of deflation subspace: r <- r - W * (E^{-1} Z^T r)
        try:
            gamma = np.linalg.solve(E, Z.T.dot(r))
        except np.linalg.LinAlgError:
            gamma, *_ = np.linalg.lstsq(E, Z.T.dot(r), rcond=None)
        r -= W.dot(gamma)
    else:
        r = b - matmult(x)
    # Initialize residual norm and check convergence
    b_norm = np.linalg.norm(b)
    if b_norm == 0:
        b_norm = 1.0
    res_norm = np.linalg.norm(r)
    res_hist = [res_norm / b_norm]
    if res_hist[-1] < tol:
        return x, 0, res_hist
    # Apply preconditioner to r
    z = r.copy() if precond is None else precond(r)
    p = z.copy()
    rz_old = r.dot(z)                        # (r, M^{-1}r)
    # CG iteration
    for it in range(1, maxits + 1):
        Ap = matmult(p)
        alpha = rz_old / p.dot(Ap)           # step size alpha
        x += alpha * p
        r -= alpha * Ap
        # Project new residual out of deflation subspace (preserve orthogonality)
        if Z is not None and Z.shape[1] > 0:
            try:
                gamma = np.linalg.solve(E, Z.T.dot(r))
            except np.linalg.LinAlgError:
                gamma, *_ = np.linalg.lstsq(E, Z.T.dot(r), rcond=None)
            r -= W.dot(gamma)
        res_norm = np.linalg.norm(r)
        res_hist.append(res_norm / b_norm)
        if res_hist[-1] < tol:
            return x, it, res_hist
        # Preconditioning and direction update
        z_new = r.copy() if precond is None else precond(r)
        rz_new = r.dot(z_new)
        beta = rz_new / rz_old               # CG beta
        p = z_new + beta * p
        rz_old = rz_new
    return x, it, res_hist


def deflated_fgmres(matmult, b, precond=None, maxits=1000, tol=1e-6, Z=None, x0=None):
    """
    Deflated Flexible GMRES (no restart).
    - matmult(v): computes A @ v
    - precond(v): applies preconditioner M^{-1} (if None, identity is used)
    - b: right-hand side vector
    - maxits: maximum iterations (Krylov dimension)
    - tol: convergence tolerance on relative residual norm
    - Z: deflation basis matrix (n x k)
    - x0: initial guess (default 0 vector)
    Returns: (x, iters, res_history)
    """
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()
    Qz = None
    if Z is not None and Z.shape[1] > 0:
        # Precompute A*Z and E = Z^T A Z for coarse solve
        k = Z.shape[1]
        Wz = np.column_stack([matmult(Z[:, j]) for j in range(k)])  # A*Z
        E = Z.T.dot(Wz)
        r0 = b - matmult(x)
        try:
            y_coarse = np.linalg.solve(E, Z.T.dot(r0))
        except np.linalg.LinAlgError:
            y_coarse, *_ = np.linalg.lstsq(E, Z.T.dot(r0), rcond=None)
        x += Z.dot(y_coarse)               # coarse correction
        r0 = b - matmult(x)                # new residual
        # Orthonormalize Z for projecting out its span
        Qz, _ = np.linalg.qr(Z, mode='reduced')
        # Project initial residual orthogonal to deflation subspace
        r0 = r0 - Qz.dot(Qz.T.dot(r0))
    else:
        r0 = b - matmult(x)
    # Initial residual norm
    b_norm = np.linalg.norm(b) or 1.0
    res_norm = np.linalg.norm(r0)
    res_hist = [res_norm / b_norm]
    if res_hist[-1] < tol:
        return x, 0, res_hist
    # Initialize Arnoldi basis containers
    V = np.zeros((n, maxits + 1))    # Krylov basis (unpreconditioned residual vectors)
    H = np.zeros((maxits + 1, maxits))  # Hessenberg matrix
    V[:, 0] = r0 / res_norm
    # Flexible GMRES Arnoldi process
    W_list = []  # store preconditioned basis vectors for solution combination
    g = np.zeros(maxits + 1)
    g[0] = res_norm  # initial residual magnitude
    iters = 0
    for j in range(maxits):
        iters = j + 1
        # Apply (flexible) preconditioner
        w_j = V[:, j] if precond is None else precond(V[:, j])
        # Mat-vec product
        u = matmult(w_j)
        # Remove deflation-space component from u
        if Qz is not None:
            u = u - Qz.dot(Qz.T.dot(u))
        # Arnoldi: orthogonalize u against current basis V[:, :j+1]
        for i in range(j + 1):
            H[i, j] = np.dot(V[:, i], u)
            u -= H[i, j] * V[:, i]
        H[j + 1, j] = np.linalg.norm(u)
        if H[j + 1, j] < 1e-14:      # happy breakdown (u is zero or nearly in span)
            break
        V[:, j + 1] = u / H[j + 1, j]
        # Least-squares solve via Arnoldi (update the residual norm)
        y, *_ = np.linalg.lstsq(H[:j + 2, :j + 1], g[:j + 2], rcond=None)
        res = g[:j + 2] - H[:j + 2, :j + 1].dot(y)
        res_norm = np.linalg.norm(res)
        res_hist.append(res_norm / b_norm)
        # Store preconditioned vector for solution reconstruction
        W_list.append(w_j)
        if res_hist[-1] < tol:
            break
    # Solve least-squares for final y (if not already done above)
    y, *_ = np.linalg.lstsq(H[:iters + 1, :iters], g[:iters + 1], rcond=None)
    # Reconstruct solution: x0 + sum(y_j * w_j)
    for j in range(iters):
        x += y[j] * W_list[j]
    return x, iters, res_hist


def deflated_fgmres2(matmult, b, precond=None, maxits=1000, tol=1e-6, Z=None, x0=None):
    """
    Deflated Flexible GMRES (no restart).
    - matmult(v): computes A @ v
    - precond(v): applies preconditioner M^{-1} (if None, identity is used)
    - b: right-hand side vector
    - maxits: maximum iterations (Krylov dimension)
    - tol: convergence tolerance on relative residual norm
    - Z: deflation basis matrix (n x k)
    - x0: initial guess (default 0 vector)
    Returns: (x, iters, res_history)
    """
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()
    Qz = None
    if Z is not None and Z.shape[1] > 0:
        # Precompute A*Z and E = Z^T A Z for coarse solve
        k = Z.shape[1]
        Wz = np.column_stack([matmult(Z[:, j]) for j in range(k)])  # A*Z
        E = Z.T.dot(Wz)
        r0 = b - matmult(x)
        try:
            y_coarse = np.linalg.solve(E, Z.T.dot(r0))
        except np.linalg.LinAlgError:
            y_coarse, *_ = np.linalg.lstsq(E, Z.T.dot(r0), rcond=None)
        x += Z.dot(y_coarse)               # coarse correction
        r0 = b - matmult(x)                # new residual
        # Orthonormalize Z for projecting out its span
        Qz, _ = np.linalg.qr(Z, mode='reduced')
        # Project initial residual orthogonal to deflation subspace
        r0 = r0 - Qz.dot(Qz.T.dot(r0))
    else:
        r0 = b - matmult(x)
    # Initial residual norm
    b_norm = np.linalg.norm(b) or 1.0
    res_norm = np.linalg.norm(r0)
    res_hist = [res_norm / b_norm]
    if res_hist[-1] < tol:
        return x, 0, res_hist
    # Initialize Arnoldi basis containers
    V = np.zeros((n, maxits + 1))    # Krylov basis (unpreconditioned residual vectors)
    H = np.zeros((maxits + 1, maxits))  # Hessenberg matrix
    V[:, 0] = r0 / res_norm
    # Flexible GMRES Arnoldi process
    W_list = []  # store preconditioned basis vectors for solution combination
    g = np.zeros(maxits + 1)
    g[0] = res_norm  # initial residual magnitude
    iters = 0
    for j in range(maxits):
        iters = j + 1
        # Apply (flexible) preconditioner
        w_j = V[:, j] if precond is None else precond(V[:, j])
        # Mat-vec product
        u = matmult(w_j)
        # Remove deflation-space component from u
        if Qz is not None:
            u = u - Qz.dot(Qz.T.dot(u))
        # Arnoldi: orthogonalize u against current basis V[:, :j+1]
        for i in range(j + 1):
            H[i, j] = np.dot(V[:, i], u)
            u -= H[i, j] * V[:, i]
        H[j + 1, j] = np.linalg.norm(u)
        if H[j + 1, j] < 1e-14:      # happy breakdown (u is zero or nearly in span)
            break
        V[:, j + 1] = u / H[j + 1, j]
        # Least-squares solve via Arnoldi (update the residual norm)
        y, *_ = np.linalg.lstsq(H[:j + 2, :j + 1], g[:j + 2], rcond=None)
        res = g[:j + 2] - H[:j + 2, :j + 1].dot(y)
        res_norm = np.linalg.norm(res)
        res_hist.append(res_norm / b_norm)
        # Store preconditioned vector for solution reconstruction
        W_list.append(w_j)
        if res_hist[-1] < tol:
            break
    # Solve least-squares for final y (if not already done above)
    y, *_ = np.linalg.lstsq(H[:iters + 1, :iters], g[:iters + 1], rcond=None)
    # Reconstruct solution: x0 + sum(y_j * w_j)
    for j in range(iters):
        x += y[j] * W_list[j]
    return x, iters, res_hist


def deflated_fgmres3(A, b, M, maxits=1000, tol=1e-6, Z=None, x0=None):
    """
    Deflated Flexible GMRES (no restart).
    - matmult(v): computes A @ v
    - precond(v): applies preconditioner M^{-1} (if None, identity is used)
    - b: right-hand side vector
    - maxits: maximum iterations (Krylov dimension)
    - tol: convergence tolerance on relative residual norm
    - Z: deflation basis matrix (n x k)
    - x0: initial guess (default 0 vector)
    Returns: (x, iters, res_history)
    """
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()
    if Z is None:
        def P(v):
            return v
        r0 = b - A(x)
    else:
        Z, _ = np.linalg.qr(Z)
        k = Z.shape[1]
        MZ = np.column_stack([M(Z[:, i]) for i in range(k)])
        MZ, _ = np.linalg.qr(MZ)
        AZ = np.column_stack([A(Z[:, i]) for i in range(k)])
        ZTAZ = Z.T @ AZ

        def P(v):
            return v - MZ @ (MZ.T @ v)

        r0 = b - A(x)
        rhs = Z.T @ r0
        u = Z @ solve(ZTAZ, rhs)
        x = x + u
        r0 = r0 - u

    z0 = P(r0)

    # Initial residual norm
    b_norm = np.linalg.norm(b) or 1.0
    res_norm = np.linalg.norm(z0)
    res_hist = [res_norm / b_norm]
    if res_hist[-1] < tol:
        return x, 0, res_hist
    # Initialize Arnoldi basis containers
    V = np.zeros((n, maxits + 1))    # Krylov basis (unpreconditioned residual vectors)
    H = np.zeros((maxits + 1, maxits))  # Hessenberg matrix
    V[:, 0] = r0 / res_norm
    # Flexible GMRES Arnoldi process
    W_list = []  # store preconditioned basis vectors for solution combination
    g = np.zeros(maxits + 1)
    g[0] = res_norm  # initial residual magnitude
    iters = 0
    for j in range(maxits):
        iters = j + 1
        # Apply (flexible) preconditioner
        w_j = M(V[:, j])
        # Mat-vec product
        u = A(w_j)
        # Remove deflation-space component from u
        u = P(u)

        # Arnoldi: orthogonalize u against current basis V[:, :j+1]
        for i in range(j + 1):
            H[i, j] = np.dot(V[:, i], u)
            u -= H[i, j] * V[:, i]

        # u = P(P(P(u)))  # Project out deflation space
        H[j + 1, j] = np.linalg.norm(u)
        if H[j + 1, j] < 1e-14:      # happy breakdown (u is zero or nearly in span)
            break
        V[:, j + 1] = u / H[j + 1, j]
        # Least-squares solve via Arnoldi (update the residual norm)
        y, *_ = np.linalg.lstsq(H[:j + 2, :j + 1], g[:j + 2], rcond=None)
        res = g[:j + 2] - H[:j + 2, :j + 1].dot(y)
        res_norm = np.linalg.norm(res)
        res_hist.append(res_norm / b_norm)
        # Store preconditioned vector for solution reconstruction
        W_list.append(w_j)
        if res_hist[-1] < tol:
            break
    # Solve least-squares for final y (if not already done above)
    y, *_ = np.linalg.lstsq(H[:iters + 1, :iters], g[:iters + 1], rcond=None)
    # Reconstruct solution: x0 + sum(y_j * w_j)
    for j in range(iters):
        x += y[j] * W_list[j]

    # Deflation correction:
    # if Z is not None:
    #     k = Z.shape[1]
    #     AZ = np.column_stack([A(Z[:, i]) for i in range(k)])

    #     rhs = Z.T @ (b - A(x))
    #     u = solve(Z.T @ AZ, rhs)
    #     x = x + Z @ u

    return x, iters, res_hist


def deflated_gmres(matmult, b, precond=None, maxits=1000, tol=1e-6, Z=None, x0=None):
    """
    Deflated Flexible GMRES (no restart).
    - matmult(v): computes A @ v
    - precond(v): applies preconditioner M^{-1} to vector v (if None, identity is used)
    - b: right-hand side vector
    - maxits: maximum iterations (Krylov dimension)
    - tol: convergence tolerance on relative residual norm
    - Z: deflation basis matrix (n x k), with k deflation vectors
    - x0: initial guess (default is zero vector)
    Returns: (x, iters, res_history)
    """
    b = np.array(b, copy=False)
    n = b.shape[0]
    # Initial guess and residual
    x = np.zeros(n) if x0 is None else np.array(x0, copy=True)
    if precond is None:
        def precond_func(v): return v
    else:
        precond_func = precond
    r = b - matmult(x)
    normb = np.linalg.norm(b)
    if normb == 0:
        normb = 1.0  # avoid division by zero if b is zero
    res_norm = np.linalg.norm(r)
    if res_norm / normb < tol:
        # Already converged
        return x, 0, [res_norm / normb]
    # Deflation: initial projection on Z subspace
    Qz = None
    if Z is not None:
        Z = np.array(Z, copy=False)
        if Z.ndim == 1:
            Z = Z.reshape(n, 1)
        # Compute coarse correction: solve Z^T A Z * gamma = Z^T r
        k = Z.shape[1]
        AZ = np.zeros((n, k))
        for j in range(k):
            AZ[:, j] = matmult(Z[:, j])
        E = Z.T @ AZ
        try:
            gamma = np.linalg.solve(E, Z.T @ r)
        except np.linalg.LinAlgError:
            raise RuntimeError("Deflation breakdown: Z^T A Z is singular and cannot be inverted.")
        # Update solution and residual
        x += Z @ gamma
        r = b - matmult(x)
        res_norm = np.linalg.norm(r)
        if res_norm / normb < tol:
            # Converged after deflation correction
            return x, 0, [res_norm / normb]
        # Orthonormalize Z for projecting out components during iterations
        Qz, _ = np.linalg.qr(Z, mode='reduced')
    # Initialize Krylov basis containers
    V = np.zeros((n, maxits + 1))
    H = np.zeros((maxits + 1, maxits))
    # Initial Arnoldi vector
    V[:, 0] = r / res_norm
    res_history = [res_norm / normb]
    # Givens rotation arrays for least-squares update
    cs = np.zeros(maxits)
    sn = np.zeros(maxits)
    g = np.zeros(maxits + 1)
    g[0] = res_norm  # initial ||r||
    # Arnoldi (FGMRES) iteration
    for i in range(maxits):
        # Apply preconditioner
        vi = V[:, i]
        zi = precond_func(vi)
        # Apply matrix A
        w = matmult(zi)
        # Explicit deflation projection (remove components along Z)
        if Z is not None:
            # (Use Qz for an orthonormal basis of Z's columns)
            alpha = Qz.T @ w
            w = w - Qz @ alpha
        # Modified Gram-Schmidt orthogonalization
        for j in range(i + 1):
            H[j, i] = np.dot(V[:, j], w)
            w -= H[j, i] * V[:, j]
        H[i + 1, i] = np.linalg.norm(w)
        # Check for breakdown (happy breakdown or stagnation)
        if H[i + 1, i] < 1e-15:
            # Solve least-squares for current H (i+1 x i system) to get solution
            y, *_ = np.linalg.lstsq(H[:i + 1, :i], g[:i + 1], rcond=None)
            x_approx = x + np.column_stack([precond_func(V[:, j]) for j in range(i)]) @ y
            # Compute true residual norm to verify
            true_res = b - matmult(x_approx)
            if np.linalg.norm(true_res) / normb < tol:
                # Converged (happy breakdown)
                res_history.append(np.linalg.norm(true_res) / normb)
                return x_approx, i, res_history
            else:
                # Unhappy breakdown
                raise RuntimeError("GMRES breakdown: stagnation at iteration %d" % i)
        # Normalize the new basis vector
        V[:, i + 1] = w / H[i + 1, i]
        # Apply Givens rotations to the new column of H
        for j in range(i):
            temp = cs[j] * H[j, i] + sn[j] * H[j + 1, i]
            H[j + 1, i] = -sn[j] * H[j, i] + cs[j] * H[j + 1, i]
            H[j, i] = temp
        # Compute new Givens rotation to eliminate H[i+1, i]
        r_val = np.hypot(H[i, i], H[i + 1, i])
        cs[i] = H[i, i] / r_val
        sn[i] = H[i + 1, i] / r_val
        H[i, i] = r_val
        H[i + 1, i] = 0.0
        # Update the residual vector g
        g[i + 1] = -sn[i] * g[i]
        g[i] = cs[i] * g[i]
        res_norm = abs(g[i + 1])
        res_history.append(res_norm / normb)
        # Check convergence
        if res_norm / normb < tol:
            # Solve for y (back-substitution since H is triangular)
            y = g[:i + 1].copy()
            for j in range(i, -1, -1):
                y[j] /= H[j, j]
                for k in range(j - 1, -1, -1):
                    y[k] -= H[k, j] * y[j]
            # Reconstruct solution from preconditioned basis
            X_basis = [precond_func(V[:, j]) for j in range(i + 1)]
            x += np.column_stack(X_basis) @ y
            return x, i + 1, res_history
    # If maxits reached without convergence, return the best solution found
    y, *_ = np.linalg.lstsq(H[:maxits + 1, :maxits], g[:maxits + 1], rcond=None)
    X_basis = [precond_func(V[:, j]) for j in range(maxits)]
    x += np.column_stack(X_basis) @ y
    return x, maxits, res_history


def recycled_fgmres(A, b, M=None, max_iter=100, tol=1e-6, recycle_data=None):
    """
    Flexible GMRES with Krylov subspace recycling (deflation).

    Parameters:
        A: function A(x) that returns the matrix-vector product A @ x.
        b: right-hand side vector.
        M: preconditioner function M(x) (approximate solve, if None then identity).
        max_iter: maximum number of iterations.
        tol: convergence tolerance (relative residual).
        recycle_data: dictionary with 'U' (recycled subspace vectors).

    Returns:
        x: solution vector.
        new_recycle_data: updated subspace U for future calls.
    """
    n = len(b)
    if M is None:
        def M(x): return x  # Default to identity preconditioner

    # Extract and orthonormalize recycled subspace U if available
    U = None
    if recycle_data is not None and 'U' in recycle_data and recycle_data['U'] is not None:
        U = recycle_data['U']
        U, _ = np.linalg.qr(U)  # Ensure U is orthonormal

    # Initial guess using projection onto U
    x0 = np.zeros(n)
    if U is not None and U.shape[1] > 0:
        W = A(U)  # Compute projected operator W = A U
        y0, *_ = np.linalg.lstsq(W, b, rcond=None)  # Solve small system
        x0 = U @ y0  # Initial guess in the span of U

    # Compute initial residual
    r0 = b - A(x0)
    b_norm = np.linalg.norm(b)
    if b_norm == 0:
        return x0, {'U': U}, 0
    r_norm = np.linalg.norm(r0)
    if r_norm / b_norm < tol:
        return x0, {'U': U}, 0

    # Augment Krylov basis with U (if available)
    if U is not None:
        if U.shape[1] > 0:
            # Project residual onto complement of U
            r0 -= U @ (U.T @ r0)
            r_norm = np.linalg.norm(r0)

    if r_norm / b_norm < tol:
        return x0, {'U': U}, 0

    # Initialize Krylov basis V and preconditioned basis Z
    V = np.zeros((n, max_iter + 1), dtype=float)
    Z = np.zeros((n, max_iter), dtype=float)
    H = np.zeros((max_iter + 1, max_iter), dtype=float)

    # If using recycling, start Arnoldi with U as additional basis vectors
    start_U = 0
    if U is not None and U.shape[1] > 0:
        V[:, :U.shape[1]] = U
        start_U = U.shape[1]

    V[:, start_U] = r0 / r_norm  # Start Krylov basis after U
    beta = r_norm

    # Givens rotation components
    cs = np.zeros(max_iter)
    sn = np.zeros(max_iter)
    e1 = np.zeros(max_iter + 1)
    e1[0] = beta

    iters = 0
    for j in range(start_U, max_iter):
        iters = j
        vj = V[:, j]

        # Apply preconditioner
        zj = M(vj)
        Z[:, j] = zj

        # Compute A * zj
        w = A(zj)

        # Arnoldi orthogonalization
        for i in range(j + 1):
            H[i, j] = np.dot(V[:, i], w)
            w -= H[i, j] * V[:, i]

        H[j + 1, j] = np.linalg.norm(w)
        if H[j + 1, j] < 1e-14:
            e1 = e1[:j + 2]
            break

        V[:, j + 1] = w / H[j + 1, j]

        # Apply Givens rotation to H to update QR factorization
        for i in range(j):
            temp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
            H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
            H[i, j] = temp

        r = np.hypot(H[j, j], H[j + 1, j])
        cs[j] = H[j, j] / r
        sn[j] = H[j + 1, j] / r
        H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
        H[j + 1, j] = 0.0
        e1[j + 1] = -sn[j] * e1[j]
        e1[j] = cs[j] * e1[j]

        res_norm = abs(e1[j + 1])
        if res_norm / b_norm < tol:
            break

    # Solve the small upper triangular system
    Hm = H[:iters + 1, :iters + 1]
    beta_e1 = e1[:iters + 1]
    y = np.linalg.solve(Hm, beta_e1)

    # Compute final solution
    x = x0 + Z[:, :iters + 1] @ y

    # Recycling: Extract new deflation subspace
    try:
        evals, evecs = np.linalg.eig(Hm)
    except np.linalg.LinAlgError:
        evals, evecs = np.linalg.eig(Hm + 1e-12 * np.random.rand(*Hm.shape))  # Perturb if needed

    k_recycle = min(10, iters + 1)  # Choose up to 10 deflation vectors
    idx = np.argsort(np.abs(evals))[:k_recycle]
    U_new = V[:, :iters + 1] @ evecs[:, idx]

    # Orthonormalize new U
    U_new, _ = np.linalg.qr(U_new)

    # Return updated recycling data
    new_recycle_data = {'U': U_new}
    return x, new_recycle_data, iters


def recycled_cg(A, b, M=None, max_iter=100, tol=1e-6, recycle_data=None):
    """
    Preconditioned Conjugate Gradient with subspace recycling (deflation).
    Solves A x = b for SPD A. Returns (x, new_recycle_data).
    A: function A(x) for matrix-vector product (or A can be a NumPy 2D array).
    M: preconditioner function M(x) (symmetric SPD) or None for no preconditioning.
    recycle_data: dict with 'U' (n x k matrix of recycled vectors).
    """
    n = b.shape[0]
    if M is None:
        def M(x): return x  # Identity preconditioner

    # Use recycled subspace if available to get initial guess
    U = None
    if recycle_data is not None and 'U' in recycle_data and recycle_data['U'] is not None:
        U = recycle_data['U']
    x0 = np.zeros(n)
    if U is not None and U.size > 0:
        # Orthonormalize U (standard Gram-Schmidt)
        Q = np.copy(U)
        for i in range(Q.shape[1]):
            for j in range(i):
                coeff = np.dot(Q[:, j], Q[:, i])
                Q[:, i] -= coeff * Q[:, j]
            norm_q = np.linalg.norm(Q[:, i])
            if norm_q < 1e-12:
                Q[:, i] = 0
            else:
                Q[:, i] /= norm_q
        Q = Q[:, ~np.all(Q == 0, axis=0)]  # remove zero vectors if any

        # Project b into span(Q) and solve A_d * alpha = Q^T b
        W = A(Q) if callable(A) else A.dot(Q)  # compute A*Q (n x k)
        A_d = Q.T.dot(W)          # small k x k matrix
        rhs_small = Q.T.dot(b)    # projected rhs
        try:
            alpha = np.linalg.solve(A_d, rhs_small)
        except np.linalg.LinAlgError:
            # If A_d is singular or ill-conditioned, use least-squares
            alpha, *_ = np.linalg.lstsq(A_d, rhs_small, rcond=None)
        x0 = Q.dot(alpha)
        # We have now an initial guess covering the deflation subspace.
    else:
        Q = None

    # Compute initial residual after deflation guess
    r = b - (A(x0) if callable(A) else A.dot(x0))
    b_norm = np.linalg.norm(b)
    if b_norm == 0:
        return x0, {'U': U}
    r_norm = np.linalg.norm(r)
    if r_norm / b_norm < tol:
        # Already converged
        return x0, {'U': U}

    # Precondition initial residual
    z = M(r)
    p = z.copy()
    rz_inner_old = np.dot(r, z)  # (r, M r)
    x = x0.copy()
    # CG iterations
    for k in range(max_iter):
        Ap = A(p) if callable(A) else A.dot(p)
        pAp = np.dot(p, Ap)
        if pAp <= 1e-16:
            # Encountered breakdown (p^T A p ~ 0)
            break
        alpha = rz_inner_old / pAp
        x += alpha * p
        r -= alpha * Ap
        r_norm = np.linalg.norm(r)
        if r_norm / b_norm < tol:
            # Converged
            break
        z = M(r)
        rz_inner_new = np.dot(r, z)
        beta = rz_inner_new / rz_inner_old
        p = z + beta * p
        rz_inner_old = rz_inner_new

    # Prepare new recycling subspace
    U_new = U.copy() if U is not None else np.zeros((n, 0))
    # Optionally add the last residual direction as a new deflation vector (if needed)
    # Here we add if we used a significant portion of max_iter (indicating a difficult direction)
    if k > 0.5 * max_iter or r_norm / b_norm > tol:
        # Normalize residual
        if np.linalg.norm(r) > 1e-14:
            new_vec = r / np.linalg.norm(r)
        else:
            new_vec = r
        # Orthogonalize against existing U_new
        if U_new.size > 0:
            for j in range(U_new.shape[1]):
                new_vec -= np.dot(U_new[:, j], new_vec) * U_new[:, j]
            # Normalize again
            if np.linalg.norm(new_vec) > 1e-14:
                new_vec = new_vec / np.linalg.norm(new_vec)
        # Append new_vec
        new_vec = new_vec.reshape(-1, 1)
        U_new = np.hstack([U_new, new_vec]) if U_new.size else new_vec
    # (Optional) Cap the size of U_new to prevent unbounded growth
    max_recycle = 10  # for example, keep at most 10 vectors
    if U_new.shape[1] > max_recycle:
        # drop the oldest vectors (you might also drop the least useful; here we drop first extra)
        U_new = U_new[:, -max_recycle:]

    new_data = {'U': U_new}
    return x, new_data


def extend_orthonormal_basis(Q, U, droptol=1e-10, max_size=10, verbose=False):
    """
    Extends an orthonormal basis Q with vectors from U that are orthogonalized
    against Q and previously processed vectors from U.

    Parameters:
        Q : ndarray
            Matrix with orthonormal columns (n x q)
        U : ndarray
            Matrix whose columns will be orthogonalized and potentially added to Q (n x p)
        droptol : float
            Tolerance for dropping nearly linearly dependent vectors
        verbose : bool
            Whether to print information about added/dropped vectors

    Returns:
        Q_extended : ndarray
            Extended orthonormal basis
    """
    # Handle edge cases
    if U.shape[1] == 0:  # No vectors to add
        return Q

    if Q.shape[0] != U.shape[0]:
        raise ValueError("Matrices Q and U must have the same number of rows")

    # cut Q from the start to be at most max_size - U.shape[1]
    number_cut = max(0, Q.shape[1] + U.shape[1] - max_size)
    if number_cut > 0:
        Q = Q[:, number_cut:]

    # Start with the original orthonormal matrix
    Q_result = Q.copy()
    added = 0
    dropped = 0

    for j in range(U.shape[1]):
        # Get the current vector to orthogonalize
        u = U[:, j].copy()
        u = u / np.linalg.norm(u)  # Normalize

        # Orthogonalize against all existing vectors in Q_result
        # for i in range(Q_result.shape[1]):
        #     u = u - np.dot(Q_result[:, i], u) * Q_result[:, i]

        # for i in range(Q_result.shape[1]):
        #     u = u - np.dot(Q_result[:, i], u) * Q_result[:, i]

        u = u - Q_result @ (Q_result.T @ u)
        u = u - Q_result @ (Q_result.T @ u)
        u = u - Q_result @ (Q_result.T @ u)

        # Check the norm after orthogonalization
        norm_u = np.linalg.norm(u)
        print(norm_u)

        if norm_u > droptol:
            # Normalize and add to the result
            u = u / norm_u
            Q_result = np.column_stack((Q_result, u))
            added += 1
        else:
            dropped += 1

    if verbose:
        print(f"Orthogonalization complete: {added} vectors added, {dropped} vectors dropped (threshold: {droptol})")

    return Q_result


def extend_A_orthonormal_basis(A, Q, U, droptol=1e-10, max_size=10, verbose=False):
    """
    Extends an A-orthonormal basis Q with vectors from U that are A-orthogonalized
    against Q and previously processed vectors from U.

    The inner product is defined by A, so the normalization and projections are with respect to
    the A-inner product, i.e., for any vector x, its A-norm is sqrt(x.T @ A @ x). The goal is that
    if Q is A-orthonormal then Q.T @ A @ Q = I.

    Parameters:
        A : ndarray
            A symmetric matrix (n x n) defining the inner product.
        Q : ndarray or None
            Matrix with A-orthonormal columns (n x q). If None, an empty basis is used.
        U : ndarray
            Matrix whose columns will be A-orthogonalized and potentially added to Q (n x p).
        droptol : float
            Tolerance for dropping nearly linearly dependent vectors (in A-norm).
        max_size : int
            Maximum number of columns allowed in the extended basis.
        verbose : bool
            Whether to print information about added/dropped vectors.

    Returns:
        Q_extended : ndarray
            Extended A-orthonormal basis.
    """
    n = U.shape[0]

    # If Q is None, start with an empty basis.
    if Q is None:
        Q = np.empty((n, 0))

    # Enforce the maximum allowed basis size by dropping the oldest vectors if needed.
    number_cut = max(0, Q.shape[1] + U.shape[1] - max_size)
    if number_cut > 0:
        Q = Q[:, number_cut:]

    # Copy the current basis to start extending.
    Q_result = Q.copy()
    added = 0
    dropped = 0

    # Process each column of U.
    for j in range(U.shape[1]):
        u = U[:, j].copy()

        # Normalize u with respect to the A-inner product.
        norm_u = np.sqrt(u.T @ A(u))
        if norm_u == 0:
            dropped += 1
            continue
        u = u / norm_u

        # Repeatedly remove the components in the span of Q_result.
        # If Q_result is A-orthonormal, the projection coefficients are given by Q_result.T @ (A @ u).
        for _ in range(20):
            if Q_result.shape[1] > 0:
                proj_coeff = Q_result.T @ (A(u))
                u = u - Q_result @ proj_coeff

        # Check the residual norm in the A-inner product.
        norm_u = np.sqrt(u.T @ A(u))
        if norm_u > droptol:
            u = u / norm_u
            Q_result = np.column_stack((Q_result, u))
            added += 1
        else:
            dropped += 1

    if verbose:
        print(f"A-orthogonalization complete: {added} vectors added, {dropped} vectors dropped (threshold: {droptol})")

    return Q_result


def gmres_deflated_preconditioned(A, b, M, maxits=30, tol=1e-8, Z=None, x0=None):
    """
    GMRES with preconditioning and deflation (single cycle version).

    Parameters:
        A: function or (n,n) ndarray
           Matrix-vector product function or matrix A.
        b: (n,) ndarray
           Right-hand side vector.
        M_solve: function
           Preconditioner solver (i.e. returns M⁻¹ * vector).
        Z: (n,k) ndarray
           Orthonormal deflation matrix.
        x0: (n,) ndarray
           Initial guess.
        tol: float, optional
           Convergence tolerance.
        max_iter: int, optional
           Maximum number of GMRES iterations.

    Returns:
        x: (n,) ndarray
           Computed solution vector.
        iter_count: int
           Total number of iterations.
        res_history: list of float
           History of the full residual norm, ||b - A(x)||.
    """

    # Define the deflation projector as a function: P(v) = v - Z (Zᵀ v)
    P = (lambda v: v) if (Z is None) else (lambda v: v - Z @ (Z.T @ v))

    # Initial guess and residual history
    x = np.zeros_like(b) if (x0 is None) else x0.copy()

    r0 = b - A(x)
    normb = np.linalg.norm(b)
    normb = 1.0 if normb == 0 else normb
    res_history = [norm(r0) / normb]
    iter_count = 0

    if res_history[-1] < tol:
        return x, iter_count, res_history
    # Compute initial projected residual
    r = P(r0)
    z = M(r)
    beta = norm(z)

    n = len(b)
    V = np.zeros((n, maxits + 1))
    H = np.zeros((maxits + 1, maxits))
    V[:, 0] = z / beta

    # GMRES Arnoldi process
    for j in range(maxits):
        iter_count += 1

        # Apply A and preconditioner then project:
        w = P(M(A(V[:, j])))

        # Modified Gram-Schmidt orthogonalization:
        for i in range(j + 1):
            H[i, j] = np.dot(w, V[:, i])
            w -= H[i, j] * V[:, i]

        H[j + 1, j] = norm(w)

        if H[j + 1, j] < 1e-14:
            # Happy breakdown: Krylov space has been fully spanned
            H = H[:j + 2, :j + 1]
            V = V[:, :j + 2]
            break

        V[:, j + 1] = w / H[j + 1, j]

        # Solve the least-squares problem min_y || beta*e1 - H*y ||
        e1 = np.zeros(j + 2)
        e1[0] = beta
        y, _, _, _ = np.linalg.lstsq(H[:j + 2, :j + 1], e1, rcond=None)
        x_new = x + V[:, :j + 1] @ y

        # Record the full residual norm
        current_res = norm(b - A(x_new)) / normb
        res_history.append(current_res)
        x = x_new

        if current_res < tol:
            break

    # Deflation correction:
    if Z is not None:
        k = Z.shape[1]
        AZ = np.column_stack([A(Z[:, i]) for i in range(k)])

        rhs = Z.T @ (b - A(x))
        u = solve(Z.T @ AZ, rhs)
        x = x + Z @ u

    res_history.append(norm(b - A(x)))

    return x, iter_count, res_history
