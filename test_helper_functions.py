import pyamg
import h5py
from operators import BlockMatrixOperator, BlockTriangularPreconditioner
from iterative_solvers import fgmres, dcg
import numpy as np
from numpy.linalg import norm
import time


def test_fgmres(A_blocks, Ptblocks, rhs_u, sol_u, rhs_p, sol_p, buffer_size=30,
                type="coarse", use_dcg=False, tol=(1e-6, 1e-6)):
    # lists for storing parts for coarse space
    iter_list = []
    err_list = []
    Z = None
    AZ = None
    Z_list = []
    AZ_list = []

    # initial guess
    x_sol = None
    x0 = None

    # time collection
    times_coarse_solver = []
    times_fgmres = []
    times_overhead = []

    # rebuild objects to clear up storage
    A = BlockMatrixOperator(A_blocks)
    M = BlockTriangularPreconditioner(Ptblocks, False, use_dcg, tol)

    for index in range(1, 716):
        rhs_vec = np.concatenate([rhs_u[index], rhs_p[index]])
        sol_vec = np.concatenate([sol_u[index], sol_p[index]])

        if Z is not None and type == "coarse":
            start = time.time()
            y, *_ = np.linalg.lstsq(AZ, rhs_vec, rcond=None)
            x0 = Z @ y
            duration = time.time() - start
            times_coarse_solver.append(duration)

        start = time.time()
        if type == "coarse":
            x0_in = x0
        if type == "previous":
            x0_in = x_sol
        if type == "zero":
            x0_in = None
        x_sol, num_iterations, residuals = fgmres(A, rhs_vec, M, maxits=20, tol=1e-10, x0=x0_in)
        duration = time.time() - start

        times_fgmres.append(duration)
        iter_list.append(num_iterations)
        err_list.append(norm(x_sol - sol_vec) / norm(sol_vec))

        if norm(x_sol) > 0 and num_iterations > 0 and type == "coarse":
            start = time.time()
            Z_list.append(x_sol)
            AZ_list.append(A(x_sol))
            Z = np.column_stack(Z_list[-buffer_size:])
            AZ = np.column_stack(AZ_list[-buffer_size:])
            duration = time.time() - start
            times_overhead.append(duration)

        print(".", end="") if index % 100 != 0 else print(".")
    return iter_list, err_list, times_coarse_solver, times_fgmres, times_overhead, M


def test_dcg(M, tol, block="u", type="y"):

    if block == "u":
        mat1 = M.blocks[0]
        probs1 = M.upper_stack
    elif block == "p":
        mat1 = M.blocks[3]
        probs1 = M.lower_stack

    # Load the elastic_kernel.h5 file
    with h5py.File('elastic_kernel.h5', 'r') as f:
        # Print the keys to see what's in the file
        R = f['R'][:]

    if block == "u":
        ml = pyamg.smoothed_aggregation_solver(
            mat1,
            B=R,
            strength=('symmetric', {'theta': 0.1}),
            smooth=('energy', {'degree': 2}),
            presmoother=('gauss_seidel', {'sweep': 'symmetric'}),
            postsmoother=('gauss_seidel', {'sweep': 'symmetric'}),
            improve_candidates=[('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 4})],
            max_coarse=500
        )
    elif block == "p":
        ml = pyamg.smoothed_aggregation_solver(
            mat1,
            strength=('symmetric', {'theta': 0.1}),
            smooth=('energy', {'degree': 2}),
            presmoother=('gauss_seidel', {'sweep': 'symmetric'}),
            postsmoother=('gauss_seidel', {'sweep': 'symmetric'}),
            improve_candidates=[('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 4})],
            max_coarse=500
        )

    M_lin = ml.aspreconditioner()

    all_vecs = []
    A_all_vecs = []
    all_vecs_norms = []
    W = None
    AW = None
    times = []
    all_iters = []
    all_errors = []
    for index in range(len(probs1)):  # 157
        rhs = probs1[index]['b']
        true_sol = probs1[index]['u']

        start = time.time()
        x, j, resvec, tag, p_list, Ap_list, pnorm_list, y = dcg(
            A=mat1, b=rhs, M=M_lin, W=W, AW=AW, tol=tol, maxiter=100, is_W_A_orthonormal=True)

        if type == "y":
            if y is not None:
                A_y = mat1 @ y
                y_norm = np.sqrt(np.dot(A_y, y))
                y = y / y_norm
                A_y = A_y / y_norm
                all_vecs.append(y)
                A_all_vecs.append(A_y)
                all_vecs_norms.append(y_norm)
                W = np.column_stack(all_vecs)
                AW = np.column_stack(A_all_vecs)
        if type == "krylov":
            all_vecs += p_list
            A_all_vecs += Ap_list
            all_vecs_norms += pnorm_list
            W = np.column_stack(all_vecs)
            AW = np.column_stack(A_all_vecs)

        duration = time.time() - start
        times.append(duration)
        all_iters.append(j)
        all_errors.append(np.linalg.norm(x - true_sol) / np.linalg.norm(true_sol))
        print(f"Index {index}: {j}, {resvec[-1]}, {all_errors[-1]}")
    return all_iters, all_errors, times
