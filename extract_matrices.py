from scipy.sparse import csr_matrix
import petsc4py
import h5py
from petsc4py import PETSc

def extract_2x2submatrices(matA, spaceV, sparse=False):
    V0_dofs = spaceV.sub(0).dofmap.list.flatten()
    V1_dofs = spaceV.sub(1).dofmap.list.flatten()
    # TODO: map from dolfinx incides to tight ranges

    is_u = PETSc.IS().createGeneral(V0_dofs)
    is_p = PETSc.IS().createGeneral(V1_dofs)

    Auu = matA.createSubMatrix(is_u, is_u)
    Aup = matA.createSubMatrix(is_u, is_p)
    Apu = matA.createSubMatrix(is_p, is_u)
    App = matA.createSubMatrix(is_p, is_p)

    res = []
    for mat in (Auu, Aup, Apu, App):
        if sparse:
            mat.convert("aij")
            data, indices, indpts = mat.getValuesCSR()  # naming taken from scipy docs
            mat_data = (data, indices, indpts)
        else:
            mat.convert("dense")
            mat_data = mat.getDenseArray()
        res.append(mat_data)
    return res

def extract_2x2subvectors(vecb, spaceV):
    V0_dofs = spaceV.sub(0).dofmap.list.flatten()
    V1_dofs = spaceV.sub(1).dofmap.list.flatten()
    # TODO: map from dolfinx incides to tight ranges

    is_u = PETSc.IS().createGeneral(V0_dofs)
    is_p = PETSc.IS().createGeneral(V1_dofs)


    vec_u = vecb.getSubVector(is_u)
    vec_p = vecb.getSubVector(is_p)

    return [vec_u.getArray().copy(), vec_p.getArray().copy()]

def extract_matrix(matA, sparse=False):
    if sparse:
        matA.convert("aij")
        data, indices, indpts = matA.getValuesCSR()
        mat_data = (data, indices, indpts)
    else:
        matA.convert("dense")
        mat_data = matA.getDenseArray()
    return mat_data

def save_matrices_to_hdf5(filename, matrices, sparse=False):
    """ generated code"""
    with h5py.File(filename, 'w') as f:
        for i, mat in enumerate(matrices):
            if sparse:
                # Save CSR components separately
                f.create_dataset(f"matrix_{i}_data", data=mat[0])
                f.create_dataset(f"matrix_{i}_indices", data=mat[1])
                f.create_dataset(f"matrix_{i}_indptr", data=mat[2])
            else:
                f.create_dataset(f"matrix_{i}", data=mat)

def load_matrices_from_hdf5(filename, sparse=False):
    """ generated code """
    matrices = []
    with h5py.File(filename, 'r') as f:
        i = 0
        while f"matrix_{i}" in f or f"matrix_{i}_data" in f:
            if sparse:
                data = f[f"matrix_{i}_data"][:]
                indices = f[f"matrix_{i}_indices"][:]
                indptr = f[f"matrix_{i}_indptr"][:]
                matrices.append(csr_matrix((data, indices, indptr)))
            else:
                matrices.append(f[f"matrix_{i}"][:])
            i += 1
    return matrices
