from scipy.sparse import csr_matrix
import petsc4py
import h5py
import numpy as np
from petsc4py import PETSc

def extract_2x2submatrices(matA, spaceV, sparse=True):
    V0_dofs = spaceV.sub(0).dofmap.list.flatten()
    V1_dofs = spaceV.sub(1).dofmap.list.flatten()

    V0_dofs = np.unique(V0_dofs)
    V1_dofs = np.unique(V1_dofs)

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
            indpts, indices, data = mat.getValuesCSR()  # naming taken from scipy docs
            mat_data = (data, indices, indpts)
        else:
            mat.convert("dense")
            mat_data = mat.getDenseArray()
        res.append(mat_data)
    return res

def extract_2x2subvectors(vecb, spaceV):
    V0_dofs = spaceV.sub(0).dofmap.list.flatten()
    V1_dofs = spaceV.sub(1).dofmap.list.flatten()

    V0_dofs = np.unique(V0_dofs)
    V1_dofs = np.unique(V1_dofs)

    is_u = PETSc.IS().createGeneral(V0_dofs)
    is_p = PETSc.IS().createGeneral(V1_dofs)


    vec_u = vecb.getSubVector(is_u)
    vec_p = vecb.getSubVector(is_p)

    return [vec_u.getArray().copy(), vec_p.getArray().copy()]

def extract_matrix(matA, sparse=True):
    if sparse:
        matA.convert("aij")
        indpts, indices, data = matA.getValuesCSR()
        mat_data = (data, indices, indpts)
    else:
        matA.convert("dense")
        mat_data = matA.getDenseArray()
    return mat_data

def save_matrices_to_hdf5(filename, matrices, sparse=True):
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

def load_matrices_from_hdf5(filename, sparse=True):
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

def save_full_matrix(matA, filename, sparse=True):
    mat_data = extract_matrix(matA, sparse)
    save_matrices_to_hdf5(filename, [mat_data])
    print(f'Matrix saved to {filename}')

def save_matrix_as_blocks(matA, spaceV, filename, sparse=True):
    matrices = extract_2x2submatrices(matA, spaceV, sparse)
    save_matrices_to_hdf5(filename, matrices)
    print(f'Matrix saved as blocks to {filename}')


def save_vectors_to_hdf5(filename, vec_u, vec_p, step):
    """ generated code """
    with h5py.File(filename, "a") as f:
        for name, vec in zip(["vec_u", "vec_p"], [vec_u, vec_p]):
            if name in f:
                dset = f[name]
                dset.resize((step + 1, vec.shape[0]))  # Expand along the first axis
            else:
                dset = f.create_dataset(name, shape=(step + 1, vec.shape[0]), maxshape=(None, vec.shape[0]),
                                        dtype=vec.dtype)

            dset[step, :] = vec  # Store the new time step

def load_vectors_from_hdf5(filename):
    """ generated code """
    with h5py.File(filename, "r") as f:
        vec_u = f["vec_u"][:]  # Load full dataset
        vec_p = f["vec_p"][:]  # Load full dataset

    # Restore as list of arrays, one per timestep
    vec_u_list = [vec_u[i, :] for i in range(vec_u.shape[0])]
    vec_p_list = [vec_p[i, :] for i in range(vec_p.shape[0])]

    return vec_u_list, vec_p_list

def test_save_and_load():
    pass  # TODO

if __name__ == "__main__":
    test_save_and_load()
