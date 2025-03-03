from scipy.sparse import csr_matrix
import petsc4py
from petsc4py import PETSc

def extract_2x2submatrices(matA, spaceV):
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
        # mat.convert("aij")  # for big matrices
        mat.convert("dense")  # for small matrices
        # mat_data = mat.getValuesCSR()  # for big matrices
        # mat_data = csr_matrix(mat_data)  # for big matrices
        mat_data = mat.getDenseArray()  # for small matrices
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

def extract_matrix(matA):
    matA.convert("dense")  # for small matrix
    # mat.convert("aij")  # for big matrices
    mat_data = matA.getDenseArray()  # for small matrix
    # mat_data = mat.getValuesCSR()  # for big matrices
    # mat_data = csr_matrix(mat_data)
    return mat_data