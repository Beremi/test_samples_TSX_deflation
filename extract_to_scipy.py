from scipy.sparse import csr_matrix
import petsc4py
from petsc4py import PETSc

def extract_to_csr(matA, spaceV):
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
        mat.convert("aij")  # mat.convert("dense") for small matrices
        mat_csr = mat.getValuesCSR()  # mat.getDenseArray() for small matrices
        res.append(csr_matrix(mat_csr))
    return res
