# Module for Symmetry operations and Gauge freedom


import numpy as np
from scipy.sparse import diags  # Used for banded matrices
from ..General.Module_Utilities import rename, print_isreal, Sorted_Eig, Eig, is_unique

# possible TODOS:
# rename `print` functons into `calc` functions
# TODO add swithc to change order of T and R in `Show commutator`
# TODO change name of `right Translation matrix` to `Translation matrix`
# TODO remove text variable in `diagonalize`

###########################################################################################################
# TODO: possibly implement triu or tril for hamilton matrix


@rename("H")
def Hopping_Matrix_with_Phase(n=6, phase=None, **kwargs):
    """
    TODO: write documentation
    """

    # Check if system is large enough, i.e. if n=>2
    assert n >= 2, "error n must be greater or equal to 2"

    diagonal_entries = [np.ones(n-1), np.ones(n-1)]
    H = diags(diagonal_entries, [-1, 1]).toarray()
    # TODO: check length of phase vector
    # TODO: check if phase vector is real

    # take care of the values not on the lower and upper main diagonal
    H[[0, n-1], [n-1, 0]] = 1

    if phase is not None:
        phase = np.array(phase)
        H = H.astype(complex)  # otherwise float conversion error
        # Calculte H * exp(1*theta_ij)
        H *= np.exp(1j*(phase[None, :] - phase[:, None]))

    return H

###########################################################################################################


def Show_Hamiltonian_Gaugefreedom(**kwargs):
    H = Hopping_Matrix_with_Phase(n=kwargs.get(
        "n", 6), phase=kwargs.get("phase", None))
    _precision = kwargs.get("precision", 2)
    print(f"H = ", print_isreal(H).round(decimals=_precision), "", sep="\n")
    eigvals = np.linalg.eigvalsh(H)
    print(f"Eigenvalues of H: {np.round(eigvals, _precision+1)}")


###########################################################################################################
@rename("M")
def Magnetic_Flux_Matrix(n=6, **kwargs):
    """
    TODO: write documentation
    """

    phi = kwargs.get("phi", np.ones(n))
    phi = np.array(phi)
    # Check if system is large enough, i.e. if n=>2
    assert n >= 2, "error n must be greater or equal to 2"

    # print(phi)
    diagonal_entries = [np.exp(-1j*phi[:-1]), np.exp(1j*phi[:-1])]
    M = diags(diagonal_entries, [-1, 1]).toarray()
    # print(np.round(M))
    # take care of the values not on the lower and upper main diagonal
    M[0, n-1] = np.exp(-1j*phi[-1])
    # print(np.round(M))
    M[n-1, 0] = np.conj(M[0, n-1])
    # print(np.round(M))

    return M

###########################################################################################################


def Show_Magnetic_Flux(**kwargs):

    M = Magnetic_Flux_Matrix(**kwargs)
    H = Hopping_Matrix_with_Phase(**kwargs)

    precision = kwargs.get("precision", 2)
    print(f"M = ", np.round(M, precision), "", sep="\n")
    eigvals_M = np.linalg.eigvalsh(M)
    print(f"Eigenvalues of M: {np.round(eigvals_M, precision+1)}")
    eigvals_H = np.linalg.eigvalsh(H)
    print(f"Eigenvalues of H: {np.round(eigvals_H, precision+1)}")

    _n = kwargs.get("n", 6)
    _phi = kwargs.get("phi", np.ones(_n))
    print(f"Total phase = {np.sum(_phi).round(2)}")


###########################################################################################################
@rename("T")
def Right_Translation_Matrix(n=6, turns=1, show=False, **kwargs):
    """
    TODO: write documentation
    """

    # Check if system is large enough, i.e. if n=>2
    assert n >= 2, "error n must be greater or equal to 2"

    T = np.roll(np.eye(n), shift=turns, axis=1)
    if show:
        print(f"T = ", T, sep="\n")
    return T


###########################################################################################################
def Translation_Group(n=6, show=False):
    """
    TODO: write documentation
    """
    T = Right_Translation_Matrix(n=n, show=False)
    T_group = np.array([np.linalg.matrix_power(T, exp)
                       for exp in np.arange(n)])
    if show:
        print(T_group)
    return T_group


###########################################################################################################
def Commutator(A, B):
    """
    TODO: write documentation
    """
    return A @ B - B @ A


###########################################################################################################
def Show_Commutator(A, B, **kwargs):
    M1 = A(**kwargs)
    M2 = B(**kwargs)
    _precision = kwargs.get("precision", 2)

    if hasattr(A, "name") and hasattr(B, "name"):
        print(f"[{A.name},{B.name}] = ",
              f"{print_isreal(Commutator(M1, M2)).round(_precision)}", sep="\n")

    eigvals_AB = np.linalg.eigvalsh(M1 @ M2)
    eigvals_BA = np.linalg.eigvalsh(M2 @ M1)
    eigvals_A = np.linalg.eigvalsh(M1)
    print("")

    show_eigvals = kwargs.get("show_eigvals", True)
    if show_eigvals:
        if hasattr(A, "name") and hasattr(B, "name"):
            print(
                f"Eigenvalues of {A.name}: {np.round(eigvals_A, _precision+1)}")
            print(
                f"Eigenvalues of {A.name}{B.name}: {np.round(eigvals_AB, _precision+1)}")
            print(
                f"Eigenvalues of {B.name}{A.name}: {np.round(eigvals_BA, _precision+1)}")


###########################################################################################################
@rename("R")
def Reflection_Matrix(n=6, axis=1, show=False, **kwargs):
    """
    TODO: write documentation
    """

    # Check if system is large enough, i.e. if n=>2
    assert n >= 2, "error n must be greater or equal to 2"

    R = np.roll(np.fliplr(np.eye(n)), shift=axis, axis=0)
    if show:
        print(R)
    return R

###########################################################################################################


def All_Reflections(n=6):
    return np.array([Reflection_Matrix(n, vertex=i+1) for i in np.arange(n)])


def Show_Commutator_Rotations(A, B, axis1, axis2, **kwargs):
    M1 = A(axis=axis1, **kwargs)
    M2 = B(axis=axis2, **kwargs)
    precision = kwargs.get("precision", 2)

    if hasattr(A, "name") and hasattr(B, "name"):
        print(f"[{A.name}1,{B.name}2] = ",
              f"{np.round(Commutator(M1, M2), precision)}", sep="\n")

    # eigvals_AB = np.linalg.eigvalsh(A @ B)
    # eigvals_BA = np.linalg.eigvalsh(B @ A)
    # eigvals_A = np.linalg.eigvalsh(A)
    # print("")
    # print(f"Eigenvalues of {name_A}: {np.round(eigvals_A, precision+1)}")
    # print(f"Eigenvalues of {name_A}{name_B}: {np.round(eigvals_AB, precision+1)}")
    # print(f"Eigenvalues of {name_B}{name_A}: {np.round(eigvals_BA, precision+1)}")


def Show_R1_R2(R1, R2, axis1, axis2, **kwargs):
    R1 = R1(axis=axis1, **kwargs)
    R2 = R2(axis=axis2, **kwargs)
    print("R1 R2 = ", R1 @ R2, sep="\n")


def Show_A_B(A, B, **kwargs):
    M1 = A(**kwargs)
    M2 = B(**kwargs)
    _precision = kwargs.get("precision", 2)

    if hasattr(A, "name") and hasattr(B, "name"):
        print(f"{A.name} {B.name} = ",
              f"{np.round(M1 @ M2, _precision)}", sep="\n")


def Sort_Eigenvalues_And_Eigenvectors(vals, vecs):
    idx = vals.argsort()
    vals = vals[idx]
    vecs = vecs[:, idx]
    return vals, vecs


def Print_Eigenvalues_And_Eigenvectors(A, **kwargs):
    _M1 = A(**kwargs)
    _precision = kwargs.get("precision", 2)
    eigvals, eigvecs = Sorted_Eig(_M1)

    if hasattr(A, "name"):
        print(
            f"Eigenvalues of {A.name}: {np.round(eigvals, _precision+1)}", "", sep="\n")
        print(f"Eigenvectors of {A.name}:", np.round(
            eigvecs, _precision), sep="\n")


def Diagonalize_A_With_B(A, B, **kwargs):
    _precision = kwargs.get("precision", 2)
    M1 = A(**kwargs)
    M2 = B(**kwargs)
    eigvals, U_B = Sorted_Eig(M2)
    D_A = print_isreal(np.round(U_B.conj().T @ M1 @ U_B, _precision)+0.)

    if hasattr(A, "name") and hasattr(B, "name"):
        print(f"Eigenvalues of {B.name}: {np.round(eigvals, _precision)}")
        print(
            f"Eigenvalues of {A.name}: {np.round(np.sort(np.linalg.eigvalsh(M1)), _precision)}", "", sep="\n")
        print(f"D = U^t {A.name} U = ", D_A, sep="\n")


def Diagonalize(A, B, text=True, **kwargs):
    state = None
    # TODO replce all np.rond with.round for convenience
    # TODO give all relevent functions specific .name attributes

    # check if A and B Commute, i.e. [A,B]==0
    assert np.count_nonzero(Commutator(
        A, B)) == 0, "Error, matrices do not commute => cannot be simultaneously diagonalized."

    # calc (possibly real) eigenvalues
    lam_A, U_A = Sorted_Eig(A)
    lam_B, U_B = Sorted_Eig(B)

    # check if A == B
    if np.array_equal(A, B):
        if text:
            print("Identical matrices => U_A = U_AB = U_B")
        D_A = U_A.conj().T @ A @ U_A
        state = "A==B"
        return U_A, D_A, None, None, state

    # check if A's or B's eigenvalues are unique => U_A or U_B is sufficient to diagonalize both A and B
    if is_unique(lam_A):
        if text:
            print("Unique eigenvalues of A => U_A = U_AB")
        D_A = U_A.conj().T @ A @ U_A
        D_B = U_A.conj().T @ B @ U_A
        state = "A"
        return U_A, D_A, D_B, None, state

    if is_unique(lam_B):
        if text:
            print("Unique eigenvalues of B => U_B = U_AB")
        D_A = U_B.conj().T @ A @ U_B
        D_B = U_B.conj().T @ B @ U_B
        state = "B"
        return U_B, D_A, D_B, None, state

    # degenerate Eigenvalues
    # Has to be rounded, else numeric errors occure
    B_Block = np.round(U_A.conj().T @ B @ U_A, 6)
    _, V_B = Eig(B_Block)
    U_AB = U_A @ V_B
    D_A = U_AB.conj().T @ A @ U_AB
    D_B = U_AB.conj().T @ B @ U_AB

    return U_AB, D_A, D_B, B_Block, state


def Show_Diagonalize(A, B, text=False, **kwargs):
    precision = kwargs.get("precision", 2)
    #n = kwargs.get("n", 6)
    U_AB, D_A, D_B, B_Block, state = Diagonalize(
        A(**kwargs), B(**kwargs), text=text)

    if hasattr(A, "name") and hasattr(B, "name"):
        if state == "A==B":
            print(
                f"Identical matrices => U_{A.name} = U_{A.name}{B.name} = U_{B.name}", "", sep="\n")
        elif state == "A":
            print(
                f"Unique eigenvalues of {A.name} => U_{A.name} = U_{A.name}{B.name}", "", sep="\n")
        elif state == "B":
            print(
                f"Unique eigenvalues of {B.name} => U_{B.name} = U_{A.name}{B.name}", "", sep="\n")

        # +0. added to avoid "-0." in output due to rounding of negative floats
        print(f"U_{A.name}{B.name} = ", U_AB.round(precision)+0., "", sep="\n")
        print(f"D_{A.name} ", print_isreal(
            D_A.round(precision)+0.), "", sep="\n")
        if D_B is not None:
            print(f"D_{B.name} = ", print_isreal(
                D_B.round(precision)+0.), "", sep="\n")
        if B_Block is not None:
            print(f"{B.name}_Block = ", B_Block.round(precision)+0., sep="\n")
    else:
        print("error")
