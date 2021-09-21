# Module for Symmetry operations and Gauge freedom


import numpy as np
from scipy.sparse import diags # Used for banded matrices
from Module_Utilities import rename


###########################################################################################################
#TODO: possibly implement triu or tril for hamilton matrix
@rename("H")
def Hopping_Matrix_with_Phase(n = 6, phase=None, **kwargs):
    """
    TODO: write documentation
    """
    
    ### Check if system is large enough, i.e. if n=>2
    assert n >= 2, "error n must be greater or equal to 2"

    diagonal_entries = [np.ones(n-1), np.ones(n-1)]
    H = diags(diagonal_entries, [-1, 1]).toarray()
    #TODO: check length of phase vector
    #TODO: check if phase vector is real
    
    # take care of the values not on the lower and upper main diagonal
    H[[0, n-1], [n-1, 0]] = 1
    
    if phase is not None:
        phase = np.array(phase)
        H = H.astype(complex) #otherwise float conversion error
        #Calculte H * exp(1*theta_ij)
        H *= np.exp(1j*(phase[None, :] - phase[:, None]))

    return H

###########################################################################################################
def Show_Hamiltonian_Gaugefreedom(**kwargs):
    H = Hopping_Matrix_with_Phase(n=kwargs.get("n", 6), phase=kwargs.get("phase", None))
    precision = kwargs.get("precision", 2)
    print(f"H = ", np.round(H, precision), "", sep="\n")
    eigvals = np.linalg.eigvalsh(H)
    print(f"Eigenvalues of H: {np.round(eigvals, precision+1)}")
    

###########################################################################################################
@rename("M")    
def Magnetic_Flux_Matrix(n = 6, **kwargs):
    """
    TODO: write documentation
    """

    ### Check if system is large enough, i.e. if n=>2
    assert n >= 2, "error n must be greater or equal to 2"#
    
    diagonal_entries = [np.ones(n-1)*np.exp(-1j), np.ones(n-1)*np.exp(1j)]
    M = diags(diagonal_entries, [-1, 1]).toarray()
    #print(np.round(M))
    # take care of the values not on the lower and upper main diagonal
    M[0, n-1] = np.exp(-1j)
    #print(np.round(M))
    M[n-1, 0] = np.conj(M[0, n-1])
    #print(np.round(M))

    return M

###########################################################################################################
def Show_Magnetic_Flux(**kwargs):
    M = Magnetic_Flux_Matrix(n=kwargs.get("n", 6))
    H = Hopping_Matrix_with_Phase(n=kwargs.get("n", 6))
    
    precision = kwargs.get("precision", 2)
    print(f"M = ", np.round(M, precision), "", sep="\n")
    eigvals_M = np.linalg.eigvalsh(M)
    print(rf"Eigenvalues of M: {np.round(eigvals_M, precision+1)}")
    eigvals_H = np.linalg.eigvalsh(H)
    print(rf"Eigenvalues of H: {np.round(eigvals_H, precision+1)}")
    


###########################################################################################################
@rename("T")
def Right_Translation_Matrix(n=6, turns=1, show=False, **kwargs):
    """
    TODO: write documentation
    """

    ### Check if system is large enough, i.e. if n=>2
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
    T_group = np.array([np.linalg.matrix_power(T, exp) for exp in np.arange(n)])
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
def Show_Commutator(A, B, name_A, name_B, **kwargs):
    M1 = A(**kwargs)
    M2 = B(**kwargs)
    precision = kwargs.get("precision", 2)
    print(f"[{name_A},{name_B}] = ", f"{np.round(Commutator(M1, M2), precision)}", sep="\n")
    
    eigvals_AB = np.linalg.eigvalsh(M1 @ M2)
    eigvals_BA = np.linalg.eigvalsh(M2 @ M1)
    eigvals_A = np.linalg.eigvalsh(M1)
    print("")
    print(f"Eigenvalues of {name_A}: {np.round(eigvals_A, precision+1)}")
    print(f"Eigenvalues of {name_A}{name_B}: {np.round(eigvals_AB, precision+1)}")
    print(f"Eigenvalues of {name_B}{name_A}: {np.round(eigvals_BA, precision+1)}")
    

###########################################################################################################
@rename("R")
def Reflection_Matrix(n=6, axis=1, show=False, **kwargs):
    """
    TODO: write documentation
    """

    ### Check if system is large enough, i.e. if n=>2
    assert n >= 2, "error n must be greater or equal to 2"
    
    R =  np.roll(np.fliplr(np.eye(n)), shift=axis, axis=0)
    if show:
        print(R)
    return R

###########################################################################################################
def All_Reflections(n=6):
    return np.array([Reflection_Matrix(n, vertex=i+1) for i in np.arange(n)])




def Show_Commutator_2(A, B, name_A, name_B, axis1, axis2, **kwargs):
    A = A(axis=axis1, **kwargs)
    B = B(axis=axis2, **kwargs)
    precision = kwargs.get("precision", 2)
    print(f"[{name_A},{name_B}] = ", f"{np.round(Commutator(A, B), precision)}", sep="\n")
    
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
    print("R1 R2 = ", R1 @ R2, sep ="\n")