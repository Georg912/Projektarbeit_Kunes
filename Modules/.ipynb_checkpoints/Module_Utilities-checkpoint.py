#Module for all kinds of helper functions
import numpy as np


###########################################################################################################
def check_if_int(string):
    """
    Convert decimal value of `string` as integer/float/complex datatype without additional zeros after comma (e.g. "3.0" => 3).

    Parameters:
    -----------
    string : str
        Input string to convert

    Returns:
    --------
    value : int, float or complex
        Converted string 
    """
    value = complex(string)
    if np.isreal(value):
        value = value.real
        if value.is_integer():
            return int(value)
        else:
            return value
    return value


###########################################################################################################
def is_hermitian(A, tol=1e-8):
    """
    Check if matrix 'M' is hermitian <=> M.H == M.

    Parameters:
    -----------
    M : ndarray

    Returns:
    --------
    bool
    """
    return np.linalg.norm(A-A.conj().T, np.Inf) < tol;


###########################################################################################################
def is_unique(arr):
    return len(np.unique(np.round(arr, 3))) == len(arr)


###########################################################################################################
def isreal(arr, precision=5):
    return np.all(np.isreal(arr.round(precision)))


###########################################################################################################
def print_isreal(arr, precision=5):
    if isreal(arr, precision):
        return arr.real
    return arr

###########################################################################################################
def Eig(M):
    if is_hermitian(M):
        return np.linalg.eigh(M)
    else:
        return np.linalg.eig(M)

    
##########################################################################################################
def Sorted_Eig(M):
    vals, vecs = Eig(M) 
    idx = vals.argsort()   
    vals = vals[idx]
    vecs = vecs[:, idx]
    return vals, vecs