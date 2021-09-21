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
    Check if matrix 'M' is hermitian, i.e. iff M.H == M. We do not check every entry explicitely, but rather if the maximum column sum of (M.H - M) is lower than the threshold `tol`.

    Parameters:
    -----------
    M : ndarray
        Matrix to check hermiticity

    tol : float, optional
        Decimal tolarance for the hermiticity check. The default is `1e-8`

    Returns:
    --------
    bool
    """
    return np.linalg.norm(A-A.conj().T, np.Inf) < tol;


###########################################################################################################
def is_unique(arr, precision=5):
    """
    Check if all values of `arr` are unique. Rounding due to numeric errors in `np.unique`, which treats almost identical numbers as unique
    
    Parameters
    ----------
    arr : ndarray
        Array to check for unique values.
    
    precision : int
        Defines the number of significant decimal digits for rounding the values of `arr`. Default is `5`.

    Returns:
    --------
    bool
    """
    return len(np.unique(np.round(arr, precision))) == len(arr)


###########################################################################################################
def isreal(arr, precision=5):
    """
    Check if all values of `arr` are real after rounding to `precision` decimal precision.
    
    Parameters
    ----------
    arr : ndarray
        Array to check for realness.
    
    precision : int
        Defines the number of significant decimal digits for rounding the values of `arr`. Default is `5`.

    Returns:
    --------
    bool
    """
    return np.all(np.isreal(arr.round(precision)))


###########################################################################################################
def print_isreal(arr, precision=5):
    """
    Return `arr` as real arr if all values of `arr` are real after rounding to `precision` decimal precision, else return unaltered array.

    Parameters
    ----------
    arr : ndarray
        Array to check for realness.
    
    precision : int
        Defines the number of significant decimal digits for rounding the values of `arr`. Default is `5`.

    Returns:
    --------
    out : ndarray
        (Possibly real) array of same shape as `arr`
    """
    if isreal(arr, precision):
        return arr.real
    return arr


###########################################################################################################
def Eig(M):
    """
    Calculate eigenvalues and eigenvectors of matrix `M`. Automatically uses `np.linalg.eigh` for the calculation of `M` if it is hermitian.
    
    Parameters
    ----------
    M : (..., n,n) array
        Matrices for which the eigenvalues and right eigenvectors will be computed

    Returns
    -------
    w : (..., n) array
        The eigenvalues, each repeated according to its multiplicity.
        The eigenvalues are not necessarily ordered. The resulting
        array will be of complex type, unless the imaginary part is
        zero in which case it will be cast to a real type. When `a`
        is real the resulting eigenvalues will be real (0 imaginary
        part) or occur in conjugate pairs
    v : (..., n,n) array
        The normalized (unit "length") eigenvectors, such that the
        column ``v[:,i]`` is the eigenvector corresponding to the
        eigenvalue ``w[i]``.

    Documentation taken from "https://github.com/numpy/numpy/blob/v1.21.0/numpy/linalg/linalg.py#L1187-L1333"
    """
    if is_hermitian(M):
        return np.linalg.eigh(M)
    else:
        return np.linalg.eig(M)

    
##########################################################################################################
def Sorted_Eig(M):
    """
    Calculate sorted eigenvalues and eigenvectors of matrix `M`. Eigenvalues and accordingly eigenvectors are sorted by ascending real part of each eigenvalue. Automatically uses `np.linalg.eigh` for the calculation of `M` if it is hermitian.

    Parameters
    ----------
    M : (..., n,n) array
        Matrices for which the eigenvalues and right eigenvectors will be computed

    Returns
    -------
    vals : ndarray, sorted
        Sorted eigenvalues, each repeated according to its multiplicity.

    vecs : ndarray
        Right eigenvectors of `M`.
    """
    vals, vecs = Eig(M) 
    idx = vals.argsort()   
    vals = vals[idx]
    vecs = vecs[:, idx]
    return vals, vecs


##########################################################################################################
def rename(newname):
    """ 
    Decorator to give function abbreviated display name 'newname' as attribute `.name`.
    """
    def decorator(func):
        func.name = newname
        return func
    return decorator