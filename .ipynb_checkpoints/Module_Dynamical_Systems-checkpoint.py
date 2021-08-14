

import ipywidgets as widgets
import numpy as np
from scipy.sparse import diags # Used for banded matrices
import math # used for `isclose`

###########################################################################################################
def check_if_int(string):
    """
    Convert decimal value of `string` as integer/float/complep1 datatype without additional zeros after comma (e.g. "3.0" => 3).

    Parameters:
    -----------
    string : str

    Returns:
    --------
    value
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
def Hopping_Matrix(n = 6):
    """
    TODO: write documentation
    """

    ### Check if system is large enough, i.e. if n=>2
    assert n >= 2, "error n must be greater or equal to 2"

    diagonal_entries = [np.ones(n-1), np.ones(n-1)]
    H = diags(diagonal_entries, [-1, 1]).toarray()

    # take care of the values not on the lower and upper main diagonal
    H[[0, n-1], [n-1, 0]] = 1

    return H


###########################################################################################################
def Transfer_Matrix(n=6, p1=0.01, p2=0.):
    """
    TODO: implement next neighbor hopping
    TODO: write documentation
    TODO: possibly extend to hopping on all places
    TODO: python assertion with markdown, ask question on stack overflow
    """

    # Ensure probabilities are non-negative
    assert 2 * (p1 + p2) <= 1, f"For consistency, twice the sum of p1 and p2 has to be at most 1, you have 2 * (p1 + p2) = {2 * (p1 + p2):.3f}"

    # n=3 NN hopping for n=3 is equal to normal hopping
    if p2 and n > 3:
        return np.eye(n) * (1 - 2*(p1 + p2)) + Hopping_Matrix(n) * p1 + NN_Hopping_Matrix(n) * p2
    elif n == 2:
        # There is only one p1 for n==2:
        return np.eye(n) * (1 - p1) + Hopping_Matrix(n) * p1
    else:
        return np.eye(n) * (1 - 2*p1) + Hopping_Matrix(n) * p1


###########################################################################################################
def NN_Hopping_Matrix(n = 6):
    """
    TODO: write documentation
    TODO: possibly extend with arbitrary neighbor hopping, beware of double hopping errors
    """

    ### Check if system is large enough, i.e. if n=>3
    assert n >= 3, "error n must be greater or equal to 2"
    # due to symmetrie, 4x4 next neighbor hopping introduces errors if not handled with care
    if n == 4:
        diagonal_entries = [np.ones(n-2), np.ones(n-2)]
        return diags(diagonal_entries, [-2, 2,]).toarray()
    else:
        diagonal_entries = [[1, 1], np.ones(n-2), np.ones(n-2), [1, 1]]
        return diags(diagonal_entries, [-n+2, -2, 2, n-2]).toarray()


def Calc_Markov(state=[1,0,0,0,0,0], n_its=400, **kwargs):
    """TODO: add doc string"""
    #Check if state is valid
    assert not any(isinstance(num, complex) for num in state), f"Markovian evolution cannot deal with complex state {state}"
    assert math.isclose(sum(state), 1, rel_tol=1e-04), f"The norm of the state vector {state} = {sum(state)} != 1"

    T = Transfer_Matrix(**kwargs)
    state = np.array(state)
    observations = [state]
    for _ in np.arange(n_its):
        state = T @ state
        observations.append(state)
    return np.array(observations)
