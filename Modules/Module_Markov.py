#Module for Markov Processes 
import numpy as np
from scipy.sparse import diags # Used for banded matrices
import math # used for `isclose`

import matplotlib as mpl
import matplotlib.pyplot as plt # Plotting
from cycler import cycler #used for color cycles in mpl


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

    # Ensure probabilities are non-negative and bounded
    assert 2 * (p1 + p2) <= 1, f"For consistency, twice the sum of p1 and p2 has to be at most 1, you have 2 * (p1 + p2) = {2 * (p1 + p2):.3f}."
    assert p1 >= 0 and p2 >= 0, f"Probabilities have to be non-negative, you have p1 = {p1:.3f} and p2 = {p2:.3f}"

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

###########################################################################################################
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


###########################################################################################################
def Plot_Markov(state=[1,0,0,0,0,0], n_its=400, **kwargs):
    #Initial_State.options = states_dict[kwargs.get("n", 6)]
    observations = Calc_Markov(state, n_its, **kwargs)
    n = kwargs.get("n", 6)
    fig = plt.figure(figsize=(10,6))

    plt.title(f"Markov evolution of the $n={n}$-ring with initial state {state} and $p_1 = {kwargs.get('p1', 0.1)}$")
    plt.xlabel(r"Number of iterations $n_{\mathrm{its}}$")
    plt.ylabel(r"Probability of finding particle at site $i$")
    plt.grid()

    mpl.rcParams['axes.prop_cycle'] = cycler("color", plt.cm.get_cmap("tab10").reversed().colors[-n:])
    colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']

    for i, site in enumerate(np.arange(n)[::-1]):
        plt.plot(observations[:, site], ".-", label=f"Site {site+1}", color=colors[i])
    legend = plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.show()
    return fig


###########################################################################################################
def Markov_Path(initial_position=1, p1=0.1, n_its=400, **kwargs):
    """write documentation"""

    # Ensure probabilities are non-negative and bounded
    assert 2 * p1 <= 1, f"For consistency, twice p1 has to be at most 1, you have 2 * p1 = {2 * p1:.3f}."
    assert p1 >= 0, f"Probabilities have to be non-negative, you have p1 = {p1:.3f}."

    Number_of_paths = kwargs.get("n_paths", 1000)
    n = kwargs.get("n", 6)
    
    initial_positions_arr = np.ones(Number_of_paths) * (initial_position - 1)
    
    #Random number Generator `rng` and Probability Vector `pvec`
    rng = np.random.default_rng(kwargs.get("seed", 42))
    p_vec = np.zeros((n_its, n))
    
    #Calculate probability of outcome for ever timestep `step`
    for step in np.arange(n_its):
        random_vector = rng.random(Number_of_paths)
        initial_positions_arr = (np.where(random_vector <= p1, initial_positions_arr+1,
                                    np.where(random_vector >= (1-p1), initial_positions_arr-1, initial_positions_arr))) % n
        p_vec[step] =  np.bincount(initial_positions_arr.astype(int), minlength=n) / Number_of_paths
    return p_vec


###########################################################################################################
def Plot_Markov_Path(initial_position=1, p1=0.1, n_its=400, **kwargs):
    #Initial_init_arr.options = init_arrs_dict[kwargs.get("n", 6)]
    #TODO: link n with position 1 
    observations = Markov_Path(initial_position=initial_position, n_its=n_its, p1=p1, **kwargs)
    n = kwargs.get("n", 6)
    
    fig = plt.figure(figsize=(10,6))
    
    plt.title(f"Markov evolution of the $n={n}$-ring, particle initialy at position {initial_position} using explicit paths, $p_1 = {p1}$")
    plt.xlabel(r"Number of iterations $n_{\mathrm{its}}$")
    plt.ylabel(r"Probability of finding particle at site $i$")
    plt.grid()
    
    mpl.rcParams['axes.prop_cycle'] = cycler("color", plt.cm.get_cmap("tab10").reversed().colors[-n:])

    for site in np.arange(n)[::-1]:
        plt.plot(observations[:, site], ".-", label=f"Site {site+1}")
    legend = plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    
    plt.show()
    return fig