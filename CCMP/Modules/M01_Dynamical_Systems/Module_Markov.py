# Module for Markov Processes
import numpy as np
from scipy.sparse import diags  # Used for banded matrices
import math  # used for `isclose`

import matplotlib as mpl
import matplotlib.pyplot as plt  # Plotting
from cycler import cycler  # used for color cycles in mpl

# possible TODOS:
# add p2 to the title of the plots

###########################################################################################################


def Hopping_Matrix(n=6):
    """
    Construct symmetric, square Hopping matrix H_ij in `n` dimensions with periodic boundary conditions.
    Defined by H[i,j] = 1 for all |i-j| = 1, H[0,n-1] = 1 = H[n-1,0] and zero else.

    Parameters
    ----------
    n : integer, optional
        Dimension of the Hopping matrix. Default is 6.

    Returns
    -------
    H : ndarray of shape (n,n)
        The Hopping matrix in `n` dimensions.

    Raises:
    ------
    AssertionError
        If `n` is smaller than 2.

    AssertionError
        if `n` is not integer.
    """

    # Check if system is large enough, i.e. if n=>2
    # Check if `n` is integer
    assert n >= 2, "n must be greater or equal to 2."
    assert type(n) == int, f"n must be an integer not {type(n)}"

    diagonal_entries = [np.ones(n-1), np.ones(n-1)]
    H = diags(diagonal_entries, [-1, 1]).toarray()

    # take care of the values not on the lower and upper main diagonal
    H[[0, n-1], [n-1, 0]] = 1

    return H


###########################################################################################################
def Transfer_Matrix(n=6, p1=0.01, p2=0.):
    """
    TODO: possibly extend to hopping on all places
    Construct the Transfer matrix T_ij in `n` dimensions with periodic boundary conditions.
    Defined by T_ij = p0 I_n + p1 H_ij + p2 HNN_ij, where p0 is given by 1 - 2*(p1+p2) and I_n is the `n` dimensional identity matrix.

    Parameters
    ----------
    n : integer, optional
        Dimension of the Transfer matrix. Default is 6.

    p1 : float, optional
        Probability of hopping once to the left or right. Default is p1 = 0.01.

    p2 : float, optional
        Probability of hopping twice to the left or right. Default is p2 = 0.

    Returns
    -------
    T : ndarray of shape (n,n)
        Transfer matrix T in `n` dimensions

    Raises
    -------
    AssertionError
        If the sum of the hopping probabilities exceeds 1.

    AssertionError
        If probabilities are negative.
    """
    # Ensure probabilities are non-negative and bounded
    assert p1 >= 0 and p2 >= 0, f"Probabilities have to be non-negative. You have p1 = {p1:.3f} and p2 = {p2:.3f}"
    assert 2 * \
        (p1 +
         p2) <= 1, f"For consistency, twice the sum of p1 and p2 has to be at most 1. You have 2 * (p1 + p2) = {2 * (p1 + p2):.3f}."

    # NN hopping for n=3 is equal to normal hopping
    if p2 and n > 3:
        return np.eye(n) * (1 - 2*(p1 + p2)) + Hopping_Matrix(n) * p1 + NN_Hopping_Matrix(n) * p2
    elif n == 2:
        # There is only one p1 for n==2:
        return np.eye(n) * (1 - p1) + Hopping_Matrix(n) * p1
    else:
        return np.eye(n) * (1 - 2*p1) + Hopping_Matrix(n) * p1


###########################################################################################################
def NN_Hopping_Matrix(n=6):
    """
    TODO: write documentation
    TODO: possibly extend with arbitrary neighbor hopping, beware of double hopping errors
    """

    # Check if system is large enough, i.e. if n=>3
    assert n >= 3, "error n must be greater or equal to 2"
    # due to symmetrie, 4x4 next neighbor hopping introduces errors if not handled with care
    if n == 4:
        diagonal_entries = [np.ones(n-2), np.ones(n-2)]
        return diags(diagonal_entries, [-2, 2,]).toarray()
    else:
        diagonal_entries = [[1, 1], np.ones(n-2), np.ones(n-2), [1, 1]]
        return diags(diagonal_entries, [-n+2, -2, 2, n-2]).toarray()


###########################################################################################################
def Calc_Markov(state=[1, 0, 0, 0, 0, 0], n_its=400, **kwargs):
    """
    Calculate the Markovian time evolution of an initial state `state` for `n_its` iterations.
    Done via `state[t] = T^t state[0]`.

    Parameters
    ----------
    state : array_like, optional
        Initial state of the system. Default is `[1,0,0,0,0,0]`.

    n_its : integer, optional
        Number of Markov iterations, such that `n_its` calculations are performed.
        Default is 400.

    Other Parameters
    ----------------
    n : integer, optional
        Dimension of the Transfer matrix. Default is 6.

    p1 : float, optional
        Probability of hopping once to the left or right. Default is p1 = 0.01.

    p2 : float, optional
        Probability of hopping twice to the left or right. Default is p2 = 0.

    Returns
    -------
    observations : ndarray of shape (n_its, n)
        Array of time-evolved system-states, where observations[i, :] corresponds to the state at time `i`. Default shape is (400, 6)

    Raises
    -------
    AssertionError
        If the initial state `state` is not valid, 
        i.e. has complex entries, does not sum to 1 or has the wrong dimension compared to `n`.

    AssertionError
        If `n_its` is not a positive integer.
    """
    # Check if state is valid, i.e real, of unit length and compatible with `n`.
    _n = kwargs.get("n", 6)
    assert len(
        state) == _n, f"Dimension of the state vector {state} = {len(state)} != n = {_n}."
    assert not any(isinstance(num, complex)
                   for num in state), f"Markovian evolution cannot deal with complex state {state}."
    assert math.isclose(sum(
        state), 1, rel_tol=1e-04), f"The norm of the state vector {state} = {sum(state)} != 1."

    # check if `n_its` is a positive integer
    assert n_its >= 0, "n_its must be greater or equal to 0."
    assert type(n_its) == int, f"n_its must be an integer not {type(_n)}"

    T = Transfer_Matrix(**kwargs)
    state = np.array(state)
    observations = [state]
    for _ in np.arange(n_its):
        state = T @ state
        observations.append(state)
    return np.array(observations)


###########################################################################################################
def Plot_Markov(state=[1, 0, 0, 0, 0, 0], n_its=400, **kwargs):
    """
    Function for widget creation, used for visualising the Markov evolution.

    Parameters
    ----------
    state : array_like, optional
        Initial state of the system. Default is `[1,0,0,0,0,0]`.

    n_its : integer, optional
        Number of Markov iterations, such that `n_its` calculations are performed.
        Default is 400.

    Other Parameters
    ----------------
    n : integer, optional
        Dimension of the Transfer matrix. Default is 6.

    p1 : float, optional
        Probability of hopping once to the left or right. Default is p1 = 0.01.

    p2 : float, optional
        Probability of hopping twice to the left or right. Default is p2 = 0.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure of Markov evolution after `n_its` steps with initial state `state`.
    """
    # Calculate states
    observations = Calc_Markov(state, n_its, **kwargs)
    _n = kwargs.get("n", 6)
    fig = plt.figure(figsize=(10, 6))

    # make plot pretty
    plt.title(
        f"Markov evolution of the $n={_n}$-ring with initial state {state} and $p_1 = {kwargs.get('p1', 0.1)}$", wrap=True)
    plt.xlabel(r"Number of iterations $n_{\mathrm{its}}$")
    plt.ylabel(r"Probability of finding particle at site $i$")
    plt.grid(which="both", axis="both", linestyle="--",
             color="black", alpha=0.4)

    # Ensure color order is consistent with site number
    mpl.rcParams['axes.prop_cycle'] = cycler(
        "color", plt.cm.get_cmap("tab10").reversed().colors[-_n:])
    colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']

    # actual plotting
    for i, site in enumerate(np.arange(_n)[::-1]):
        plt.plot(observations[:, site], ".-",
                 label=f"Site {site+1}", color=colors[i])

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1],
               bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.show()
    return fig


###########################################################################################################
def Calc_Markov_Path(initial_position=1, p1=0.1, n_its=400, **kwargs):
    """
    Calculate the Markovian time evolution of a particle at initial position `initial_position`. Calculated by averaging the final position of `n_paths` particles after `n_its` iterations.

    Parameters
    ----------
    initial_position : integer, optional
        Initial position of all `n_paths` particles in the system.
        Default is 1.

    p1 : float, optional
        Probability of hopping once to the left or right. Default is p1 = 0.01.

    n_its : integer, optional
        Number of iterations, such that `n_its` hoppings are performed.
        Default is 400.

    Other Parameters
    ----------------
    n : integer, optional
        Dimension of the System.
        Default is 6.

    p2 : float, optional
        Probability of hopping twice to the left or right. Default is p2 = 0.

    n_paths : integer, optional
        Number of particles (and thus "paths") to consider when averaging.
        Default is 1000.

    seed : integer, optional
        Seed for the random number generator used for calculating the hopping processes.
        Default is 42.

    Returns
    -------
    p_vec : ndarray of shape (n_its, n)
        Array of probabilties where p_vec[t, j] corresponds to the probability that a particle is on position `i` at time `t`.
        Default shape is (400, 6)
    #TODO: finish documentation
    Raises
    -------
    AssertionError
        If the initial state `state` is not valid, 
        i.e. has complex entries, does not sum to 1 or has the wrong dimension compared to `n`.

    AssertionError
        If `n_its` is not a positive integer.
    """

    # Ensure probabilities are non-negative and bounded
    assert 2 * \
        p1 <= 1, f"For consistency, twice p1 has to be at most 1, you have 2 * p1 = {2 * p1:.3f}."
    assert p1 >= 0, f"Probabilities have to be non-negative, you have p1 = {p1:.3f}."

    Number_of_paths = kwargs.get("n_paths", 1000)
    _n = kwargs.get("n", 6)

    initial_positions_arr = np.ones(Number_of_paths) * (initial_position - 1)

    # Random number Generator `rng` and Probability Vector `pvec`
    rng = np.random.default_rng(kwargs.get("seed", 42))
    p_vec = np.zeros((n_its, _n))

    # Calculate probability of outcome for every timestep `step`
    for step in np.arange(n_its):
        random_vector = rng.random(Number_of_paths)
        initial_positions_arr = (np.where(random_vector <= p1, initial_positions_arr+1,
                                          np.where(random_vector >= (1-p1), initial_positions_arr-1, initial_positions_arr))) % _n
        p_vec[step] = np.bincount(initial_positions_arr.astype(
            int), minlength=_n) / Number_of_paths
    return p_vec


###########################################################################################################
def Plot_Markov_Path(initial_position=1, p1=0.1, n_its=400, **kwargs):
    # Initial_init_arr.options = init_arrs_dict[kwargs.get("n", 6)]
    # TODO: link n with position 1
    observations = Calc_Markov_Path(
        initial_position=initial_position, n_its=n_its, p1=p1, **kwargs)
    n = kwargs.get("n", 6)

    fig = plt.figure(figsize=(12, 6))

    plt.title(
        f"Markov evolution of the $n={n}$-ring, particle initialy at position {initial_position} using explicit paths, $p_1 = {p1}$")
    plt.xlabel(r"Number of iterations $n_{\mathrm{its}}$")
    plt.ylabel(
        r"Probability $P_{n_\mathrm{its}}(i)$ of finding particle at site $i$ ")
    plt.grid(which="both", axis="both", linestyle="--",
             color="black", alpha=0.4)

    mpl.rcParams['axes.prop_cycle'] = cycler(
        "color", plt.cm.get_cmap("tab10").reversed().colors[-n:])

    for site in np.arange(n)[::-1]:
        plt.plot(observations[:, site], ".-", label=f"Site {site+1}")
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1],
               bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    plt.show()
    return fig
