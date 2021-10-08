#Module for QM time evolution
from Module_Markov import Hopping_Matrix
from scipy.linalg import expm # used for matrix exponentiation
import numpy as np
import math

import matplotlib as mpl
import matplotlib.pyplot as plt # Plotting
from cycler import cycler #used for color cycles in mpl

#possible TODOS:
    # normalize wavefunction after user input, such that no errors occur
    #TODO: display U or T next to image to compare
    #TODO let input recognize + between complex numbers, e.g. 1+1j

###########################################################################################################
def Time_Evolution_Operator(n=6, t=0.1):
    '''TODO: write documentation'''
    H = Hopping_Matrix(n)
    #H = Transfer_Matrix(n=n, x=t, x2=t2)
    return expm(-1j * t * H)

###########################################################################################################
def Calc_QM(state=[1,0,0,0,0,0], n_its=400, **kwargs):
    """TODO: add doc string"""

    #Check if state is valid
    #TODO: possibly add optin of automatically normalizing state
    assert math.isclose(np.linalg.norm(state), 1, rel_tol=1e-04), f"The norm of the state vector {state} = {np.linalg.norm(state)} != 1"

    U = Time_Evolution_Operator(**kwargs)

    state = np.array(state)
    observations = [state]
    for _ in np.arange(n_its):
        state = U @ state
        observations.append(state)
    return np.array(observations)

###########################################################################################################
def Plot_QM_Evolution(state=[1,0,0,0,0,0], n_its=400, **kwargs):
    #TODO: write documentation
    observations = Calc_QM(state, n_its, **kwargs)
    n = kwargs.get("n", 6)

    fig = plt.figure(figsize=(10,6))
    plt.title(f"Qm evolution of the $n={n}$-ring with initial state {state} and $p_1 = {kwargs.get('p1', 0.1)}$")
    plt.xlabel(r"Number of iterations $n_{\mathrm{its}}$")
    plt.ylabel(r"Probability of finding particle at site $i$")
    plt.grid()

    mpl.rcParams['axes.prop_cycle'] = cycler("color", plt.cm.get_cmap("tab10").reversed().colors[-n:])

    for site in np.arange(len(state))[::-1]:
        plt.plot(np.abs(observations[:, site])**2, ".-", label=f"Site {site+1}", )
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.show()
    return fig

###########################################################################################################
def Calc_QM_with_Eigenstates(state=[1,0,0,0,0,0], n_its=50, **kwargs):
    """TODO: add doc string"""

    #Check if state is valid
    #TODO: possibly add optin of automatically normalizing state
    assert math.isclose(np.linalg.norm(state), 1, rel_tol=1e-04), f"The norm of the state vector {state} = {np.linalg.norm(state)} != 1"

    #TODO: check numerics of eigenvalues when rounding
    # Note, eigh is necessary, as np.eigdoes not necessarily compute orthogonal eigevectors
    eig_vals, eig_vecs = np.linalg.eigh(Hopping_Matrix(kwargs.get("n", 6)))
    state = np.array(state)
    observations = []

    for its in np.arange(n_its):
        c_n = np.einsum("i, ij -> j", np.conj(state), eig_vecs)
        psi_t = np.einsum("j, kj -> k",  (c_n * np.exp(-1.j * its * kwargs.get("t", 0.1) * eig_vals)), eig_vecs)
        #psi_t = eig_vecs @(c_n * np.exp(-1j * its * kwargs.get("t", 0.1) * eig_vals))
        observations.append(psi_t)
    return np.array(observations)

###########################################################################################################
def Plot_QM_with_Eigenstates(state=[1,0,0,0,0,0], n_its=50, **kwargs):
    #TODO: write documentation
    observations = Calc_QM_with_Eigenstates(state, n_its, **kwargs)
    n = kwargs.get("n", 6)

    fig = plt.figure(figsize=(12,6))
    plt.title(f"QM evolution of the $n={n}$-ring with initial state {state} using eigenbasis of $H$, $p_1 = {kwargs.get('p1', 0.1)}$")
    plt.xlabel(r"Number of iterations $n_{\mathrm{its}}$")
    plt.ylabel(r"Probability of finding particle at site $i$")
    plt.grid()

    mpl.rcParams['axes.prop_cycle'] = cycler("color", plt.cm.get_cmap("tab10").reversed().colors[-n:])

    for site in np.arange(len(state))[::-1]:
        plt.plot(np.abs(observations[:, site])**2, ".-", label=f"Site {site+1}", )
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.show()
    return fig
