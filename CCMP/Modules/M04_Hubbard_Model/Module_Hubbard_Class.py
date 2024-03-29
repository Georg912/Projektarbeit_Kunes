from more_itertools import distinct_permutations

import scipy
import scipy.spatial.distance as sp
import scipy.linalg as sp_la
import numpy as np
import ipywidgets as widgets
import time
import matplotlib.pyplot as plt  # Plotting
import matplotlib as mpl
from cycler import cycler  # used for color cycles in mpl

from .Module_Widgets import n_Slider, s_up_Slider, s_down_Slider
from .Module_Widgets import u_Slider, t_Slider, u_range_Slider, t_range_Slider
from .Module_Widgets import basis_index_Slider, t_ij_inputfile_checkbox
from .Module_Widgets import u1_Slider, u1_checkbox
from .Module_Cache_Decorator import Cach  # used for caching functions

import numba as nb
from numba import jit, njit
import scipy.sparse.linalg as splin
from fractions import Fraction
from textwrap import fill
from matplotlib.ticker import FormatStrFormatter
from pathlib import Path
import inspect
from enum import IntEnum


@jit(nopython=True)
def is_single_hopping_process(x, y, cre, anh) -> bool:
    """
    returns `True` if hopping from state `x` at position `cre` to state `y` at position `anh` is allowed, i.e. if it is a single hopping process which transforms state x into state y.

    Keep in mind that this does **not** check, if the spin is conserved, i.e. also hopping from a spin up side (0 <= cre < n) to a spin down site (n <= anh < 2n) is allowed.

    Parameters
    ----------
    x, y : ndarray
                    initial and final state of hopping
    cre, anh : int
                    index to which/from where to hop

    Returns
    -------

    allowed : bool
                    `True` if hopping is allowed
    """
    return (x[cre] * y[anh]) == 1 and (x[anh] + y[cre]) == 0


@jit(nopython=True)
def hop_sign(state, i) -> float:
    """
    Calculates the sign of creation/annihilation of a particle at site `i`.
    Due to cleverly choosing the order of the creation operators the sign of a hopping is just total number of particles right to that state. We choose the order such that all spin down operators are left of all the spin up operators and the index of the operators are decreasing, e.g. c3_down c2_down, c3_up c1_up. Further all states are numbered from the lowest spin down to the largest spin up state.

    Parameters
    ----------
    state : ndarray (1, 2n)
                    One state of the system.
    i : int
                    the index at which the creation/annihilation operator is applied.

    Returns
    -------
    sign : int
                    sign of the hopping
    """
    return (-1)**np.sum(state[i+1:])


class Hubbard:

    def __init__(self, n=None, s_up=None, s_down=None) -> None:

        self.out = widgets.Output()

        # Set all site and spin sliders
        ############################################
        self.n = n_Slider
        # set default value but allow creation of class with custom n
        # in hindsight, necessary for DynamicalHubbardPropagator class
        if n is None:
            self.n.value = 6
        self.n.observe(self.on_change_n, names="value")

        self.s_up = s_up_Slider
        if s_up is None:
            self.s_up.value = 3
        self.s_up.observe(self.on_change_s_up, names="value")

        self.s_down = s_down_Slider
        if s_down is None:
            self.s_down.value = 3
        self.s_down.observe(self.on_change_s_down, names="value")

        # Calculate basis states and set index sliders
        ##############################################
        self.basis = self.Construct_Basis()
        self.basis_index = basis_index_Slider
        self.basis_index.max = self.basis.shape[0] - 1

        self.hoppings = self.Allowed_Hoppings_H()

        self.Reset()

        # Set all u and t Sliders
        ##############################################
        self.u = u_Slider
        self.u_range = u_range_Slider
        du = (self.u_range.max - self.u_range.min) / self.u_range.step
        self.u_array = np.linspace(
            self.u_range.min, self.u_range.max, num=int(du) + 1, endpoint=True)

        self.t = t_Slider
        self.t_range = t_range_Slider
        dt = (self.t_range.max - self.t_range.min) / self.t_range.step
        self.t_array = np.linspace(
            self.t_range.min, self.t_range.max, num=1 + int(dt), endpoint=True)

        self.u1 = u1_Slider
        self.u1.layout.visibility = "hidden"
        self.u1.observe(self.on_change_u1, names="value")
        self.u1_prev = self.u1.value

        # Set all checkboxes
        self.t_ij = t_ij_inputfile_checkbox
        self.t_ij.value = False
        self.t_ij.observe(self.on_change_t_ij, names="value")

        self.u1_checkbox = u1_checkbox
        self.u1_checkbox.observe(self.on_change_u1_checkbox, names="value")

    def Construct_Basis(self):
        """
        Return all possible m := nCr(n, s_up) * nCr(n, s_down) basis states for specific values of `n`, `s_up` and `s_down` by permuting one specific up an down state.

        Returns
        -------
        basis : ndarray (m, 2*n)
                                        array of basis states
        """
        _s_up = self.s_up.value
        _s_down = self.s_down.value
        _n = self.n.value

        up_state = np.concatenate((np.ones(_s_up), np.zeros(_n - _s_up)))
        down_state = np.concatenate(
            (np.ones(_s_down), np.zeros(_n - _s_down)))

        all_up_states = np.array(tuple(distinct_permutations(up_state)))
        all_down_states = np.array(
            tuple(distinct_permutations(down_state)))

        # reshape and repeat or tile to get all possible combinations:
        up_repeated = np.repeat(
            all_up_states, all_down_states.shape[0], axis=0)
        down_repeated = np.tile(
            all_down_states, (all_up_states.shape[0], 1))

        # Combine up and down states
        return np.concatenate((up_repeated, down_repeated), axis=1)

    def show_basis(self, index, **kwargs) -> None:
        """
        Method to print basis vector at position `index` in occupation number basis. Also prints the total number of basis vectors, which is analytically given by nCr(n, s_up) * nCr(n, s_down)

        Parameters
        ----------
        index : int
                                        Index of the basis vector to display

        Returns
        -------
        None

        Other Parameters
        ----------------
        **Kwargs : Widgets
                                        used to add sliders and other widgets to the displayed output
        """
        _n = self.n.value
        _s_up = self.s_up.value
        _s_down = self.s_down.value

        # check analtical formula equals numeric calculation
        assert self.basis.shape[0] == scipy.special.comb(_n, _s_up) * scipy.special.comb(_n, _s_down), \
            "Error: Basis vector count does not match the analytical formula"

        print(f"Total number of basis states = {self.basis.shape[0]}")
        print(f"Basis state {index} = {self.basis[index]}")

    def on_change_n(self, change) -> None:
        """
        Method to update basis states and hoppings when `n` is changed

        Parameters
        ----------
        change : dict
                        dictionary containing information about the change
        """
        self.basis = self.Construct_Basis()
        self.basis_index.max = self.basis.shape[0] - 1
        self.hoppings = self.Allowed_Hoppings_H()
        self.Reset()

    def on_change_s_up(self, change) -> None:
        """
        Method to update basis states and hoppings when `s_up` is changed

        Parameters
        ----------
        change : dict
                        dictionary containing information about the change
        """
        if (self.s_down.value > self.n.value or self.s_up.value > self.n.value):
            pass
        else:
            self.basis = self.Construct_Basis()
            self.basis_index.max = self.basis.shape[0] - 1
            self.Reset()

    def on_change_s_down(self, change) -> None:
        """
        Method to update basis states and hoppings when `s_down` is changed

        Parameters
        ----------
        change : dict
                dictionary containing information about the change
        """
        if (self.s_down.value > self.n.value or self.s_up.value > self.n.value):
            pass
        else:
            self.basis = self.Construct_Basis()
            self.basis_index.max = self.basis.shape[0] - 1
            self.Reset()

    def on_change_t_ij(self, change) -> None:
        """
        Method to update cached properties when `t_ij` is changed

        Parameters
        ----------
        change: dict
                dictionary containing information about the change
        """
        self.Reset()

    def on_change_u1(self, change) -> None:
        """
        Method to update cached properties when `u1` is changed

        Parameters
        ----------
        change: dict
                dictionary containing information about the change
        """
        self.Reset()

    def on_change_u1_checkbox(self, change) -> None:
        """
        Method to display or hide `u1` slider if `u1_checkbox` is changed

        Parameters
        ----------
        change: dict
                dictionary containing information about the change
        """
        if self.u1_checkbox.value == True:
            self.u1.layout.visibility = "visible"
            self.u1.value = self.u1_prev
        else:
            self.u1.layout.visibility = "hidden"
            self.u1_prev = self.u1.value
            self.u1.value = 0

    @staticmethod
    def methods_with_decorator(cls, decoratorName="Cach"):
        """
        Returns a list of all methods of a class and its parent classes that are decorated with a specific decorator <decoratorName> .

        Parameters
        ----------
        cls: class
                Class (and parent classes) to search for methods.
        decoratorName: str
                Name of the decorator to search for .

        Returns
        -------
        methods: list[str]
                List of methods that are decorated with the decorator.
        """
        methods = []
        cls_list = inspect.getmro(cls)
        for c in cls_list:
            if c in (type, object):
                continue
            sourcelines = inspect.getsourcelines(c)[0]

            # find `@decoratorName` in sourcelines => next line is target property
            for i, line in enumerate(sourcelines):
                if line.strip() == '@'+decoratorName:
                    # If method additionally uses `@jit` use the next but one line
                    if sourcelines[i+1].split('(')[0].strip() == '@jit':
                        idx = i + 2
                    else:
                        idx = i + 1

                    methods.append(sourcelines[idx].split(
                        'def')[1].split('(')[0].strip())
        return methods

    @Cach
    def up(self):
        """
        Return all possible spin up states.

        Returns
        -------
        basis: ndarray(m, n)
                                        array of spin up basis states
        """
        return self.basis[:, : self.n.value]

    @Cach
    def down(self):
        """
        Return all possible spin down states.

        Returns
        -------
        basis: ndarray(m, n)
                                                                                                                                        array of spin down basis states
        """
        return self.basis[:, self.n.value:]

    @Cach
    def Op_nn(self):
        """
        Return the double occupation number operator `nn`, which is diagonal in the occupation number basis

        Returns
        -------
        nn: ndarray(m, m)
        """
        return np.diag(np.sum(self.up * self.down, axis=1))

    @Cach
    def Op_nn_nearest_neighbour(self):
        """
        Return the sum of all double occupation number operators nearest neighbour sites = sum[(n_i_up + n_i_down) * (n_i+1_up + n_i+1_down)], which is diagonal in the occupation number basis

        Returns
        -------
        nn: ndarray(m, m)
        """
        _n = self.n.value

        return np.sum([(self.Op_n_up(i) + self.Op_n_down(i)) @ (self.Op_n_up((i+1) % _n) + self.Op_n_down((i+1) % _n)) for i in np.arange(_n)], axis=0)

    @jit(forceobj=True)  # otherwise error message
    def Allowed_Hoppings_H(self):
        """
        Calculates all allowed hoppings in the Hamiltonian H from position `r1` to position `r2` for a given `n`.

        Returns
        -------
        hoppings: ndarray(4n, 2)
                                        Array of allowed hopping pairs
        """
        _n = self.n.value
        r1 = np.arange(0, _n)
        r2 = np.arange(1, _n+1) % _n

        up_clockwise = np.stack((r1, r2)).T
        up_counter_clockwise = np.fliplr(up_clockwise)
        down_clockwise = up_clockwise + _n
        down_counter_clockwise = up_counter_clockwise + _n

        if (_n == 2):  # clockwise and counterclockwise are the same
            return np.vstack((up_clockwise, down_clockwise))
        else:
            return np.vstack(
                (up_clockwise, up_counter_clockwise, down_clockwise, down_counter_clockwise))

    @jit(forceobj=True)  # otherwise error message
    def Allowed_Hoppings_Sx(self):
        """
        Calculates all allowed hoppings in the Spin operator S_x from position `r1` to position `r2` for a given `n`.

        Returns
        -------
        hoppings: ndarray(4n, 2)
                                        Array of allowed hopping pairs
        """
        _n = self.n.value
        r1 = np.arange(0, _n)
        r2 = r1 + _n

        clockwise = np.stack((r1, r2)).T
        counter_clockwise = np.fliplr(clockwise)

        return np.vstack((clockwise, counter_clockwise))

    @staticmethod
    @njit(fastmath=True, parallel=True)
    def Sign_Matrix(A, B, ii, jj):
        """
        Calculates the sign matrix for a hopping from site ii to site jj from all possible states in A to all possible states in B.

        Parameters
        ----------
        A: ndarray(m, 2n)
            Basis to hop into
        B: ndarray(m, 2n)
            Basis to hop out of
        ii: int
            index of site to hop into
        jj: int
            index of site to hop out of

        Returns
        -------
        C: ndarray(m, m)
            Sign matrix of the hopping
        """
        assert A.shape[1] == B.shape[1]
        C = np.zeros((A.shape[0], B.shape[0]), A.dtype)

        for i in nb.prange(A.shape[0]):
            for j in range(B.shape[0]):
                if is_single_hopping_process(A[i, :], B[j, :], ii, jj):
                    C[i, j] = hop_sign(A[i, :], ii) * hop_sign(B[j, :], jj)
        return C

    @Cach
    @jit(forceobj=True)  # , cache=True)
    def Ht(self):
        """
        Calculates Hopping matrix Ht from the allowed hoppings in the Hamiltonian H. First all allowed nearest neighbor hoppings `NN_hoppings` are calculated, then the total sign matrix `NN_sign` is calculated for each hopping. The resulting Hopping Hamiltonian `Ht` is then the product of `NN_sign` and `NN_hoppings`.
        """
        _base = self.basis
        NN_hoppings = sp.cdist(_base, _base, metric="cityblock")
        NN_hoppings = np.where(NN_hoppings == 2, 1, 0)

        NN_sign = np.sum(np.array([self.Sign_Matrix(_base, _base, i, j)
                                   for i, j in self.hoppings]), axis=0)

        NN_sign = np.where(NN_sign >= 1, 1, np.where(NN_sign <= -1, -1, 0))
        return NN_hoppings * NN_sign

    @Cach
    @jit(forceobj=True)  # , cache=True)
    def Ht_ij(self):
        """
        Calculates Hopping matrix Ht from the allowed hoppings in the Hamiltonian H. First all allowed hoppings `NN_hoppings` are calculated, then the total sign matrix `NN_sign` is calculated for each hopping. The resulting Hopping Hamiltonian `Ht` is then the product of `NN_sign` and `NN_hoppings`.

        Note that this method uses an input file to specify the the hopping amplitudes, located in the 't_ij' folder.
        """
        _base = self.basis
        _n = self.n.value
        NN_hoppings = sp.cdist(_base, _base, metric="cityblock")
        NN_hoppings = np.where(NN_hoppings == 2, 1, 0)

        path = Path(__file__).parent / "t_ij"
        _t_ij = np.loadtxt(path / f"n{_n}.txt", delimiter=",")

        NN_sign = np.sum(np.array([_t_ij[i, j] * self.Sign_Matrix(_base, _base, i, j)
                                   for i in np.arange(_n) for j in np.arange(_n)]), axis=0)
        NN_sign += np.sum(np.array([_t_ij[i - _n, j - _n] * self.Sign_Matrix(_base, _base, i, j)
                                    for i in np.arange(_n, 2*_n) for j in np.arange(_n, 2*_n)]), axis=0)

        # NN_sign = np.where(NN_sign > 0., 1, np.where(NN_sign < -0., -1, 0))

        return NN_hoppings * NN_sign

    @Cach
    def Hu(self):
        """
        Return cached on-site interaction Hamiltonian H_u given exactly by the occupation number operator `nn`.

        Returns
        -------
        Hu: ndarray(m, m)
                                        on-site Interaction Hamiltonian
        """
        return self.Op_nn

    @Cach
    def Hu1(self):
        """
        Return cached nearest-neighbour-site interaction Hamiltonian H_u1 given exactly by the occupation number operator `nn_nearest_neighbour`.

        Returns
        -------
        Hu: ndarray(m, m)
                                        on-site Interaction Hamiltonian
        """
        return self.Op_nn_nearest_neighbour

    def H(self, u, t, u1):
        """
        Compute the system's total Hamiltonian H = u*Hu + t*Ht + u1 * Hu1 for given prefactors `u`, `t` and possibly `u1`.

        Parameters
        ----------
        u: float
                                        on-site interaction strength
        t: float
                                        hopping strength
        u1: float
                                        nearest-neighbour-site interaction strength
        """
        if self.t_ij.value == True:
            return t * self.Ht_ij + u * self.Hu + u1 * self.Hu1
        return t * self.Ht + u * self.Hu + u1 * self.Hu1

    def Show_H(self, u, t, u1, **kwargs):
        """
        Method to print total Hamiltonian H = u*Hu + t*Ht and the dimension of H

        Parameters
        ----------
        u: float
                                        on-site interaction strength due to electric field
        t: float
                                        global hopping amplitude from site i to site j

        Returns
        -------
        None

        Other Parameters
        ----------------
        **Kwargs: Widgets
                                        used to add sliders and other widgets to the displayed output
        """
        dimension = self.basis_index.max + 1
        print(f"Dimension of H = {dimension} x {dimension}")
        print(f"H = \n{self.H(u, t, u1)}")

    def Reset(self):
        """
        Method to reset all cached properties(if they were already cached).
        """
        for prop in self.methods_with_decorator(self.__class__, "Cach"):
            self.__dict__.pop(prop, None)

    @Cach
    @jit(forceobj=True, cache=True)
    def Eigvals_Hu(self):
        """
        Calculates the eigenvalues of the on-site interaction Hamiltonian Hu for hopping amplitude t = 1.

        Returns
        -------
        vals: ndarray(len(u), m)
                                                                                                                                        Eigenvalues of the on-site interaction Hamiltonian
        """
        _u1 = self.u1.value
        vals = [np.linalg.eigvalsh(self.H(u, 1, _u1)) for u in self.u_array]
        return np.array(vals)

    def Plot_Eigvals_Hu(self, **kwargs):
        """
        Method to plot the eigenvalues of the on-site interaction Hamiltonian H_u for constant hopping amplitude t = 1. Method als plots the number of eigenvalues for the atomic limit as labels for the correspondingly colored curves.

        Returns
        -------
        fig: matplotlib.figure.Figure
                                                                                                                                        figure object to save as image-file
        Other Parameters
        ----------------
        **Kwargs: Widgets
                                        used to add sliders and other widgets to the displayed output
        """
        _u1 = self.u1.value
        u_idx, u = self.find_indices_of_slider(self.u_array, self.u_range)

        last_eigvals_u = self.Eigvals_Hu[-1, :]
        sorted_eigval_idx = np.argsort(last_eigvals_u)
        eig_u = self.Eigvals_Hu[u_idx][:, sorted_eigval_idx]

        if self.u1_checkbox.value == True:
            uniq = np.unique(np.diag(u[-1] * self.Op_nn + self.u1.value *
                                     self.Op_nn_nearest_neighbour).astype(float).round(2), return_counts=True)
        elif self.u1_checkbox.value == False:
            uniq = np.unique(np.diag(self.Op_nn).astype(int),
                             return_counts=True)
        color = mpl.cm.tab10(np.repeat(np.arange(uniq[0].size), uniq[1]) % 10)
        mpl.pyplot.rcParams["axes.prop_cycle"] = cycler("color", color)

        u1_str = f" and $U_1 = {_u1}$" if self.u1_checkbox.value == True else ""

        fig = plt.figure(figsize=(10, 6))
        _title = fill(
            f"Eigenvalues of Hubbard-Ring Hamiltonian $H$ as a function of the on-site interaction strength $U$ for $n={self.n.value}$ sites with {self.s_up.value} spin up electron(s), {self.s_down.value} spin down electron(s), hopping amplitude $t = 1$" + u1_str, width=80)
        plt.title(rf"{_title}")
        plt.xlabel(r"$U$")
        plt.ylabel(r"Eigenvalue(s)")
        plt.grid(which="both", axis="both",
                 linestyle="--", color="black", alpha=0.4)
        axes = plt.plot(u, eig_u, ".-")

        for idx, num in enumerate(np.cumsum(uniq[1])):
            axes[num-1].set_label(f"{uniq[1][idx]}")

        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left",
                   ncol=1, title="# of eigenvalues")
        # plt.show()
        return fig

    @Cach
    @jit(forceobj=True, cache=True)
    def Eigvals_Ht(self):
        """
        Calculates the eigenvalues of the hopping Hamiltonian Ht for on-site interaction U = 10.

        Returns
        -------
        vals: ndarray(len(u), m)
                Eigenvalues of the hopping Hamiltonian
        """
        _u1 = self.u1.value
        vals = [np.linalg.eigvalsh(self.H(10, t, _u1)) for t in self.t_array]
        return np.array(vals)

    def Plot_Eigvals_Ht(self, **kwargs):
        """
        Method to plot the eigenvalues of the hopping Hamiltonian H_t for constant on-site interaction u = 10. Method als plots the number of eigenvalues for the atomic limit as labels for the correspondingly colored curves.

        Returns
        -------
        fig: matplotlib.figure.Figure
                                                                                                                                        figure object to save as image-file
        Other Parameters
        ----------------
        **Kwargs: Widgets
                                        used to add sliders and other widgets to the displayed output
        """
        _u1 = self.u1.value
        t_idx, t = self.find_indices_of_slider(self.t_array, self.t_range)

        last_eigvals_t = self.Eigvals_Ht[-1, :]
        sorted_eigval_idx = np.argsort(last_eigvals_t)
        eig_t = self.Eigvals_Ht[t_idx][:, sorted_eigval_idx]

        if self.u1_checkbox.value == True:
            uniq = np.unique(np.diag(10 * self.Op_nn + _u1 *
                                     self.Op_nn_nearest_neighbour).astype(float).round(2), return_counts=True)
        elif self.u1_checkbox.value == False:
            uniq = np.unique(np.diag(self.Op_nn).astype(int),
                             return_counts=True)

        color = mpl.cm.tab10(np.repeat(np.arange(uniq[0].size), uniq[1]) % 10)
        mpl.pyplot.rcParams["axes.prop_cycle"] = cycler("color", color)

        u1_str = f" and $U_1 = {_u1}$" if self.u1_checkbox.value == True else ""

        fig = plt.figure(figsize=(10, 6))
        _title = fill(
            f"Eigenvalues of Hubbard-Ring Hamiltonian $H$ as a function of the hopping amplitude $t$ for $n={self.n.value}$ sites \n with {self.s_up.value} spin up electron(s), {self.s_down.value} spin down electron(s), on-site interaction $U=10$" + u1_str, width=80)
        plt.title(rf"{_title}")
        plt.xlabel(r"$t$")
        plt.ylabel(r"Eigenvalue(s)")
        plt.grid(which="both", axis="both", linestyle="--",
                 color="black", alpha=0.4)
        axes = plt.plot(t, eig_t, ".-")

        for idx, num in enumerate(np.cumsum(uniq[1])):
            axes[num-1].set_label(f"{uniq[1][idx]}")

        plt.gca().invert_xaxis()
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left",
                   ncol=1, title="# of eigenvalues")
        # plt.show()
        return fig

    def Plot_Eigvals_H(self, **kwargs):
        """
        Method to plot the eigenvalues of the Hamiltonian H(u, t) . Method als plots the number of eigenvalues for the atomic limit as labels for the correspondingly colored curves.

        Returns
        -------
        fig: matplotlib.figure.Figure
                                                                                                                                        figure object to save as image-file
        Other Parameters
        ----------------
        **Kwargs: Widgets
                                        used to add sliders and other widgets to the displayed output
        """
        t_idx, t = self.find_indices_of_slider(self.t_array, self.t_range)
        u_idx, u = self.find_indices_of_slider(self.u_array, self.u_range)
        u_max = self.u_range.value[1]
        t_max = self.t_range.value[1]
        _u1 = self.u1.value

        eig_t = np.array([np.linalg.eigvalsh(self.H(u_max, t, _u1))
                          for t in self.t_array])
        eig_u = np.array([np.linalg.eigvalsh(self.H(u, t_max, _u1))
                          for u in self.u_array])

        last_eigvals_t = eig_t[-1, :]
        sorted_eigval_idx = np.argsort(last_eigvals_t)
        eig_t = eig_t[t_idx][:, sorted_eigval_idx]

        last_eigvals_u = eig_u[-1, :]
        sorted_eigval_idx = np.argsort(last_eigvals_u)
        eig_u = eig_u[u_idx][:, sorted_eigval_idx]

        if self.u1_checkbox.value == True:
            uniq = np.unique(np.diag(u_max * self.Op_nn + self.u1.value *
                                     self.Op_nn_nearest_neighbour).astype(float), return_counts=True)
        elif self.u1_checkbox.value == False:
            uniq = np.unique(np.diag(self.Op_nn).astype(
                float), return_counts=True)
        color = mpl.cm.tab10(np.repeat(np.arange(uniq[0].size), uniq[1]) % 10)
        mpl.pyplot.rcParams["axes.prop_cycle"] = cycler("color", color)

        fig, axes = plt.subplots(ncols=2, sharey=True,
                                 figsize=(15, 8))
        fig.subplots_adjust(wspace=0.02)

        axes[0].set_xlabel(r"$U$")
        axes[0].set_ylabel(r"Eigenvalue(s)")
        axes[1].set_xlabel(r"$t$")
        axes[0].grid(which="both", axis="both", linestyle="--",
                     color="black", alpha=0.4)
        axes[1].grid(which="both", axis="both", linestyle="--",
                     color="black", alpha=0.4)

        axes[0].plot(u, eig_u, ".-")
        ax = axes[1].plot(t, eig_t, ".-")

        for idx, num in enumerate(np.cumsum(uniq[1])):
            ax[num-1].set_label(f"{uniq[1][idx]}")

        u1_str = f" and $U_1 = {_u1}$" if self.u1_checkbox.value == True else ""
        _title = fill(
            f"Eigenvalues of Hubbard-Ring Hamiltonian $H$ for $n={self.n.value}$ sites with {self.s_up.value} spin up electron(s) and {self.s_down.value} spin down electron(s). On the left as a function of $U$ for constant $t={t_max}$ and on the right as a function of $t$ for constant $U={u_max}$" + u1_str, width=100)
        fig.suptitle(fr"{_title}")

        fig.gca().invert_xaxis()
        fig.legend(bbox_to_anchor=(0.9, 0.9), loc="upper left",
                   ncol=1, title="# of eigenvalues")
        # fig.show()
        return fig

    def is_degenerate(self):
        """
         Method to check if the ground state of the Hamiltonian is degenerate.
         Checks the degeneracy for H(U=5, t=1) because U = 0 is a special case

        Returns
        -------
        bool
                                        True if the Hamiltonian is degenerate, else False
        """
        _H = self.H(5, 1, 0)

        if _H.shape[0] == 1:
            return False
        else:
            vals = sp_la.eigvalsh(_H, subset_by_index=[0, 1])
            _, count = np.unique(vals.round(5), return_counts=True)

            return True if count[0] == 2 else False

    def k_list(self):
        """
        Method to return the list of possible k-values to use for translation symmetry(Fourier transform), given by {2*pi*i*j / n | j in {0, ..., n-1}} .

        Returns
        -------
        k_list: nd_array(n,)
                                                                                                                                        array of k-values
        """
        _n = self.n.value
        return np.exp(2 * np.pi * 1j / _n * np.arange(_n))

    @Cach
    def GS(self):
        """
        Calculate the (normalized) ground state eigevector ( or eigenvectors if degenerate) of the Hamiltonian H(u, t) for u in [u_min, u_max] and t = 1. Methods uses sparse matrices to speed up the computation of the eigenvector associtated with the smallest (algebraic) eigenvalue of H(u, t). k is the degeneracy of the ground state, i.e. k = 1 for a non-degenerate ground state.

        Returns
        -------
        eig_vec: ndarray(len(u), m, k)
        """
        _u1 = self.u1.value
        # sparse method does not work for matrixes with dimension 1x1
        if self.basis.shape[0] == 1:
            eig_vecs = np.ones((self.u_array.size, 1, 1))
        # deal with degenerate ground state
        elif self.is_degenerate():
            _H = [scipy.sparse.csr_matrix(self.H(u, 2, _u1))
                  for u in self.u_array]
            eig_vecs = np.array([splin.eigsh(h, k=2, which="SA")[1]
                                 for h in _H]) / np.sqrt(2)
        # deal with non-degenerate ground state
        else:
            _H = [scipy.sparse.csr_matrix(self.H(u, 2, _u1))
                  for u in self.u_array]
            eig_vecs = np.array([splin.eigsh(h, k=1, which="SA")[1]
                                 for h in _H])

        return eig_vecs

    def diag(self, i):
        return self.basis[:, i] * self.basis[:, self.n.value + i]

    def Double(self, i):
        return np.diag(self.diag(i))

    @Cach
    def Op_nn_mean(self):
        """
        Return the average double occuption number operator `nn / n` which is diagonal occupation number basis

        Return
        ------
        nn_mean: ndarray(m, m)
        """
        return self.Op_nn / self.n.value

    def Exp_Val_0(self, Op):
        """
        Calculate the ground state expectation value < GS | Op | GS > of the operator Op for u in [u_min, u_max].

        Parameters
        ----------
        Op: ndarray(m, m)
                                        Matrix representation ( in occupation number basis) of the operator Op.

        Returns
        -------
        exp_val: ndaray(len(u),)
                                        Expectation value of the operator Op for u in [u_min, u_max]
        """
        # Calculates (vectorized) vector-wise matrix vector sandwich EV_i = vec_i.T * Op * vec_i
        # np.einsum("ij, ji->i", self.GS, Op @ self.GS.T)
        a = np.einsum("ijk, kji -> i", self.GS, Op @ self.GS.T)

        # k = min(10, self.basis.shape[0] - 1)
        # if k > 0:
        #     vals, vecs = splin.eigsh(
        #         scipy.sparse.csr_matrix(self.H(0, 1)), k=k, which="SA")
        #     vals, counts = np.unique(vals.round(5), return_counts=True)
        #     vecs1 = vecs[np.newaxis, :, :counts[0]] / np.sqrt(counts[0])
        #     b = np.einsum("ijk, kji -> i", vecs1, Op @ vecs1.T)[0]
        # else:
        #     b = Op.item(0, 0)
        a[0] = a[1] - (a[2] - a[1])
        return a

    def Op_n_up(self, i):
        """
        Calculate the spin-up occupation number operator `n_up` for site i.

        Parameters
        ----------
        i: int
                                        Site index

        Returns
        -------
        n_up: ndarray(m, m)
        """
        return np.diag(self.up[:, i])

    def Op_n_down(self, i):
        """
        Calculate the spin-down occupation number operator `n_down` for site i.

        Parameters
        ----------
        i: int
                                        Site index

        Returns
        -------
        n_down: ndarray(m, m)
        """
        return np.diag(self.down[:, i])

    def Op_Sz(self, i):
        """
        Calculate the spin operator in z-direction `Sz` for site i.

        Returns
        -------
        Sz: ndarray(m, m)
        """
        return np.diag((self.up - self.down)[:, i])

    def Op_SzSz(self, i, j):
        """
        Calculate the spin-spin-correlation operator in z-direction `SzSz(i, j)` for sites i and j.

        Returns
        -------
        SzSz: ndarray(m, m)
        """
        return self.Op_Sz(i) @ self.Op_Sz(j)

    def Op_Szk(self, k):
        """
        Calculate the reciprocal spin-operator in z-direction `Szk`

        Parameters
        ----------
        k: int
            k-value at which to evaluate the operator

        Returns
        -------
        Szk: ndarray(m, m)
        """
        l = self.k_list()**k
        _n = self.n.value
        return np.sum([self.Op_Sz(i) * l[i] for i in np.arange(_n)], axis=0)

    @Cach
    def ExpVal_nn_mean(self):
        """
        Return the cached ground state expectation value of the average double occuption number operator `nn_mean` (for performance reasons).

        Return
        ------
        expval_nn_mean: ndarray(m,)
        """
        return self.Exp_Val_0(self.Op_nn_mean)

    @Cach
    def ExpVal_Sz(self):
        """
        Return the cached ground state expectation value of the spin operator `Sz` (for performance reasons).

        Return
        ------
        expval_Sz: ndarray(m,)
        """
        return self.Exp_Val_0(self.Op_Sz(0))

    @Cach
    def ExpVal_SzSz_ii(self):
        """
        Return the cached ground state expectation value of the spin-spin correlation operator `SzSz` (for performance reasons).

        Return
        ------
        expval_SzSz: ndarray(m,)
        """
        return self.Exp_Val_0(self.Op_SzSz(0, 0))

    @Cach
    def ExpVal_SzSz_ij(self):
        """
        Return the cached ground state expectation value of the spin-spin correlation operator `Sz_i Sz_j` for all relevant correlation sites(for performance reasons).

        Return
        ------
        expval_SzSz: list[ndarray(m,)]
        """
        return [self.Exp_Val_0(self.Op_SzSz(0, i)) for i in np.arange(self.n.value // 2 + 1)]

    @Cach
    def ExpVal_SzkSzk(self):
        """
        Return the cached ground state expectation value of the reciprocal-space spin-spin correlation operator `Sz_k Sz_k'` for all relevant correlation sites (i.e. those that are non-zero, in other words, were k' = -k).

        Return
        - -----
        expval_SzkSzk: list[ndarray(m,)]
        """
        _n = self.n.value
        # number_of_unique_correlations
        _l = np.floor(_n / 2) + 1

        return [np.real(self.Exp_Val_0(self.Op_Szk(i) @ self.Op_Szk((_n-i) % _n))) for i in np.arange(_l)]

    @Cach
    def Chi(self):
        """
        Return the average local susceptibility `chi`=1/n * Sum_{i=1} ^ {n} Sum_{m > g} | <psi_m | S_iz | psi_g > | ^ 2 / (E_m - E_g) of the system.

        Returns
        - ------
        Chi: ndarray(len(u),)
        """

        _n = self.n.value
        _chi = 0.

        for i in np.arange(_n):
            _chi += self.Calc_Coupling(self.Op_Sz(i))
        return _chi / _n

    @Cach
    def Chi_staggered(self):
        """
        Return the staggered susceptibility `chi_staggered`=Sum_i Sum_{n > g} | <psi_n | (-1) ^ i S_iz | psi_g > | ^ 2 / (E_n - E_g) of the system.

        Returns
        - ------
        Chi_staggered: ndarray(len(u),)
        """
        Sz_staggered = np.sum([(-1)**i * self.Op_Sz(i)
                               for i in np.arange(self.n.value)], axis=0)

        return self.Calc_Coupling(Sz_staggered)

    @ staticmethod
    def find_indices_of_slider(array, range_slider):
        """
        Find indices of `array` that are within the range of the `range_slider`.

        Parameters
        - ---------
        array: ndarray
            array of values to be filtered
        range_slider: Slider
            Slider object that contains the range of values to use as bounds

        Returns
        - ------
        s_idx: ndarray
            indices of `array` that are within the range of the `range_slider`
        s_arr: ndarray
            array of values that are within the range of the `range_slider`
        """
        s_min = range_slider.value[0]
        s_max = range_slider.value[1]

        s_idx = np.nonzero(np.logical_and(array <= s_max, array >= s_min))
        s_arr = array[s_idx]
        return s_idx, s_arr

    def Plot_ExpVal_nn(self, **kwargs):
        """
        Method to plot the expectation value of the operator `nn_mean` for u in [u_min, u_max] and t=1.

        Returns
        - ------
        fig: matplotlib.figure.Figure
                                        figure object to save as image-file

        Other Parameters
        - ---------------
        **Kwargs: Widgets
                                        used to add sliders and other widgets to the displayed output
        """
        _s_up = self.s_up.value
        _s_down = self.s_down.value
        _n = self.n.value
        _u1 = self.u1.value

        u_idx, u = self.find_indices_of_slider(self.u_array, self.u_range)
        nn_max = _s_up * _s_down / _n**2
        nn_min = max(0, (_s_up + _s_down - _n) / _n)
        nn = np.round(self.ExpVal_nn_mean, 5)
        if self.u1_checkbox.value == False:
            nn[0] = np.round(nn_max, 5)
        nn = nn[u_idx]
        nn_str = r"$\left\langle n_i^\mathrm{up} n_i^\mathrm{down} \right\rangle $"

        color = mpl.cm.tab10(np.arange(0, 10))
        mpl.pyplot.rcParams["axes.prop_cycle"] = cycler("color", color)
        fig = plt.figure(figsize=(10, 6))

        u1_str = f" and $U_1 = {_u1}$" if self.u1_checkbox.value == True else ""
        _title = fill(
            r"Average double occupation $\langle$$n_i^\mathrm{up}$$n_i^\mathrm{down}$$\rangle$ " f"as a function of the on-site interaction $U$ for $n = {_n}$ sites with {_s_up} spin up electron(s), {_s_down} spin down electron(s), hopping amplitude $t = 1$" + u1_str, width=80)
        plt.title(rf"{_title}")
        plt.xlabel(r"$U$")
        plt.ylabel(nn_str)
        plt.grid(which="both", axis="both", linestyle="--",
                 color="black", alpha=0.4)

        plt.plot(u, nn.round(4), ".-", label=nn_str)
        plt.plot(u, (nn_max * np.ones(u.shape)).round(4), "--",
                 label=str(Fraction(nn_max).limit_denominator(100)))
        plt.plot(u, (nn_min * np.ones(u.shape)).round(4), "--",
                 label=str(Fraction(nn_min).limit_denominator(100)))

        # otherwise n=3, s_up=3, s_down=2 or vice versa has weird format
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=1)
        return fig

    def Plot_ExpVal_Sz(self, **kwargs):
        """
        Method to plot the expectation value of the spin operator `Sz_i`, `Sz_i ^ 2` and `dSz_i ^ 2` for u in [u_min, u_max] and t=1.

        Returns
        - ------

        fig: matplotlib.figure.Figure
                                        figure object to save as image-file

        Other Parameters
        - ---------------
        **Kwargs: Widgets
                                        used to add sliders and other widgets to the displayed output
        """
        _s_up = self.s_up.value
        _s_down = self.s_down.value
        _n = self.n.value
        _u1 = self.u1.value

        u_idx, u = self.find_indices_of_slider(self.u_array, self.u_range)
        # nn_max = _s_up * _s_down / _n**2
        # nn_min = max(0, (_s_up + _s_down - _n) / _n)
        Sz = np.round(self.ExpVal_Sz, 5)
        Sz2 = np.round(self.ExpVal_SzSz_ii, 5)
        # Sz[0] = np.round(Sz_max, 5)
        Sz = Sz[u_idx]
        Sz2 = Sz2[u_idx]
        Sz_str = r"$\left\langle S_{iz} \right\rangle $"
        Sz2_str = r"$\left\langle S^2_{iz} \right\rangle $"
        dSz_str = r"$\left\langle \Delta S^2_{iz} \right\rangle $"

        color = mpl.cm.tab10(np.arange(0, 10))
        mpl.pyplot.rcParams["axes.prop_cycle"] = cycler("color", color)
        fig = plt.figure(figsize=(10, 6))

        u1_str = f" and $U_1 = {_u1}$" if self.u1_checkbox.value == True else ""
        _title = fill(
            r"Average Spin moment $\langle$$S_{iz}$$\rangle$, $\langle$$S^2_{iz}$$\rangle$ and $\langle$$\Delta$$S_{iz}^2$$\rangle$"f" as a function of the on-site interaction $U$ for $n = {self.n.value}$ sites with {self.s_up.value} spin up electron(s), {self.s_down.value} spin down electron(s), hopping amplitude $t = 1$" + u1_str, width=80)
        plt.title(rf"{_title}")
        plt.xlabel(r"$U$")
        plt.ylabel(
            r"Expectation value $\left\langle \hat O \right\rangle $")
        plt.grid(which="both", axis="both", linestyle="--",
                 color="black", alpha=0.4)

        plt.plot(u, Sz, ".-", label=f"{Sz_str}")
        plt.plot(u, Sz2, ".-", label=f"{Sz2_str}")
        plt.plot(u, Sz2 - Sz**2, ".-", label=f"{dSz_str}")
        # plt.plot(u, 1 - 2*self.ExpVal_nn_mean[u_idx])
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=1)
        return fig

    def Plot_ExpVal_SzSz(self, **kwargs):
        """
        Method to plot the expectation value of the spin-spin correlation operator `Sz_i Sz_j` for u in [u_min, u_max] and t=1.

        Returns
        - ------

        fig: matplotlib.figure.Figure
                                        figure object to save as image-file

        Other Parameters
        - ---------------
        **Kwargs: Widgets
                                                                        used to add sliders and other widgets to the displayed output
        """
        _s_up = self.s_up.value
        _s_down = self.s_down.value
        _n = self.n.value
        _u1 = self.u1.value

        u_idx, u = self.find_indices_of_slider(self.u_array, self.u_range)
        SzSz_ij = [S[u_idx] for S in self.ExpVal_SzSz_ij]

        SzSz_str = r"$\left\langle S_{iz} S_{jz} \right\rangle $"

        color = mpl.cm.tab10(np.arange(0, 10))
        mpl.pyplot.rcParams["axes.prop_cycle"] = cycler("color", color)
        fig = plt.figure(figsize=(10, 6))

        u1_str = f" and $U_1 = {_u1}$" if self.u1_checkbox.value == True else ""
        title = fill(
            r"Spin-spin correlation $\langle$$S_{iz}$$S_{jz}$$\rangle$ " f"as a function of the on-site interaction $U$ for $n = {self.n.value}$ sites with {self.s_up.value} spin up electron(s), {self.s_down.value} spin down electron(s), hopping amplitude $t = 1$" + u1_str, width=80)
        plt.title(rf"{title}")
        plt.xlabel(r"$U$")
        plt.ylabel(SzSz_str)
        plt.grid(which="both", axis="both", linestyle="--",
                 color="black", alpha=0.4)

        for i in np.arange(_n // 2 + 1):
            plt.plot(u, SzSz_ij[i], ".-",
                     label=r"$\langle S_{1z}S_{"f"{i+1}"r"z}\rangle$")

        plt.legend(bbox_to_anchor=(1.0, 1), loc="upper left", ncol=1)
        return fig

    def Plot_Local_Chi(self, **kwargs):
        """x
        Method to plot the local, magnetic susceptibility chi for u in [u_min, u_max] and t=1.

        Returns
        - ------

        fig: matplotlib.figure.Figure
                                        figure object to save as image-file

        Other Parameters
        - ---------------
        **Kwargs: Widgets
                                                                        used to add sliders and other widgets to the displayed output
        """
        _s_up = self.s_up.value
        _s_down = self.s_down.value
        _n = self.n.value
        _u1 = self.u1.value

        if self.is_degenerate():
            print(
                r"Degenerate ground state! Using non-degenerate perturbation theory for susceptibility not possible")
            return

        u_idx, u = self.find_indices_of_slider(self.u_array, self.u_range)

        fig = plt.figure(figsize=(10, 6))

        u1_str = f" and $U_1 = {_u1}$" if self.u1_checkbox.value == True else ""
        title = fill(
            r"Local susceptibility $\chi_\mathrm{loc}$ " f"as a function of the on-site interaction $U$ for $n = {self.n.value}$ sites with {self.s_up.value} spin up electron(s), {self.s_down.value} spin down electron(s), hopping amplitude $t = 1$" + u1_str, width=80)
        plt.title(rf"{title}")
        plt.xlabel(r"$U$")
        plt.ylabel(r"$\chi_\mathrm{loc}$")
        plt.grid(which="both", axis="both", linestyle="--",
                 color="black", alpha=0.4)

        # For U=0 some combinations of n and s_up/s_down are more than twice degenerate, to not deal with these corner cases we just skip them completely
        plt.plot(u[1:], self.Chi[u_idx][1:].round(5), ".-")
        # Has to be set after plotting, otherwise range [0,1]
        plt.ylim(bottom=-0.1,)
        return fig

    def Plot_Chi_Staggered(self, **kwargs):
        """
        Method to plot the staggered, magnetic susceptibility chi for u in [u_min, u_max] and t=1.

        Returns
        - ------

        fig: matplotlib.figure.Figure
                                        figure object to save as image-file

        Other Parameters
        - ---------------
        **Kwargs: Widgets
                                                                        used to add sliders and other widgets to the displayed output
        """
        _s_up = self.s_up.value
        _s_down = self.s_down.value
        _n = self.n.value
        _u1 = self.u1.value

        if self.is_degenerate():
            print(
                r"Degenerate ground state! Using non-degenerate perturbation theory for susceptibility not possible")
            return

        u_idx, u = self.find_indices_of_slider(self.u_array, self.u_range)

        fig = plt.figure(figsize=(10, 6))

        u1_str = f" and $U_1 = {_u1}$" if self.u1_checkbox.value == True else ""
        title = fill(
            r"Staggered susceptibility $\chi_\mathrm{staggered}$ " f"as a function of the on-site interaction $U$ for $n = {self.n.value}$ sites with {self.s_up.value} spin up electron(s), {self.s_down.value} spin down electron(s), hopping amplitude $t = 1$" + u1_str, width=80)
        plt.title(rf"{title}")
        plt.xlabel(r"$U$")
        plt.ylabel(r"$\chi_\mathrm{staggered}$")
        plt.grid(which="both", axis="both", linestyle="--",
                 color="black", alpha=0.4)

        plt.plot(u[1:], self.Chi_staggered[u_idx][1:].round(5), ".-")
        return fig

    @Cach
    def All_Eigvals_and_Eigvecs(self):
        """
        Method to calculate all m eigenvalues and eigenvectors for all U in [u_min, u_max] and t=1.

        Returns
        - ------
        eigvals, eigvecs: list of numpy.ndarray with shape(len(u_array), m)
        """
        _u1 = self.u1.value
        _H = np.array([self.H(u, 1, _u1) for u in self.u_array])
        return np.linalg.eigh(_H)

    def Calc_Coupling(self, Op):
        """
        Helper function to calculate the perturbative coupling to the given operator, given by Sum_{n > g} | <psi_n | Op | psi_g > | ^ 2 / (E_n - E_g) of the system.

        Returns
        - ------
        Coupling: ndarray(len(u),)
        """

        eig_vals, eig_vecs = self.All_Eigvals_and_Eigvecs

        # deal with degenerate ground states
        if self.is_degenerate():
            eig_vec0 = np.sum(eig_vecs[:, :, :2], axis=2) / np.sqrt(2)
            eig_val0 = eig_vals[:, 0]
            shifted_eigvals = eig_vals[:, 2:] - eig_val0[:, None]

            return 2 * np.sum(np.einsum("ijk, ji->ik", eig_vecs[:, :, 2:], Op @ eig_vec0.T)**2 / shifted_eigvals, axis=1)

        # deal with non-degenerate ground state
        else:
            eig_vec0 = eig_vecs[:, :, 0]
            eig_val0 = eig_vals[:, 0]
            shifted_eigvals = eig_vals[:, 1:] - eig_val0[:, None]

            return 2 * np.sum(np.einsum("ijk, ji->ik", eig_vecs[:, :, 1:], Op @ eig_vec0.T)**2 / shifted_eigvals, axis=1)

    def Plot_ExpVal_SzkSzk(self, **kwargs):
        """
        Method to plot the expectation value of the reciprocal-space spin-spin correlation operator `Sz_k Sz_-k` for u in [u_min, u_max] and t=1.

        Returns
        - ------

        fig: matplotlib.figure.Figure
                                        figure object to save as image-file

        Other Parameters
        - ---------------
        **Kwargs: Widgets
                                                                        used to add sliders and other widgets to the displayed output
        """
        _s_up = self.s_up.value
        _s_down = self.s_down.value
        _n = self.n.value
        _u1 = self.u1.value

        u_idx, u = self.find_indices_of_slider(self.u_array, self.u_range)
        SzSz_kk = [S[u_idx] for S in self.ExpVal_SzkSzk]

        SzSz_kk_str = r"$\left\langle S_z(k) S_z(-k) \right\rangle $"

        color = mpl.cm.tab10(np.arange(0, 10))
        mpl.pyplot.rcParams["axes.prop_cycle"] = cycler("color", color)
        fig = plt.figure(figsize=(10, 6))

        u1_str = f" and $U_1 = {_u1}$" if self.u1_checkbox.value == True else ""
        title = fill(
            r"Reciprocal-space spin-spin correlation $\langle$$S_z(k)$$S_z(-k)$$\rangle$ " f"as a function of the on-site interaction $U$ for $n = {self.n.value}$ sites with {self.s_up.value} spin up electron(s), {self.s_down.value} spin down electron(s), hopping amplitude $t = 1$" + u1_str, width=80)
        plt.title(rf"{title}")
        plt.xlabel(r"$U$")
        plt.ylabel(SzSz_kk_str)
        plt.grid(which="both", axis="both", linestyle="--",
                 color="black", alpha=0.4)

        labels = [str(Fraction(2 / _n * i).limit_denominator(100))
                  for i in np.arange(np.floor(_n / 2) + 1)]

        for i, S in enumerate(SzSz_kk):
            plt.plot(u, S.round(2), ".-",
                     label=rf"$k = {{{labels[i]}}} \cdot \pi $")

        # Has to be set after plotting, otherwise range [0,1]
        plt.ylim(bottom=-0.1,)
        plt.legend(bbox_to_anchor=(1.0, 1), loc="upper left", ncol=1)
        return fig
