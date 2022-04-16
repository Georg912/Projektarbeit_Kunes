# #Module for arbitrary hopping by user defined hopping and transition matrices


from re import A
from more_itertools import distinct_permutations

import scipy.spatial.distance as sp
import scipy
import numpy as np
import ipywidgets as widgets
import time  # , math, copy
# # from IPython.display import clear_output
# # from distutils.spawn import find_executable
import matplotlib.pyplot as plt  # Plotting
import matplotlib as mpl
from cycler import cycler  # used for color cycles in mpl

from Modules.Widgets import n_Slider, s_up_Slider, s_down_Slider
from Modules.Widgets import u_Slider, t_Slider, u_range_Slider, t_range_Slider
from Modules.Widgets import basis_index_Slider
from Modules.Cache_Decorator import Cach  # used for caching functions
# # from Module_Widgets_and_Sliders import button_to_add, button_to_undo, button_to_reset, button_to_show
# # from Module_Widgets_and_Sliders import i_IntText, j_IntText, p_BoundedFloatText
# # from Module_Widgets_and_Sliders import checkbox_periodic_boundary, p1_BoundedFloatText, p2_BoundedFloatText, p3_BoundedFloatText
# # from scipy.sparse import diags

# # mpl.rcParams['text.usetex'] = True
import numba as nb
from numba import jit, njit

# #possible TODOS:
#     #write function to store hopping matrices and load them
#     #clean class
#     #document module
#     # implement sparse matrices
#     # implement faster calcualtion of operator matrices


@jit
def is_single_hopping_process(x, y, cre, anh):
    """
    returns `True` if hopping from state `x` at postion `cre` to state `y` at position `anh` is allowed, i.e. if it is a single hopping process which transforms state x into state y.

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


@jit
def hop_sign(state, i):
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

    def __init__(self, n=6, s_up=3, s_down=3):

        self.out = widgets.Output()

        # Set all site and spin sliders
        ############################################
        self.n = n_Slider
        self.n.value = n
        self.n.observe(self.on_change_n, names="value")

        self.s_up = s_up_Slider
        self.s_up.value = s_up
        self.s_up.observe(self.on_change_s_up, names="value")

        self.s_down = s_down_Slider
        self.s_down.value = s_down
        self.s_down.observe(self.on_change_s_down, names="value")

        # Calculate basis states and set index sliders
        ##############################################
        self.basis = self.Construct_Basis()
        self.basis_index = basis_index_Slider
        self.basis_index.max = self.basis.shape[0] - 1

        self.hoppings = self.Allowed_Hoppings_H()

        self.Reset_H()

        # Set all u and t Sliders
        ##############################################
        self.u = u_Slider
        self.t = t_Slider
        self.u_range = u_range_Slider
        self.t_range = t_range_Slider
        self.u_array = np.linspace(self.u_range.min, self.u_range.max, num=int(
            2*self.u_range.max + 1), endpoint=True)
        self.t_array = np.linspace(self.t_range.min, self.t_range.max, num=int(
            2*self.t_range.max + 1), endpoint=True)

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

    def show_basis(self, index, **kwargs):
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

    def on_change_n(self, change):
        self.basis = self.Construct_Basis()
        self.basis_index.max = self.basis.shape[0] - 1
        self.hoppings = self.Allowed_Hoppings_H()
        self.Reset_H()

    def on_change_s_up(self, change):
        if (self.s_down.value > self.n.value or self.s_up.value > self.n.value):
            pass
        else:
            self.basis = self.Construct_Basis()
            self.basis_index.max = self.basis.shape[0] - 1
            self.Reset_H()

    def on_change_s_down(self, change):
        if (self.s_down.value > self.n.value or self.s_up.value > self.n.value):
            pass
        else:
            self.basis = self.Construct_Basis()
            self.basis_index.max = self.basis.shape[0] - 1
            self.Reset_H()

    def up(self):
        """
        Return all possible spin up states.

        Returns
        -------
        basis : ndarray (m, n)
            array of spin up basis states
        """
        return self.basis[:, : self.n.value]

    def down(self):
        """
        Return all possible spin down states.

        Returns
        -------
        basis : ndarray (m, n)
                        array of spin down basis states
        """
        return self.basis[:, self.n.value:]

    @Cach
    def Op_n(self):
        """
        Return the occupation number operator `n`, which is diagonal in the occupation number basis

        Returns
        -------
        n : ndarray (m)
        """
        return np.diag(np.sum(self.up() * self.down(), axis=1))

    # def diag(self, i):
    #     return self.basis[:, i] * self.basis[:, self.n.value + i]

    # def Double(self, i):
    #     return np.diag(self.diag(i))

    # def DoubleSiteAvg(self):
    #     return np.diag(self.nn() / self.n.value)
    #     # np.mean(sup(basis) * down(basis), axis=1))

    # def OpSz(self, i):
    #     return np.diag((self.up() - self.down())[:, i])

    # def OpSzSz(self, i, j):
    #     return self.OpSz(i) @ self.OpSz(j)

    @jit(forceobj=True)  # otherwise error message
    def Allowed_Hoppings_H(self):
        """
        Calculates all allowed hoppings in the Hamiltonian H from position `r1` to position `r2` for a given `n`.

        Returns
        -------
        hoppings : ndarray (4n, 2)
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
        hoppings : ndarray (4n, 2)
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
        A : ndarray (m, 2n)
            Basis to hop into
        B : ndarray (m, 2n)
            Basis to hop out of
        ii : int
            index of site to hop into
        jj : int
            index of site to hop out of

        Returns
        -------
        C : ndarray (m, m)
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
    def Hu(self):
        """
        Return cached on-site interaction Hamiltonian H_u given exactly by the occupation number operator `n`.

        Returns
        -------
        Hu : ndarray (m, m)
        on-site Interaction Hamiltonian
        """
        return self.Op_n

    def H(self, u, t):
        """
        Compute the system's total Hamiltonian H = u*Hu + t*Ht for given prefactors `u` and `t`.

        Parameters
        ----------
        u : float
                on-site interaction strength
        t : float
                hopping strength
        """
        return t * self.Ht + u * self.Hu

    def Show_H(self, u, t, **kwargs):
        """
        Method to print total Hamiltonian H = u*Hu + t*Ht and the dimension of H

        Parameters
        ----------
        u : float
            on-site interaction strength due to electric field
        t : float
            global hopping amplitude from site i to site j

        Returns
        -------
        None

        Other Parameters
        ----------------
        **Kwargs : Widgets
            used to add sliders and other widgets to the displayed output
        """
        dimension = self.basis_index.max + 1
        print(f"Dimension of H = {dimension} x {dimension}")
        print(f"H = \n{self.H(u,t)}")

    def Reset_H(self):
        """
        Method to reset the cached Hamiltonian H.
        """
        try:
            del self.Op_n
        except:
            pass
        try:
            del self.Ht
        except:
            pass
        try:
            del self.Hu
        except:
            pass
        try:
            del self.Eigvals_Hu
        except:
            pass
        try:
            del self.Eigvals_Ht
        except:
            pass

    @Cach
    @jit(forceobj=True, cache=True)
    def Eigvals_Hu(self):
        """
        Calculates the eigenvalues of the on-site interaction Hamiltonian Hu for hopping amplitude t = 1.

        Returns
        -------
        vals : ndarray (len(u),m)
                        Eigenvalues of the on-site interaction Hamiltonian
        """
        vals = [np.linalg.eigvalsh(self.H(u, 1)) for u in self.u_array]
        return np.array(vals)

    def Plot_Eigvals_Hu(self, **kwargs):
        """
        Method to plot the eigenvalues of the on-site interaction Hamiltonian H_u for constant hopping amplitude t=1. Method als plots the number of eigenvalues for the atomic limit as labels for the correspondingly colored curves.

        Returns
        -------
        fig : matplotlib.figure.Figure
                        figure object to save as image-file
        Other Parameters
        ----------------
        **Kwargs : Widgets
            used to add sliders and other widgets to the displayed output
        """

        # find indices of u_array that are within the range `u_range_Slider``
        u_min = self.u_range.value[0]
        u_max = self.u_range.value[1]
        u = self.u_array

        u_idx = np.nonzero(np.logical_and(u <= u_max, u >= u_min))
        u = u[u_idx]

        last_eigvals_u = self.Eigvals_Hu[-1, :]
        sorted_eigval_idx = np.argsort(last_eigvals_u)
        eig_u = self.Eigvals_Hu[u_idx][:, sorted_eigval_idx]

        uniq = np.unique(np.diag(self.Op_n).astype(int), return_counts=True)
        color = mpl.cm.tab10(np.repeat(np.arange(uniq[0].size), uniq[1]))
        mpl.pyplot.rcParams["axes.prop_cycle"] = cycler("color", color)

        fig = plt.figure(figsize=(10, 6))
        plt.title(
            f"Eigenvalues of Hubbard-Ring Hamiltonian $H$ as a function of the on-site interaction strength $U$ for $n={self.n.value}$ sites \n with {self.s_up.value} spin up electron(s), {self.s_down.value} spin down electron(s) and hopping amplitude $t = 1$")
        plt.xlabel(r"$U$")
        plt.ylabel(r"Eigenvalue(s)")
        plt.grid()
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
        vals : ndarray (len(u),m)
                        Eigenvalues of the hopping Hamiltonian
        """
        vals = [np.linalg.eigvalsh(self.H(10, t)) for t in self.t_array]
        return np.array(vals)

    def Plot_Eigvals_Ht(self, **kwargs):
        """
        Method to plot the eigenvalues of the hopping Hamiltonian H_t for constant on-site interaction u=10. Method als plots the number of eigenvalues for the atomic limit as labels for the correspondingly colored curves.

        Returns
        -------
        fig : matplotlib.figure.Figure
                        figure object to save as image-file
        Other Parameters
        ----------------
        **Kwargs : Widgets
            used to add sliders and other widgets to the displayed output
        """
        # find indices of t_array that are within the range `t_range_Slider``
        t_min = self.t_range.value[0]
        t_max = self.t_range.value[1]
        t = self.t_array

        t_idx = np.nonzero(np.logical_and(t <= t_max, t >= t_min))
        t = t[t_idx]

        last_eigvals_t = self.Eigvals_Ht[-1, :]
        sorted_eigval_idx = np.argsort(last_eigvals_t)
        eig_t = self.Eigvals_Ht[t_idx][:, sorted_eigval_idx]
        uniq = np.unique(np.diag(self.Op_n).astype(int), return_counts=True)
        color = mpl.cm.tab10(np.repeat(np.arange(uniq[0].size), uniq[1]))
        mpl.pyplot.rcParams["axes.prop_cycle"] = cycler("color", color)

        fig = plt.figure(figsize=(10, 6))
        plt.title(
            f"Eigenvalues of Hubbard-Ring Hamiltonian $H$ as a function of the hopping amplitude $t$ for $n={self.n.value}$ sites \n with {self.s_up.value} spin up electron(s), {self.s_down.value} spin down electron(s) and on-site interaction $U=10$")
        plt.xlabel(r"$t$")
        plt.ylabel(r"Eigenvalue(s)")
        plt.grid()
        axes = plt.plot(t, eig_t, ".-")

        for idx, num in enumerate(np.cumsum(uniq[1])):
            axes[num-1].set_label(f"{uniq[1][idx]}")

        plt.gca().invert_xaxis()
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left",
                   ncol=1, title="# of eigenvalues")
        # plt.show()
        return fig

    def Plot_Eigvals_H(self, **kwargs):
        # find indices of t_array that are within the range `t_range_Slider``
        t_min = self.t_range.value[0]
        t_max = self.t_range.value[1]
        t = self.t_array

        t_idx = np.nonzero(np.logical_and(t <= t_max, t >= t_min))
        t = t[t_idx]

        # find indices of u_array that are within the range `u_range_Slider``
        u_min = self.u_range.value[0]
        u_max = self.u_range.value[1]
        u = self.u_array

        u_idx = np.nonzero(np.logical_and(u <= u_max, u >= u_min))
        u = u[u_idx]

        eig_t = np.array([np.linalg.eigvalsh(self.H(u_max, t))
                         for t in self.t_array])
        eig_u = np.array([np.linalg.eigvalsh(self.H(u, t_max))
                         for u in self.u_array])

        last_eigvals_t = eig_t[-1, :]
        sorted_eigval_idx = np.argsort(last_eigvals_t)
        eig_t = eig_t[t_idx][:, sorted_eigval_idx]

        last_eigvals_u = eig_u[-1, :]
        sorted_eigval_idx = np.argsort(last_eigvals_u)
        eig_u = eig_u[u_idx][:, sorted_eigval_idx]

        uniq = np.unique(np.diag(self.Op_n).astype(int), return_counts=True)
        color = mpl.cm.tab10(np.repeat(np.arange(uniq[0].size), uniq[1]))
        mpl.pyplot.rcParams["axes.prop_cycle"] = cycler("color", color)

        fig, axes = plt.subplots(ncols=2, sharey=True,
                                 figsize=(15, 8))
        fig.subplots_adjust(wspace=0.02)

        axes[0].set_xlabel(r"$U$")
        axes[0].set_ylabel(r"Eigenvalue(s)")
        axes[1].set_xlabel(r"$t$")
        axes[0].grid()
        axes[1].grid()

        axes[0].plot(u, eig_u, ".-")
        ax = axes[1].plot(t, eig_t, ".-")

        for idx, num in enumerate(np.cumsum(uniq[1])):
            ax[num-1].set_label(f"{uniq[1][idx]}")

        fig.suptitle(
            f"Eigenvalues of Hubbard-Ring Hamiltonian $H$ for $n={self.n.value}$ sites with {self.s_up.value} spin up electron(s) and {self.s_down.value} spin down electron(s). \n On the left as a function of $U$ for constant $t={t_max}$ and on the right as a function of $t$ for constant $U={u_max}$")
        fig.gca().invert_xaxis()
        fig.legend(bbox_to_anchor=(0.9, 0.9), loc="upper left",
                   ncol=1, title="# of eigenvalues")
        # fig.show()
        return fig
