# #Module for arbitrary hopping by user defined hopping and transition matrices


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
# # from cycler import cycler #used for color cycles in mpl
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

        self.u = u_Slider
        self.t = t_Slider
        self.u_range = u_range_Slider
        self.t_range = t_range_Slider

        self.eig_t = None

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
        A : ndarray (m, n)
            Basis to hop into
        B : ndarray (m, n)
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
    @jit(forceobj=True, cache=True)
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
        self._H = None
        self._Hu = None
        self._Ht = None
        self.eig_u = None
        self.eig_t = None
        try:
            del self.Op_n
            del self.Ht
            del self.Hu
        except:
            pass

    def Calc_Eigvals_u(self, steps=10):
        u_array = np.linspace(self.u_range.min, self.u_range.max, num=100)
        vals = [np.linalg.eigvalsh(self.Calc_H(u, 1)) for u in u_array]
        self.eig_u = np.array(vals)
        self.u_array = u_array
        return np.array(vals), u_array

    def Calc_Eigvals_t(self, steps=10):
        t_array = np.linspace(self.t.min, self.t.max, num=steps)
        vals = [np.linalg.eigvalsh(self.Calc_H(10, t)) for t in t_array]
        self.eig_t = np.array(vals)
        self.t_array = t_array

        return np.array(vals), t_array

    def Plot_Eigvals_u(self, **kwargs):
        if self.eig_u is None:
            with self.out:
                print("test")
            self.Calc_Eigvals_u(kwargs.get("steps", 10))

        fig = plt.figure(figsize=(10, 6))
        plt.title(
            f"Eigenvalues of Hubbard-Ring Hamiltonian H as a function of TODO $u$ for $n={self.n.value}$ sites \n with {self.s_up.value} spin up electron(s) and {self.s_down.value} spin down electron(s) and hopping amplitude $t = 1$")
        plt.xlabel(r"todo $u$")
        plt.ylabel(r"Eigenvalue(s)")
        plt.grid()
        _u = self.u_array[self.u_array <= self.u_range.value[1]]
        _u = _u[_u >= self.u_range.value[0]]

        index_u_upper = np.where(self.u_array <= self.u_range.value[1], 1, 0)
        index_u_lower = np.where(self.u_array >= self.u_range.value[0], 1, 0)

        _eig_u = self.eig_u[np.argwhere(
            (index_u_lower * index_u_upper) >= 1)].reshape(_u.size, -1)
        axes = plt.plot(_u, _eig_u, ".-", c="orange")

        n_bins = self.Get_Bin_Number_u()
        max_eigvals = np.linalg.eigvalsh(self.Calc_H(self.u_range.max, 1))
        bins = np.histogram(max_eigvals, bins=n_bins)[1]
        digs = np.digitize(max_eigvals, bins)
        digs[-1] -= 1
        color_list = mpl.cm.get_cmap('tab10')
        unique = np.unique(digs, return_counts=True)[1]
        indices = np.cumsum(unique)

        count = 0
        for idx, ax in enumerate(axes):
            ax.set_color(color_list(digs[idx]))
            if idx == indices[count]-1:
                ax.set_label(unique[count])
                count += 1

        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left",
                   ncol=1, title="# of eigenvalues")
        plt.show()
        return fig

    def Plot_Eigvals_t(self, **kwargs):
        eigvals, t_array = self.Calc_Eigvals_t(kwargs.get("steps", 10))

        fig = plt.figure(figsize=(10, 6))
        plt.title(
            f"Eigenvalues of Hubbard-Ring Hamiltonian H as a function of TODO $u$ for $n={self.n.value}$ sites \n with {self.s_up.value} spin up electron(s) and {self.s_down.value} spin down electron(s) and hopping amplitude $t = 1$")
        plt.xlabel(r"todo $t$")
        plt.ylabel(r"Eigenvalue(s)")
        plt.grid()
        axes = plt.plot(t_array, eigvals, ".-", c="orange")

        n_bins = self.Get_Bin_Number_u()
        max_eigvals = np.linalg.eigvalsh(self.Calc_H(self.u_range.max, 1))
        bins = np.histogram(max_eigvals, bins=n_bins)[1]
        digs = np.digitize(max_eigvals, bins)
        digs[-1] -= 1
        color_list = mpl.cm.get_cmap('tab10')
        unique = np.unique(digs, return_counts=True)[1]
        indices = np.cumsum(unique)

        count = 0
        for idx, ax in enumerate(axes):
            ax.set_color(color_list(digs[idx]))
            if idx == indices[count]-1:
                ax.set_label(unique[count])
                count += 1
        plt.gca().invert_xaxis()
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left",
                   ncol=1, title="# of eigenvalues")
        plt.show()
        return fig

    def Get_Bin_Number_u(self):
        _n = self.n.value
        _s_up = self.s_up.value
        _s_down = self.s_down.value

        if _n == 6 and _s_down == 3 and _s_up == 3:
            return 4

        if _n == 7:
            if _s_up == 3 or _s_up == 4:
                if _s_down == 3 or _s_down == 4:
                    return 4
        return 3


# def

# self.Reset_H_and_T()
# self.Add_All_Buttons()
# self.Add_ijp_widgets()

# self.old_Hs = []
# self.old_Ts = []

# self.pbc_checkbox = checkbox_periodic_boundary
# self.pbc_checkbox.observe(self.tick_boundary_checkbox, names="value")
# self.hide()

#     def tick_boundary_checkbox(self, change):
#         self.hide()
#         self.Reset_H_and_T()

#     def hide(self):
#         if self.pbc_checkbox.value == True:
#             self.i.layout.visibility = "hidden"
#             self.j.layout.visibility = "hidden"
#             self.p.layout.visibility = "hidden"
#             self.p1.layout.visibility = "visible"
#             self.p2.layout.visibility = "visible"
#             self.p3.layout.visibility = "visible"
#             self.button_to_undo.layout.visibility = "hidden"
#         else:
#             self.i.layout.visibility = "visible"
#             self.j.layout.visibility = "visible"
#             self.p.layout.visibility = "visible"
#             self.p1.layout.visibility = "hidden"
#             self.p2.layout.visibility = "hidden"
#             self.p3.layout.visibility = "hidden"
#             self.button_to_undo.layout.visibility = "visible"

#     def is_checkbox(self):
#         return self.checkbox.value

#     def Add_All_Buttons(self):
#         self.button_to_add = button_to_add
#         self.button_to_undo = button_to_undo
#         self.button_to_reset = button_to_reset
#         self.button_to_show = button_to_show

#         self.button_to_add.on_click(self.click_add)
#         self.button_to_undo.on_click(self.click_undo)
#         self.button_to_reset.on_click(self.click_reset)
#         self.button_to_show.on_click(self.click_show)

#         self.out = widgets.Output()
#         self.checkbox = checkbox

#     def Add_ijp_widgets(self):
#         self.i = i_IntText
#         self.j = j_IntText
#         self.p = p_BoundedFloatText

#         self.p1 = p1_BoundedFloatText
#         self.p2 = p2_BoundedFloatText
#         self.p3 = p3_BoundedFloatText

#     def Show_H_and_T(self):
#         with self.out:
#             clear_output()
#             print(f"H = ", self.H, "", sep="\n")
#             print(f"T = ", self.T, sep="\n")

#     #TODO: check if n is large enough for p2 and p3
#     def Add_Hop_PBC(self):
#         _n = self.n.value
#         _p1 = self.p1.value
#         _p2 = self.p2.value
#         _p3 = self.p3.value

#         H1, H2, H3 = np.zeros((_n,_n)), np.zeros((_n,_n)), np.zeros((_n,_n))
#         if self.p1.value > 0.:
#             diagonal_entries = [np.ones(_n-1), np.ones(_n-1)]
#             H1 = diags(diagonal_entries, [-1, 1]).toarray()
#             if _n >= 3:
#                 H1[[0, _n-1], [_n-1, 0]] = 1
#         if self.p2.value > 0.:
#             diagonal_entries = [np.ones(2), np.ones(_n-2), np.ones(_n-2), np.ones(2)]
#             if _n >= 5:
#                 H2 = diags(diagonal_entries, [-_n+2, -2, 2, _n-2]).toarray()
#                 H2[[0, _n-2], [_n-2, 0]] = 1
#         if self.p3.value > 0.:
#             if _n >= 7:
#                 diagonal_entries = [np.ones(3), np.ones(_n-3), np.ones(_n-3), np.ones(3)]
#                 H3 = diags(diagonal_entries, [-_n+3, -3, 3, _n-3]).toarray()
#         assert 2 * (_p1 + _p2 * (_n>=5) + _p3 * (_n>=7)) <= 1. , self.out.append_stderr(f"Error negative probability. For n = {_n}, p0 = 1 - 2(p1{' + p2' if _n>=5 else ''}{' + p3' if _n>=7 else ''}) = 1 - 2 * ({_p1}{' + ' + str(_p2) if _n>=5 else ''}{' + ' +str(_p3) if _n>=7 else ''}) = {1 - 2 * (_p1 + (_p2 if _n>=5 else 0) + (_p3 if _n>=7 else 0)):.3f})")

#         _H = H1 + H2 + H3
#         _T = (1 - 2 * (_p1 + _p2 * (_n>=5) + _p3 * (_n>=7))) * np.eye(_n) + _p1 * H1 + _p2 * H2 + _p3 * H3
#         self.H = _H
#         self.T = _T
#         return
#     ### take care of the values not on the lower and upper main diagonal
#         #H[[0, n-1], [n-1, 0]] = 1

#     def AddHop(self):
#         if self.pbc_checkbox.value == True:
#             self.Add_Hop_PBC()
#             return None
#         with self.out:
#             clear_output()
#         _i = self.i.value - 1
#         _j = self.j.value - 1
#         _p = self.p.value
#         _n_max = self.n.max

#         if _i != _j:
#             assert self.T[_i, _i] -_p >= 0 , self.out.append_stderr(f"Error hopping would have caused T[{_i},{_i}] <=0.")

#         self.H[_i, _j] = 1.
#         self.H[_j, _i] = 1.
#         self.T[_i, _j] += _p
#         self.T[_j, _i] += _p
#         self.T[_i, _i] -= _p
#         self.T[_j, _j] -= _p

#         #add to Ts and Hs after second hopping
#         self.old_Hs.append(copy.deepcopy(self.H))
#         self.old_Ts.append(copy.deepcopy(self.T))

#     def Reset_H_and_T(self):
#         _n = self.n.value
#         self.H = np.eye(_n)
#         self.T = np.eye(_n)

#     #realy memory inefficent, work in progress
#     def Undo_Hopping(self):
#         if len(self.old_Hs) > 1:
#             self.H = self.old_Hs[-2]
#             self.T = self.old_Ts[-2]
#             self.old_Hs.pop()
#             self.old_Ts.pop()
#         else:
#             self.Reset_H_and_T()

#     def click_undo(self, b):
#         self.Undo_Hopping()

#         if self.is_checkbox():
#             with self.out:
#                 print("undone")
#                 time.sleep(2)
#                 clear_output()

#     def click_add(self, b):
#         self.AddHop()

#         if self.is_checkbox():
#             with self.out:
#                 print("added hopping")
#                 time.sleep(2)
#                 clear_output()

#     def click_reset(self, b):
#         self.Reset_H_and_T()

#         if self.is_checkbox():
#             with self.out:
#                 print("H and T reset")
#                 time.sleep(2)
#                 clear_output()

#     def click_show(self, b):
#         self.Show_H_and_T()

#     def Calc_Markov(self, state=[1,0,0,0,0,0], n_its=400):
#     ### Check if state is valid, i.e real, of unit length and compatible with `n`.
#         _n = self.n.value
#         assert len(state) == _n, f"Dimension of the state vector {state} = {len(state)} != n = {_n}."
#         assert not any(isinstance(num, complex) for num in state), f"Markovian evolution cannot deal with complex state {state}."
#         assert math.isclose(sum(state), 1, rel_tol=1e-04), f"The norm of the state vector {state} = {sum(state)} != 1."
#         assert any(math.isclose(num, 1, rel_tol=1e-04) for num in state), f"In the beginning, particle must be at a definite site, e.g. [1, 0, 0, 0, 0, 0]. You have state = {state}."

#         ### check if `n_its` is a positive integer
#         assert n_its >= 0, "n_its must be greater or equal to 0."
#         assert type(n_its) == int, f"n_its must be an integer not {type(_n)}"

#         state = np.array(state)
#         observations = [state]
#         for _ in np.arange(n_its):
#             state = self.T @ state
#             observations.append(state)
#         return np.array(observations)

#     def Plot_Markov(self, state=[1,0,0,0,0,0], n_its=400):
#     ### Calculate states
#         observations = self.Calc_Markov(state, n_its)
#         _n = self.n.value
#         fig = plt.figure(figsize=(10,6))

#         ### make plot pretty
#         plt.title(f"Markov evolution for graph with $n={_n}$ sites, with initial state {state}")
#         plt.xlabel(r"Number of iterations $n_{\mathrm{its}}$")
#         plt.ylabel(r"Probability of finding particle at site $i$")
#         plt.grid()

#         ### Ensure color order is consistent with site number
#         mpl.rcParams['axes.prop_cycle'] = cycler("color", plt.cm.get_cmap("tab10").reversed().colors[-_n:])
#         colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']

#         ### actual plotting
#         for i, site in enumerate(np.arange(_n)[::-1]):
#             plt.plot(observations[:, site], ".-", label=f"Site {site+1}", color=colors[i])
#         ax = plt.gca()
#         handles, labels = ax.get_legend_handles_labels()
#         plt.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.05, 1), loc="upper left", ncol=2)

#         if self.find_latex():
#             matrix = self.Latex_Matrix() % tuple(self.T.flatten())
#             plt.annotate(text=f"T = {matrix}", xy=(0,0), xytext=(1.02, 0.0), xycoords="axes fraction", horizontalalignment="left", verticalalignment="bottom")
#         else:
#             matrix = self.T.round(2)
#             plt.annotate(text="T = ""\n"f"{matrix}", xy=(0,0), xytext=(1.02, 0.0), xycoords="axes fraction", horizontalalignment="left", verticalalignment="bottom")

#         plt.show()
#         return fig

#     def Latex_Matrix(self, precision=2):
#         _p = precision
#         _N = self.n.value

#         beginning=r"$ \left( \begin{array}{"
#         formatting=_N*r'c'+r'}'
#         array_rows=(_N-1)*((_N-1)*rf"%.{_p}f & "+rf"%.{_p}f \\")
#         final_row=(_N-1)*rf"%.{_p}f & "+rf"%.{_p}f "
#         end=r"\end{array} \right) $"

#         matrix=beginning+formatting+array_rows+final_row+end
#         return matrix

#     def Latex_Matrix2(self, precision=1):
#         _p = precision
#         _N = self.n.value

#         beginning=r"$ \left( \begin{array}{"
#         formatting=_N*r'c'+r'}'
#         array_rows=(_N-1)*((_N-1)*rf"%.{_p}f \mathrm{{e}}^{{%.{_p}f \mathrm{{i}}}} & "+rf"%.{_p}f \mathrm{{e}}^{{%.{_p}f \mathrm{{i}}}} \\")
#         final_row=(_N-1)*rf"%.{_p}f \mathrm{{e}}^{{%.{_p}f \mathrm{{i}}}} & "+rf"%.{_p}f \mathrm{{e}}^{{%.{_p}f \mathrm{{i}}}} "
#         end=r"\end{array} \right) $"

#         matrix=beginning+formatting+array_rows+final_row+end
#         return matrix

#     def Time_Evolution_Operator(self):
#         return expm(-1j * self.T)

#     def Calc_QM(self, state=[1,0,0,0,0,0], n_its=400):
#         """TODO: add doc string"""
#         _n = self.n.value
#         #Check if state is valid
#         #TODO: possibly add optin of automatically normalizing state
#         assert math.isclose(np.linalg.norm(state), 1, rel_tol=1e-04), f"The norm of the state vector {state} = {np.linalg.norm(state)} != 1"

#         ### check if `n_its` is a positive integer
#         assert n_its >= 0, "n_its must be greater or equal to 0."
#         assert type(n_its) == int, f"n_its must be an integer not {type(_n)}"

#         U = self.Time_Evolution_Operator()

#         state = np.array(state)
#         observations = [state]
#         for _ in np.arange(n_its):
#             state = U @ state
#             observations.append(state)
#         return np.array(observations)

#     def Plot_QM_Evolution(self, state=[1,0,0,0,0,0], n_its=400):
#         #TODO: write documentation
#         observations = self.Calc_QM(state, n_its)
#         _n = self.n.value

#         fig = plt.figure(figsize=(10,6))
#         plt.title(f"QM evolution for graph with $n={_n}$ sites, with initial state {state}")
#         plt.xlabel(r"Number of iterations $n_{\mathrm{its}}$")
#         plt.ylabel(r"Probability of finding particle at site $i$")
#         plt.grid()

#         mpl.rcParams['axes.prop_cycle'] = cycler("color", plt.cm.get_cmap("tab10").reversed().colors[-_n:])

#         for site in np.arange(len(state))[::-1]:
#             plt.plot(np.abs(observations[:, site])**2, ".-", label=f"Site {site+1}", )
#         ax = plt.gca()
#         handles, labels = ax.get_legend_handles_labels()
#         plt.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.05, 1), loc="upper left", ncol=2)

#         U = self.Time_Evolution_Operator()
#         phase = np.angle(U)
#         magnitude = np.abs(U)
#         vals = tuple(np.dstack((magnitude, phase)).flatten())

#         if self.find_latex():
#             matrix = self.Latex_Matrix2() % vals
#             plt.annotate(text=f"U = {matrix}", xy=(0,0), xytext=(1.02, 0.0), xycoords="axes fraction", horizontalalignment="left", verticalalignment="bottom")
#         else:
#             matrix = U.round(2)
#             plt.annotate(text="U = ""\n"f"{matrix}", xy=(0,0), xytext=(1.02, 0.0), xycoords="axes fraction", horizontalalignment="left", verticalalignment="bottom")

#         plt.show()
#         return fig

#     def find_latex(self):
#         return find_executable('latex')
