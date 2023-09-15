from ..M04_Hubbard_Model.Module_Hubbard_Class import Hubbard, is_single_hopping_process, hop_sign
from ..M04_Hubbard_Model.Module_Cache_Decorator import Cach
from .Module_Widgets import u25_Slider, omega_range_Slider, delta_Slider, site_i_Slider, site_j_Slider, mu_Slider
import matplotlib.pyplot as plt  	# Plotting
import matplotlib as mpl
from cycler import cycler  			# used for color cycles in mpl
from textwrap import fill  			# used for text wrapping in mpl
from fractions import Fraction  	# used for fractions in mpl
import numpy as np
from more_itertools import distinct_permutations

import numba as nb  # used for just in time compilation
from numba import jit, njit  # used for just in time compilation

import inspect  # used to get source code of functions
# used to calculate distances between vectors
import scipy.spatial.distance as sp
import scipy.linalg as sp_la
from enum import Enum, auto  # used for enumerations


class Sector(Enum):
    """
    Enumeration of the different sectors of the Hilbert space.
    """
    PLUS = "plus"
    MINUS = "minus"


class DynamicalHubbard(Hubbard):

    def __init__(self, n=None, s_up=None, s_down=None):

        # set all new sliders
        self.u25 = u25_Slider  		# on-site Interaction strength
        self.delta = delta_Slider  	# control width of Lorentzian

        self.omega_range = omega_range_Slider
        _w_max = self.omega_range.max
        _w_min = self.omega_range.min
        _dw = (_w_max - _w_min) / self.omega_range.step
        self.omega_array = np.linspace(_w_min, _w_max, num=int(_dw) + 1,
                                       endpoint=True)

        super().__init__(n, s_up, s_down)  # call parent initilizer

    def Calc_GS_Overlap_Elements(self, Op):
        """
        Calculate the two dimensional matrix element array mel[U, i] = <psi_i| Op |psi_g>| of the system for all m eigenvectors of H(U).

        Returns
        -------
        Coupling : ndarray (len(u_array), m-1)
        """

        eig_vals, eig_vecs = self.All_Eigvals_and_Eigvecs

        # deal with degenerate ground states
        if self.is_degenerate():
            eig_vec0 = np.sum(eig_vecs[:, :, :2], axis=2) / np.sqrt(2)
            return np.einsum("ijk, ji->ik", eig_vecs[:, :, 2:], Op @ eig_vec0.T)

        # deal with non-degenerate ground state
        else:
            eig_vec0 = eig_vecs[:, :, 0]
            return np.einsum("ijk, ji->ik", eig_vecs[:, :, 1:], Op @ eig_vec0.T)

    def Imag_Lorentzian(self, U_idx: int):
        """
        Imaginary part of the Lorentzian function for visualizing the poles of the Green's function for a specific value of the on-site interaction U.

        Parameters
        ----------
        U_idx : int
            u index of `E_n_bar` for which to calculate Lorentzian that corresponds to U.

        Returns
        -------
        Imag_Lorentzian : ndarray (len(omega_array), m-1)
        """
        _d = self.delta.value
        _w = self.omega_array
        _E = self.E_n_bar[U_idx, :]

        return _d / ((_w[:, np.newaxis] - _E)**2 + _d**2) / 2.

    def Real_Lorentzian(self, U_idx: int):
        """
        Real part of the Lorentzian function for visualizing the poles of the Green's function for a specific value of the on-site interaction U.

        Parameters
        ----------
        U_idx : int
            u index of `E_n_bar` for which to calculate Lorentzian, that corresponds to U.

        Returns
        -------
        Real_Lorentzian : ndarray (len(omega_array), m-1)
        """
        _d = self.delta.value
        _w = self.omega_array
        _E = self.E_n_bar[U_idx, :]

        return (_E - _w[:, np.newaxis]) / ((_w[:, np.newaxis] - _E)**2 + _d**2)

    def find_index_matching_u25_value(self, array):
        """
        Return the index of `array` that matches the value of the `u25` slider.

        Parameters
        ----------
        array : ndarray
            Array of values to search through.

        Returns
        -------
        index : int
            Index of `array` that matches the value of the `u25` slider.
        """
        _u25 = self.u25.value

        idx = np.flatnonzero((array >= _u25) & (array <= _u25))
        assert len(idx) > 0, f"U={_u25} value not found in array {array}"
        assert len(
            idx) < 2, f"U={_u25} values found multiple times in array {array}"

        return idx[0]

    def Greens_Function(self, Lorentzian: callable, idx: int, A, B=1.):
        """
        Calculate the Green's function G_AB(w) = sum_n <g|A|n><n|B|g> L(w) for a given `Lorentzian` function L and matrix element coefficients for Operators A and B.

        Parameters
        ----------
        Lorentzian : function
            Lorentzian function to use for calculating Green's function.
            Either `Imag_Lorentzian` or `Real_Lorentzian`
        idx : int
            u index of `E_n_bar` for which to calculate Green's function.
        A : ndarray (m-1)
            Matrix element coefficients for Operator A.
        B : ndarray (m-1), optional
            Matrix element coefficients for Operator B.

        Returns
        -------
        G : ndarray (len(omega_array),)
        """
        return np.sum(A * B * Lorentzian(idx), axis=1)

    @Cach
    def E_n_bar(self):
        """
        Calculates the shifted eigenvalues of H, reduced by the ground state eigenvalue.
        i.e. E_n_bar = E_n - E_0 for all eigenvalues > E_0 and all U in u_array.

        Returns
        -------
        E_n_bar : ndarray (len(u_array), m-1)
        """
        eig_vals, eig_vecs = self.All_Eigvals_and_Eigvecs

        # deal with degenerate ground states
        if self.is_degenerate():
            return eig_vals[:, 2:] - eig_vals[:, 0][:, np.newaxis]

        # deal with non-degenerate ground state
        else:
            return eig_vals[:, 1:] - eig_vals[:, 0][:, np.newaxis]

    def Mel_Sz(self, i: int):
        """
        Convenience function to calculate the matrix elements of the operator Sz_i as a function for all the on-site interaction values of U, with the ground state wavefunction, i.e. <psi_n| Sz_i |psi_g>|, for all m eigenvalues of H(U).

        Parameters
        ----------
        i : int
            site index of Sz_i

        Returns
        -------
        <n|Sz_i|g> : ndarray (len(u_array), m-1)
        """
        return self.Calc_GS_Overlap_Elements(self.Op_Sz(i))

    @Cach
    def Mel_Sz_0(self):
        """
        Convenience function to calculate the matrix elements of the operator Sz_0 as a function for all the on-site interaction values of U, with the ground state wavefunction, i.e. <psi_n| Sz_0 |psi_g>|, for all m eigenvalues of H(U).
        Also stores the result in the cache.

        Returns
        -------
        <n|Sz_0|g> : ndarray (len(u_array), m-1)
        """
        return self.Calc_GS_Overlap_Elements(self.Op_Sz(0))

    @Cach
    def Mel_SzSz(self) -> list[np.ndarray]:
        """
        Calculates and stores the matrix elements of the operator Sz_i Sz_j for all relevant pairs of i and j, as a function for all the on-site interaction values of U, with the ground state wavefunction, i.e. <psi_n| Sz_i Sz_j |psi_g>|, for all m eigenvalues of H(U).

        Returns
        -------
        <n|Sz_i Sz_j|g> : list[ndarray (len(u_array), m-1)]
        """
        return [self.Mel_Sz_0 * self.Mel_Sz(i) for i in np.arange(self.n.value // 2 + 1)]

    def Plot_G_SzSz(self, Lorentzian: str, **kwargs):
        """
        Method to plot the spin-spin correlation G_SzSz of the operator `Sz_i Sz_j` for u in [u_min, u_max] and t=1, for all relevant combinations of i and j.

        Returns
        -------
        fig : matplotlib.figure.Figure
            figure object to save as image-file

        Parameters
        ----------
        Lorentzian : str
            which Lorentzian function to use for the calculation of the Greens function, either "Imag" or "Real"

        Other Parameters
        ----------------
        **Kwargs : Widgets
            used to add sliders and other widgets to the displayed output
        """
        _s_up = self.s_up.value
        _s_down = self.s_down.value
        _n = self.n.value
        _u25 = self.u25.value
        _d = self.delta.value

        w_idx, w = self.find_indices_of_slider(
            self.omega_array, self.omega_range)
        u_idx = self.find_index_matching_u25_value(self.u_array)

        G_Szi_Szj = [S[u_idx, :] for S in self.Mel_SzSz]
        if Lorentzian == "Imaginary":
            Lorentzian_func = self.Imag_Lorentzian
            L_str = "Im"
        elif Lorentzian == "Real":
            Lorentzian_func = self.Real_Lorentzian
            L_str = "Re"
        G_Szi_Szj = [self.Greens_Function(Lorentzian_func, u_idx, S)[
            w_idx] for S in G_Szi_Szj]

        G_SzSz_str = f"{L_str} " r"$\left\langle S_{iz} S_{jz} \right\rangle_\omega $"

        color = mpl.cm.tab10(np.arange(0, 10))
        mpl.pyplot.rcParams["axes.prop_cycle"] = cycler("color", color)
        # fig = plt.figure(figsize=(10, 6))#
        fig, axs = plt.subplots(2, 2, figsize=(
            10, 6))

        title = fill(
            f"{Lorentzian}" r" part of Green's function $\langle$$S_{iz}$$S_{jz}$$\rangle_\omega$ " f"for on-site interaction $U = {_u25}$, $\delta = {_d:.2f}$, $n = {_n}$ sites with {_s_up} spin up electron(s), {_s_down} spin down electron(s) and hopping amplitude $t = 1$", width=80)
        fig.set_tight_layout(True)
        fig.suptitle(title)
        # plt.title(rf"{title}")
        # plt.xlabel(r"$\omega$")
        # plt.ylabel(G_SzSz_str)
        # plt.grid(which="both", axis="both", linestyle="--",
        #          color="black", alpha=0.4)

        for i, ax in enumerate(axs.flatten()):
            if i >= _n // 2 + 1:
                fig.delaxes(ax)
            else:
                ax.plot(w, G_Szi_Szj[i],
                        label=r"$\langle S_{1z}S_{"f"{i+1}"r"z}\rangle_\omega$", color=color[i])

                # ax.set_xlabel(r"$\omega$")
                # ax.set_ylabel(G_SzSz_str)
                ax.grid(which="both", axis="both",
                        linestyle="--", color="black", alpha=0.4)
                ax.legend(loc="best")
            if i % 2 == 0:
                ax.set_ylabel(G_SzSz_str)
            if i >= _n // 2 - 1:
                ax.set_xlabel(r"$\omega$")
        # plt.legend(bbox_to_anchor=(1.0, 1), loc="upper left", ncol=1)
        return fig

    def Mel_Szk(self, k: int):
        """
        Convenience function to calculate the matrix elements of the operator Szk_i as a function for all the on-site interaction values of U, with the ground state wavefunction, i.e. <psi_n| Szk_i |psi_g>|, for all m eigenvalues of H(U).

        Parameters
        ----------
        k : int
            k index of Szk

        Returns
        -------
        <n|Szk_i|g> : ndarray (len(u_array), m-1)
        """
        return self.Calc_GS_Overlap_Elements(self.Op_Szk(k))

    @Cach
    def Mel_Szk_list(self) -> list[np.ndarray]:
        """
        Convenience function to calculate the matrix elements of the operators Szk as a function for all the on-site interaction values of U, with the ground state wavefunction, i.e. <psi_n| Szk_0 |psi_g>|, for all m eigenvalues of H(U).
        Also stores the result in the cache.

        Returns
        -------
        <n|Szk_i|g> : list[ndarray (len(u_array), m-1)]
        """
        _n = self.n.value

        return [self.Mel_Szk(k) for k in np.arange(_n)]

    @Cach
    def Mel_SzkSzk(self) -> list[np.ndarray]:
        """
        Calculates and stores the matrix elements of the operator Szk_i Sz_j for all relevant pairs of i and j, as a function for all the on-site interaction values of U, with the ground state wavefunction, i.e. <psi_n| Szk_i Szk_j |psi_g>|, for all m eigenvalues of H(U).

        Returns
        -------
        <n|Szk_i Szk_j|g> : list[ndarray (len(u_array), m-1)]
        """
        _n = self.n.value
        # number_of_unique_correlations
        _l = int(np.floor(_n / 2) + 1)

        return [self.Mel_Szk_list[k] * self.Mel_Szk_list[(_n-k) % _n] for k in np.arange(_l)]

    def Plot_G_SzkSzk(self, **kwargs) -> plt.figure:
        """
        Method to plot the spin-spin correlation G_SzkSzk of the operator `Sz_i Sz_j` for u in [u_min, u_max] and t=1, for all relevant combinations of i and j.

        Returns
        -------
        fig : matplotlib.figure.Figure
            figure object to save as image-file

        Other Parameters
        ----------------
        **Kwargs : Widgets
            used to add sliders and other widgets to the displayed output
        """
        _s_up = self.s_up.value
        _s_down = self.s_down.value
        _n = self.n.value
        _u25 = self.u25.value
        _d = self.delta.value

        w_idx, w = self.find_indices_of_slider(
            self.omega_array, self.omega_range)
        u_idx = self.find_index_matching_u25_value(self.u_array)

        G_Szki_Szkj = [S[u_idx, :] for S in self.Mel_SzkSzk]
        ImG_Szki_Szkj = [np.round(self.Greens_Function(self.Imag_Lorentzian, u_idx, S), 5)[
            w_idx] for S in G_Szki_Szkj]
        ReG_Szki_Szkj = [np.round(self.Greens_Function(self.Real_Lorentzian, u_idx, S), 5)[
            w_idx] for S in G_Szki_Szkj]

        G_SzkSzk_str = r"$\left\langle S_z(k) S_z(-k) \right\rangle_\omega $"

        color = mpl.cm.tab10(np.arange(0, 10))
        mpl.pyplot.rcParams["axes.prop_cycle"] = cycler("color", color)
        fig, axs = plt.subplots(2, 2, figsize=(
            10, 6))

        title = fill(
            r"Green's function $\langle$$S_z(k)$$S_z(-k)$$\rangle_\omega$ in $k$-space " f"for on-site interaction $U = {_u25}$, $\delta = {_d:.2f}$, $n = {_n}$ sites with {_s_up} spin up electron(s), {_s_down} spin down electron(s) and hopping amplitude $t = 1$", width=80)
        fig.set_tight_layout(True)
        fig.suptitle(title)

        # number_of_unique_correlations
        _l = np.floor(_n / 2) + 1
        labels = [str(Fraction(2 / _n * i).limit_denominator(100))
                  for i in np.arange(_l)]

        for i, ax in enumerate(axs.flatten()):
            if i >= _l:
                fig.delaxes(ax)
            else:
                ax.plot(w, ImG_Szki_Szkj[i],
                        label="Im", color=color[i])
                ax.plot(w, ReG_Szki_Szkj[i], linestyle="--", label="Re",
                        color=color[int(i+_l)], alpha=0.9)
                ax.annotate(rf"$k = {{{labels[i]}}} \cdot \pi $", xy=(
                    0.01, 0.89), xycoords="axes fraction", color="b",)

                ax.grid(which="both", axis="both",
                        linestyle="--", color="black", alpha=0.4)
                ax.legend(loc="upper right")
            if i % 2 == 0:
                ax.set_ylabel(G_SzkSzk_str)
            if i >= _l - 2:
                ax.set_xlabel(r"$\omega$")
        return fig


class DynamicalHubbardPropagator(DynamicalHubbard):

    def __init__(self, n=None, s_up=None, s_down=None) -> None:
        super().__init__(n, s_up, s_down)  # call parent initilizer

        if self.s_up.value < self.n.value:
            self.hubbard_plus = Hubbard_NEW(
                n=self.n.value, s_up=self.s_up.value + 1, s_down=self.s_down.value, u_array=self.u_array)
        if self.s_up.value > 0:
            self.hubbard_minus = Hubbard_NEW(
                n=self.n.value, s_up=self.s_up.value - 1, s_down=self.s_down.value, u_array=self.u_array)

        self.n.observe(self.on_change_n2, names="value")
        self.s_up.observe(self.on_change_s_up2, names="value")
        self.s_down.observe(self.on_change_s_down2, names="value")

        self.site_i = site_i_Slider
        self.site_j = site_j_Slider
        self.mu = mu_Slider

    def check_basis_creation(self) -> None:
        if self.s_up.value < self.n.value:
            self.hubbard_plus = Hubbard_NEW(
                n=self.n.value, s_up=self.s_up.value + 1, s_down=self.s_down.value, u_array=self.u_array)
        else:
            self.hubbard_plus = None

        if self.s_up.value > 0:
            self.hubbard_minus = Hubbard_NEW(
                n=self.n.value, s_up=self.s_up.value - 1, s_down=self.s_down.value, u_array=self.u_array)

        else:
            self.hubbard_minus = None

    def on_change_n2(self, change) -> None:
        self.check_basis_creation()

    def on_change_s_up2(self, change) -> None:
        self.check_basis_creation()

    def on_change_s_down2(self, change) -> None:
        self.check_basis_creation()

    @staticmethod
    @njit(fastmath=True, parallel=True)
    def Sign_Matrix_C(A, B, site) -> np.ndarray:
        """
        Calculates the sign matrix for an annihilation process from site `site` from all possible states in A to all possible states in B.

        Parameters
        ----------
        A: ndarray(m, 2n)
            Basis to hop into
        B: ndarray(m, 2n)
            Basis to hop out of
        site: int
            index of site to annihilate particle

        Returns
        -------
        C: ndarray(m, m)
            Sign matrix of the hopping
        """
        assert A.shape[1] == B.shape[1]
        C1 = np.zeros((A.shape[0], B.shape[0]), A.dtype)

        for i in nb.prange(A.shape[0]):
            for j in range(B.shape[0]):
                if is_single_creation_annihilation_process(A[i, :], B[j, :], site):
                    C1[i, j] = hop_sign(A[i, :], site)
                    # C1[i, j] = hop_sign(B[j, :], site)
        return C1

    @jit(forceobj=True)
    def Op_C(self, site) -> np.ndarray:
        """
        Calculates the matrix representation of the creation operator c_i.
        First all allowed creation transitions are calculated, then the total sign matrix `NN_sign` is calculated for each such process. The resulting operator c_i is then the product of `NN_sign` and `NN_creations`.
        """
        bra = self.basis
        ket = self.hubbard_minus.basis
        NN_creations = sp.cdist(bra, ket, metric="cityblock")
        NN_creations = np.where(NN_creations == 1, 1, 0)

        NN_sign = self.Sign_Matrix_C(bra, ket, site)

        return (NN_sign * NN_creations).T

    @Cach
    def Op_C_list(self) -> list[np.ndarray]:
        """
        Calculates and stores the matrix elements of the operator c_i for all relevant sites i.
        """
        return [self.Op_C(i) for i in np.arange(self.n.value)]

    @jit(forceobj=True)
    def Op_C_dagger(self, site) -> np.ndarray:
        """
        Calculates the matrix representation of the creation operator c_i.
        First all allowed creation transitions are calculated, then the total sign matrix `NN_sign` is calculated for each such process. The resulting operator c_i is then the product of `NN_sign` and `NN_creations`.
        """
        bra = self.basis
        ket = self.hubbard_plus.basis
        NN_creations = sp.cdist(bra, ket, metric="cityblock")
        NN_creations = np.where(NN_creations == 1, 1, 0)

        NN_sign = self.Sign_Matrix_C(bra, ket, site)

        return (NN_sign * NN_creations).T

    @Cach
    def Op_C_dagger_list(self) -> list[np.ndarray]:
        """
        Calculates and stores the matrix elements of the operator c_i for all relevant sites i.
        """
        return [self.Op_C_dagger(i) for i in np.arange(self.n.value)]

    @jit(forceobj=True)
    def Op_Ck(self, k: int) -> np.ndarray:
        """
        Calculates the matrix representation of the reciprocal space creation operator c_k.

        Parameters
        ----------
        k: int
            k-value at which to calculate the operator

        Returns
        -------
        Op_Ck: ndarray(m, m)
            Matrix representation of the operator c_k
        """
        _l = self.k_list()**k
        _n = self.n.value

        return np.sum([self.Op_C_list[i] * _l[i] for i in np.arange(_n)], axis=0) / np.sqrt(_n)

    @jit(forceobj=True)
    def Op_Ck_dagger(self, k: int) -> np.ndarray:
        """
        Calculates the matrix representation of the reciprocal space creation operator c_k^dagger.

        Parameters
        ----------
        k: int
            k-value at which to calculate the operator

        Returns
        -------
        Op_Ck^dagger: ndarray(m, m)
            Matrix representation of the operator c_k^dagger
        """
        _l = self.k_list()**(-1*k)
        _n = self.n.value

        return np.sum([self.Op_C_dagger_list[i] * _l[i] for i in np.arange(_n)], axis=0) / np.sqrt(_n)

    @Cach
    def Op_Ck_list(self) -> list[np.ndarray]:
        """
        Calculates and stores the matrix elements of the operator c_k for all relevant sites k.
        """
        return [self.Op_Ck(k) for k in np.arange(self.n.value)]

    @Cach
    def Op_Ck_dagger_list(self) -> list[np.ndarray]:
        """
        Calculates and stores the matrix elements of the operator c_k^dagger for all relevant sites k.
        """
        return [self.Op_Ck_dagger(k) for k in np.arange(self.n.value)]

    def Calc_GS_Overlap_Elements_PM(self, Op: np.ndarray, sector: Sector) -> np.ndarray:
        """
        Calculate the two dimensional matrix element array mel[U, i] = <psi_i| Op |psi_g>| of the system for all m eigenvectors of H(U), where the ground state is in the sector n and the states <psi_i| are in the sector `sector`.

        Parameters
        ----------
        Op : ndarray (p, q)
            Operator to calculate the matrix elements of
        sector : Sector
            Sector of the states <psi_i| to calculate the matrix elements of
            Either `PLUS` or `MINUS`

        Returns
        -------
        Coupling : ndarray (len(u_array), m-1)
        """

        _, eig_vecs_n = self.All_Eigvals_and_Eigvecs

        if sector == Sector.MINUS:
            _, eig_vecs_nPM = self.hubbard_minus.All_Eigvals_and_Eigvecs
        elif sector == Sector.PLUS:
            _, eig_vecs_nPM = self.hubbard_plus.All_Eigvals_and_Eigvecs
        else:
            raise ValueError(f"Unknown sector {sector}")

        # deal with degenerate ground states
        if self.is_degenerate():
            eig_vec0 = np.sum(eig_vecs_n[:, :, :2], axis=2) / np.sqrt(2)
        # deal with non-degenerate ground state
        else:
            eig_vec0 = eig_vecs_n[:, :, 0]

        return np.einsum("ijk, ji->ik", eig_vecs_nPM[:, :, :], Op @ eig_vec0.T)

    def Mel_Ckd(self, k: int) -> np.ndarray:
        """
        Convenience function to calculate the matrix elements of the operator Ck_i as a function for all the on-site interaction values of U, with the ground state wavefunction, i.e. <psi_n| Ck_i |psi_g>|, for all m eigenvalues of H(U).

        Parameters
        ----------
        k : int
            k index of Ck

        Returns
        -------
        <n|Ck_i|g> : ndarray (len(u_array), m-1)
        """
        return self.Calc_GS_Overlap_Elements_PM(self.Op_Ck_dagger(k), Sector.PLUS)

    @Cach
    def Mel_CkdCkd(self) -> list[np.ndarray]:
        """
        Calculates and stores the matrix elements of the operator Ck_i Ck_j for all relevant pairs of i and j, as a function for all the on-site interaction values of U, with the ground state wavefunction, i.e. <psi_n| Ck_i Ck_j |psi_g>|, for all m eigenvalues of H(U).

        Returns
        -------
        <n|Ck_i Ck_j|g> : list[ndarray (len(u_array), m-1)]
        """
        _n = self.n.value
        # number_of_unique_correlations
        _l = int(np.floor(_n / 2) + 1)

        return [np.abs(self.Mel_Ckd(k))**2 for k in np.arange(_l)]

    def Mel_Ck(self, k: int) -> np.ndarray:
        """
        Convenience function to calculate the matrix elements of the operator Ck_i as a function for all the on-site interaction values of U, with the ground state wavefunction, i.e. <psi_n| Ck_i |psi_g>|, for all m eigenvalues of H(U).

        Parameters
        ----------
        k : int
            k index of Ck

        Returns
        -------
        <n|Ck_i|g> : ndarray (len(u_array), m-1)
        """
        return self.Calc_GS_Overlap_Elements_PM(self.Op_Ck(k), Sector.MINUS)

    @Cach
    def Mel_CkCk(self) -> list[np.ndarray]:
        """
        Calculates and stores the matrix elements of the operator Ck_i Ck_j for all relevant pairs of i and j, as a function for all the on-site interaction values of U, with the ground state wavefunction, i.e. <psi_n| Ck_i Ck_j |psi_g>|, for all m eigenvalues of H(U).

        Returns
        -------
        <n|Ck_i Ck_j|g> : list[ndarray (len(u_array), m-1)]
        """
        _n = self.n.value
        # number_of_unique_correlations
        _l = int(np.floor(_n / 2) + 1)

        return [np.abs(self.Mel_Ck(k))**2 for k in np.arange(_l)]

    def En_bar_PM(self, sector: Sector) -> np.ndarray:
        """
        Calculates the shifted eigenvalues of H, reduced by the ground state eigenvalue.
        i.e. E_n_bar = E_n - E_0 for all eigenvalues > E_0 and all U in u_array for the Hamiltonian of the sector `sector`.

        Parameters
        ----------
        sector : str
            which sector to calculate the eigenvalues for, either `PLUS` or `MINUS`.
            `PLUS` means that the eigenvalues of the Hamiltonian are calculated for the sector with one additional up electron.
            `MINUS` means that the eigenvalues of the Hamiltonian are calculated for the sector with one less up electron.

        Returns
        -------
        En_bar : ndarray (len(u_array), m-1)
        """
        eig_vals_n, _ = self.All_Eigvals_and_Eigvecs
        if sector == Sector.PLUS:
            eig_vals, _ = self.hubbard_plus.All_Eigvals_and_Eigvecs
        elif sector == Sector.MINUS:
            eig_vals, _ = self.hubbard_minus.All_Eigvals_and_Eigvecs
        else:
            raise ValueError("sector must be either PLUS or MINUS")

        return eig_vals[:, :] - eig_vals_n[:, 0][:, np.newaxis]

    @Cach
    def E_n_bar_p(self) -> np.ndarray:
        """
        Convenience function to calculate and store the shifted eigenvalues of H in the sector with one electron more, reduced by the ground state eigenvalue.

        Returns
        -------
        E_n_bar : ndarray (len(u_array), m-1)
        """
        return self.En_bar_PM(Sector.PLUS)

    @Cach
    def E_n_bar_m(self) -> np.ndarray:
        """
        Convenience function to calculate and store the shifted eigenvalues of H in the sector with one electron less, reduced by the ground state eigenvalue.

        Returns
        -------
        E_n_bar : ndarray (len(u_array), m-1)
        """
        return self.En_bar_PM(Sector.MINUS)

    def Imag_Lorentzian_PM(self, U_idx: int, sector: Sector, mu) -> np.ndarray:
        """
        Imaginary part of the Lorentzian function for visualizing the poles of the Green's function for a specific value of the on-site interaction U for the Hamiltonian of the sector `sector`.

        Parameters
        ----------
        U_idx : int
            u index of `E_n_bar` for which to calculate Lorentzian that corresponds to U.
        sector : Sector
            which sector to calculate the eigenvalues for, either `PLUS` or `MINUS`.
        mu : float
            chemical potential, used to shift the eigenvalues and usually set to U/2.

        Returns
        -------
        Imag_Lorentzian : ndarray (len(omega_positive_array), m-1)
        """
        _d = self.delta.value
        _w = self.omega_array
        if sector == Sector.PLUS:
            _E = self.E_n_bar_p[U_idx, :] - mu
        elif sector == Sector.MINUS:
            _E = self.E_n_bar_m[U_idx, :] + mu
        else:
            raise ValueError("sector must be either PLUS or MINUS")

        return _d / ((_w[:, np.newaxis] - _E)**2 + _d**2) / 2.

    def Real_Lorentzian_PM(self, U_idx: int, sector: Sector, mu) -> np.ndarray:
        """
        Real part of the Lorentzian function for visualizing the poles of the Green's function for a specific value of the on-site interaction U for the Hamiltonian of the sector `sector`.

        Parameters
        ----------
        U_idx : int
            u index of `E_n_bar` for which to calculate Lorentzian, that corresponds to U.
        sector : Sector
            which sector to calculate the eigenvalues for, either `PLUS` or `MINUS`.
        mu : float
            chemical potential, used to shift the eigenvalues and usually set to U/2.

        Returns
        -------
        Real_Lorentzian : ndarray (len(omega_positive_array), m-1)
        """
        _d = self.delta.value
        _w = self.omega_array
        if sector == Sector.PLUS:
            _E = self.E_n_bar_p[U_idx, :] - mu
        elif sector == Sector.MINUS:
            _E = self.E_n_bar_m[U_idx, :] + mu
        else:
            raise ValueError("sector must be either PLUS or MINUS")

        return (_E - _w[:, np.newaxis]) / ((_w[:, np.newaxis] - _E)**2 + _d**2)

    def Mel_CidCjd(self, i: int, j: int) -> np.ndarray:
        """
        Convenience function to calculate the matrix elements of the operator Cd_i Cd_j as a function for all the on-site interaction values of U, with the ground state wavefunction, i.e. <psi_n-1| Cd_i Cd_j |psi_g>|, for all m eigenvalues of H(U).

        Parameters
        ----------
        i : int
            i index of Ci
        j : int
            j index of Cj

        Returns
        -------
        <n|Cd_i Cd_j|g> : ndarray (len(u_array), m-1)
        """
        return self.Calc_GS_Overlap_Elements_PM(self.Op_C_dagger_list[i], Sector.PLUS) * self.Calc_GS_Overlap_Elements_PM(self.Op_C_dagger_list[j], Sector.PLUS)

    def Mel_CiCj(self, i: int, j: int) -> np.ndarray:
        """
        Convenience function to calculate the matrix elements of the operator C_i C_j as a function for all the on-site interaction values of U, with the ground state wavefunction, i.e. <psi_n-1| C_i C_j |psi_g>|, for all m eigenvalues of H(U).

        Parameters
        ----------
        i : int
            i index of Ci
        j : int
            j index of Cj

        Returns
        -------
        <n|Cd_i Cd_j|g> : ndarray (len(u_array), m-1)
        """
        return self.Calc_GS_Overlap_Elements_PM(self.Op_C_list[i], Sector.MINUS) * self.Calc_GS_Overlap_Elements_PM(self.Op_C_list[j], Sector.MINUS)

    @Cach
    def Mel_CiCj_list(self) -> list[list[np.ndarray]]:
        """
        Calculates and stores the matrix elements of the operator C_i C_j for all relevant pairs of i and j, as a function for all the on-site interaction values of U, with the ground state wavefunction, i.e. <psi_n-1| C_i C_j |psi_g>|, for all m eigenvalues of H(U).

        Returns
        -------
        <n|C_i C_j|g> : list[list[ndarray (len(u_array), m-1)]]
        """
        _n = self.n.value

        return [[self.Mel_CiCj(i, j) for j in np.arange(_n)] for i in np.arange(_n)]

    @Cach
    def Mel_CidCjd_list(self) -> list[list[np.ndarray]]:
        """
        Calculates and stores the matrix elements of the operator Cd_i Cd_j for all relevant pairs of i and j, as a function for all the on-site interaction values of U, with the ground state wavefunction, i.e. <psi_n-1| Cd_i Cd_j |psi_g>|, for all m eigenvalues of H(U).

        Returns
        -------
        <n|Cd_i Cd_j|g> : list[list[ndarray (len(u_array), m-1)]]
        """
        _n = self.n.value

        return [[self.Mel_CidCjd(i, j) for j in np.arange(_n)] for i in np.arange(_n)]

    @staticmethod
    def find_negative_indices_of_slider(array: np.ndarray, range_slider) -> tuple[np.ndarray, np.ndarray]:
        """
        Find indices of `array` that are within the negative range of the `range_slider`.

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
        s_max = -range_slider.value[0]
        s_min = -range_slider.value[1]

        s_idx = np.nonzero(np.logical_and(array <= s_max, array >= s_min))
        s_arr = array[s_idx]
        return s_idx, s_arr

    def Plot_A_w(self, **kwargs) -> plt.figure:
        """
        Method to plot the  one-particle spectral funtion A(w, k) in [u_min, u_max] and t=1, for all relevant values of k==k'

        Returns
        -------
        fig : matplotlib.figure.Figure
            figure object to save as image-file

        Other Parameters
        ----------------
        **Kwargs : Widgets
            used to add sliders and other widgets to the displayed output
        """
        _s_up = self.s_up.value
        _s_down = self.s_down.value
        _n = self.n.value
        _u25 = self.u25.value
        _d = self.delta.value
        _mu = self.mu.value
        _site_i = self.site_i.value - 1
        _site_j = self.site_j.value - 1

        w_idx, w = self.find_indices_of_slider(
            self.omega_array, self.omega_range)
        w_negativ_idx, w_negativ = self.find_negative_indices_of_slider(
            self.omega_array, self.omega_range)
        u_idx = self.find_index_matching_u25_value(self.u_array)

        def lor_plus(x) -> np.ndarray:
            return self.Imag_Lorentzian_PM(x, Sector.PLUS, mu=_mu)

        def lor_minus(x):
            return self.Imag_Lorentzian_PM(
                x, Sector.MINUS, mu=_mu)

        if _s_up == 0:
            A_plus = [A[u_idx, :] for A in self.Mel_CkdCkd]

            # zero those matrix elements that are smaller than 1e-6
            A_plus = [np.where(np.abs(A)**2 < 1e-6, 0, A) for A in A_plus]

            ImA_plus = [np.round(self.Greens_Function(lor_plus, u_idx, A), 5)[
                w_idx] for A in A_plus]
            ImA_minus = None

            A_local_plus = self.Mel_CidCjd_list[_site_i][_site_j][u_idx, :]
            ImA_local_plus = np.round(self.Greens_Function(
                lor_plus, u_idx, A_local_plus), 5)[w_idx]
            ImA_local_minus = None

        elif _s_up == _n:
            A_minus = [A[u_idx, :] for A in self.Mel_CkCk]

            # zero those matrix elements that are smaller than 1e-6
            A_minus = [np.where(np.abs(A)**2 < 1e-6, 0, A) for A in A_minus]

            ImA_plus = None
            ImA_minus = [np.round(self.Greens_Function(lor_minus, u_idx, A), 5)[
                w_negativ_idx] for A in A_minus]

            A_local_minus = self.Mel_CiCj_list[_site_i][_site_j][u_idx, :]
            ImA_local_plus = None
            ImA_local_minus = np.round(self.Greens_Function(
                lor_minus, u_idx, A_local_minus), 5)[w_negativ_idx]

        else:
            A_plus = [A[u_idx, :] for A in self.Mel_CkdCkd]
            A_minus = [A[u_idx, :] for A in self.Mel_CkCk]

            # zero those matrix elements that are smaller than 1e-6
            A_plus = [np.where(np.abs(A)**2 < 1e-6, 0, A) for A in A_plus]
            A_minus = [np.where(np.abs(A)**2 < 1e-6, 0, A) for A in A_minus]

            ImA_plus = [np.round(self.Greens_Function(lor_plus, u_idx, A), 5)[
                w_idx] for A in A_plus]
            ImA_minus = [np.round(self.Greens_Function(lor_minus, u_idx, A), 5)[
                w_negativ_idx] for A in A_minus]

            A_local_plus = self.Mel_CidCjd_list[_site_i][_site_j][u_idx, :]
            A_local_minus = self.Mel_CiCj_list[_site_i][_site_j][u_idx, :]
            ImA_local_plus = np.round(self.Greens_Function(
                lor_plus, u_idx, A_local_plus), 5)[w_idx]
            ImA_local_minus = np.round(self.Greens_Function(
                lor_minus, u_idx, A_local_minus), 5)[w_negativ_idx]

        A_str = r"$A(\omega, k)$"

        color = mpl.cm.tab10(np.arange(0, 10))
        mpl.pyplot.rcParams["axes.prop_cycle"] = cycler("color", color)
        if _n >= 6:
            fig = plt.figure(figsize=(10, 9), layout="constrained")
            gs = fig.add_gridspec(3, 2)
        else:
            fig = plt.figure(figsize=(10, 7), layout="constrained")
            gs = fig.add_gridspec(2, 2)

        title = fill(
            r"One-particle spectral function $A(\omega)$ in $k$-space " f"for on-site interaction $U = {_u25}$, $\delta = {_d:.2f}$, $\mu = {_mu:.2f}$, $n = {_n}$ sites with {_s_up} spin up electron(s), {_s_down} spin down electron(s) and hopping amplitude $t = 1$", width=80)
        fig.set_tight_layout(True)
        fig.suptitle(title)

        # print(w, len(w))
        # print(w_negativ, len(w_negativ))

        # number_of_unique_correlations
        _l = int(np.floor(_n / 2) + 1)
        labels = [str(Fraction(2 / _n * i).limit_denominator(100))
                  for i in np.arange(_l)]

        for i in range(_l + 1):
            ax = fig.add_subplot(gs[i])
            if i > _l:
                fig.delaxes(ax)
                # add a subplot for the local spectral function at the end
            elif i == _l:
                if ImA_local_plus is not None:
                    ax.plot(w, ImA_local_plus,
                            label=rf"$c^\dagger_{{{_site_i+1},{_site_i+1}}}$", color="tab:red")
                if ImA_local_minus is not None:
                    ax.plot(w[:len(ImA_local_minus)], ImA_local_minus[::-1], linestyle="--", label=rf"$c_{{{_site_i+1},{_site_i+1}}}$",
                            color="tab:green", alpha=0.9)
                ax.grid(which="both", axis="both",
                        linestyle="--", color="black", alpha=0.4)
                ax.legend(loc="upper right")
                ax.set_xlabel(r"$\omega$")
                ax.set_ylabel("TODO")
                ax.annotate(rf"$i = {_site_i+1}, j= {_site_j+1}$", xy=(
                    0.01, 0.89), xycoords="axes fraction", color="b",)
            else:
                if ImA_plus is not None:
                    ax.plot(w, ImA_plus[i],
                            label=r"$c^\dagger(k)$", color="tab:orange")
                if ImA_minus is not None:
                    ax.plot(w[:len(ImA_minus[i])], ImA_minus[i][::-1], linestyle="--", label=r"$c(k)$",
                            color="tab:blue", alpha=0.9)
                # ax.plot(w, ReG_Szki_Szkj[i], linestyle="--", label="Re",
                # color=color[int(i+_l)], alpha=0.9)
                ax.annotate(rf"$k = {{{labels[i]}}} \cdot \pi $", xy=(
                    0.01, 0.89), xycoords="axes fraction", color="b",)

                ax.grid(which="both", axis="both",
                        linestyle="--", color="black", alpha=0.4)
                ax.legend(loc="upper right")
            if i % 2 == 0:
                ax.set_ylabel(A_str)
            if i >= _l - 1:
                ax.set_xlabel(r"$\omega$")
            if i == _l:
                ax.set_ylabel(r"local $A(\omega, i, j)$")

        return fig


class Hubbard_NEW():

    def __init__(self, n, s_up, s_down, u_array) -> None:

        self.n = n
        self.s_up = s_up
        self.s_down = s_down
        self.u_array = u_array
        self.t_ij = False  # maybe fix later

        self.basis = self.Construct_Basis()
        self.hoppings = self.Allowed_Hoppings_H()

        self.Reset()

    def Construct_Basis(self):
        """
        Return all possible m := nCr(n, s_up) * nCr(n, s_down) basis states for specific values of `n`, `s_up` and `s_down` by permuting one specific up an down state.

        Returns
        -------
        basis : ndarray (m, 2*n)
                                        array of basis states
        """
        _s_up = self.s_up
        _s_down = self.s_down
        _n = self.n

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
        return self.basis[:, : self.n]

    @Cach
    def down(self):
        """
        Return all possible spin down states.

        Returns
        -------
        basis: ndarray(m, n)
                                                                                                                                        array of spin down basis states
        """
        return self.basis[:, self.n:]

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
    def Hu(self):
        """
        Return cached on-site interaction Hamiltonian H_u given exactly by the occupation number operator `nn`.

        Returns
        -------
        Hu: ndarray(m, m)
            on-site Interaction Hamiltonian
        """
        return self.Op_nn

    @jit(forceobj=True)  # otherwise error message
    def Allowed_Hoppings_H(self):
        """
        Calculates all allowed hoppings in the Hamiltonian H from position `r1` to position `r2` for a given `n`.

        Returns
        -------
        hoppings: ndarray(4n, 2)
            Array of allowed hopping pairs
        """
        _n = self.n
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
    @jit(forceobj=True)
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
        _n = self.n
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

    def H(self, u, t):
        """
        Compute the system's total Hamiltonian H = u*Hu + t*Ht for given prefactors `u` and `t`.

        Parameters
        ----------
        u: float
            on-site interaction strength
        t: float
            hopping strength
        u1: float
            nearest-neighbour-site interaction strength
        """
        if self.t_ij == True:
            return t * self.Ht_ij + u * self.Hu
        return t * self.Ht + u * self.Hu

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
        vals = [np.linalg.eigvalsh(self.H(u, 1)) for u in self.u_array]
        return np.array(vals)

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
        _u1 = self.u1
        vals = [np.linalg.eigvalsh(self.H(10, t, _u1)) for t in self.t_array]
        return np.array(vals)

    def is_degenerate(self):
        """
         Method to check if the ground state of the Hamiltonian is degenerate.
         Checks the degeneracy for H(U=5, t=1) because U = 0 is a special case

        Returns
        -------
        bool
                                        True if the Hamiltonian is degenerate, else False
        """
        _H = self.H(5, 1)

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
        _n = self.n
        return np.exp(2 * np.pi * 1j / _n * np.arange(_n))

    @Cach
    def GS(self):
        """
        Calculate the (normalized) ground state eigevector ( or eigenvectors if degenerate) of the Hamiltonian H(u, t) for u in [u_min, u_max] and t = 1. Methods uses sparse matrices to speed up the computation of the eigenvector associtated with the smallest (algebraic) eigenvalue of H(u, t). k is the degeneracy of the ground state, i.e. k = 1 for a non-degenerate ground state.

        Returns
        -------
        eig_vec: ndarray(len(u), m, k)
        """
        _u1 = self.u1
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
        return self.basis[:, i] * self.basis[:, self.n + i]

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
        return self.Op_nn / self.n

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
        _n = self.n
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
        return [self.Exp_Val_0(self.Op_SzSz(0, i)) for i in np.arange(self.n // 2 + 1)]

    @Cach
    def ExpVal_SzkSzk(self):
        """
        Return the cached ground state expectation value of the reciprocal-space spin-spin correlation operator `Sz_k Sz_k'` for all relevant correlation sites (i.e. those that are non-zero, in other words, were k' = -k).

        Return
        - -----
        expval_SzkSzk: list[ndarray(m,)]
        """
        _n = self.n
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

        _n = self.n
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
                               for i in np.arange(self.n)], axis=0)

        return self.Calc_Coupling(Sz_staggered)

    @Cach
    def All_Eigvals_and_Eigvecs(self):
        """
        Method to calculate all m eigenvalues and eigenvectors for all U in [u_min, u_max] and t=1.

        Returns
        - ------
        eigvals, eigvecs: list of numpy.ndarray with shape(len(u_array), m)
        """
        _H = np.array([self.H(u, 1) for u in self.u_array])
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


@jit
def is_single_creation_annihilation_process(x, y, site):
    """
    Check if the state y/x is obtained from the state x/y by creating/annihilating a particle at site `site`.

    Parameters
    ----------
    x: ndarray(n,)
        state
    y: ndarray(n,)
        state
    site: int
        Site at which the particle is created/anihilated

    Returns
    -------
    bool
        True if the state y/x is obtained from the state x/y by creating/annihilating a particle at site `site`, else False
    """
    return int(x[site]) ^ int(y[site])
