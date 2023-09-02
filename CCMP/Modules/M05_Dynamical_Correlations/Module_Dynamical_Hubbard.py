from ..M04_Hubbard_Model.Module_Hubbard_Class import Hubbard
from ..M04_Hubbard_Model.Module_Cache_Decorator import Cach
from .Module_Widgets import u25_Slider, omega_range_Slider, delta_Slider

import matplotlib.pyplot as plt  	# Plotting
import matplotlib as mpl
from cycler import cycler  			# used for color cycles in mpl
from textwrap import fill  			# used for text wrapping in mpl
from fractions import Fraction  	# used for fractions in mpl
import numpy as np


class DynamicalHubbard(Hubbard):

    def __init__(self, n=6, s_up=3, s_down=3):

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
