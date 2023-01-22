from ..M04_Hubbard_Model.Module_Hubbard_Class import Hubbard
from ..M04_Hubbard_Model.Module_Cache_Decorator import Cach
from .Module_Widgets import u25_Slider, omega_range_Slider, delta_Slider

import matplotlib.pyplot as plt  	# Plotting
import matplotlib as mpl
from cycler import cycler  			# used for color cycles in mpl
from textwrap import fill  			# used for text wrapping in mpl
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

    def Imag_Lorentzian(self, idx):
        """
        Imaginary part of the Lorentzian function for visualizing the poles of the Green's function.

        Parameters
        ----------
        idx : int
            u index of `E_n_bar` for which to calculate Lorentzian.

        Returns
        -------
        Imag_Lorentzian : ndarray (len(omega_array), m-1)
        """
        _d = self.delta.value
        _w = self.omega_array
        _E = self.E_n_bar[idx, :]

        return _d / ((_w[:, np.newaxis]**2 - _E**2)**2 + _d**2)

    def Real_Lorentzian(self, idx):
        """
        Real part of the Lorentzian function for visualizing the poles of the Green's function.

        Parameters
        ----------
        idx : int
            u index of `E_n_bar` for which to calculate Lorentzian.

        Returns
        -------
        Real_Lorentzian : ndarray (len(omega_array), m-1)
        """
        _d = self.delta.value
        _w = self.omega_array
        _E = self.E_n_bar[idx, :]

        return (_E - _w[:, np.newaxis]**2) / ((_w[:, np.newaxis]**2**2 - _E**2)**2 + _d**2)

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

    def Greens_Function(self, Lorentzian, idx, A, B=1.):
        """
        Calculate the Green's function G_AB(w) = sum_n <g|A|n><n|B|g> L(w) for a given `Lorentzian` function L and matrix element coefficients for Operators A and B.

        Parameters
        ----------
        Lorentzian : function
            Lorentzian function to use for calculating Green's function.
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
# TODO document code from here on

    def Mel_Sz(self, i):
        return self.Calc_GS_Overlap_Elements(self.Op_Sz(i))

    @Cach
    def Mel_Sz_0(self):
        return self.Calc_GS_Overlap_Elements(self.Op_Sz(0))

    @Cach
    def Mel_SzSz(self):
        return [self.Mel_Sz_0 * self.Mel_Sz(i) for i in np.arange(self.n.value // 2 + 1)]

    def Plot_G_SzSz(self, **kwargs):
        """
        Method to plot the spin-spin correlation G_SzSz of the operator `Sz_i Sz_j` for u in [u_min, u_max] and t=1, for all relevant combinations of i and j.

        Returns
        -------

        fig : matplotlib.figure.Figure
            figure object to save as image-file

        Lorentzian : function
            which Lorentzian function to use for the calculation of the Greens function, either "Imag_Lorentzian" or "Real_Lorentzian"

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
        G_Szi_Szj = [self.Greens_Function(
            self.Real_Lorentzian, u_idx, S)[w_idx] for S in G_Szi_Szj]

        G_SzSz_str = r"$\left\langle S_{iz} S_{jz} \right\rangle_\omega $"

        color = mpl.cm.tab10(np.arange(0, 10))
        mpl.pyplot.rcParams["axes.prop_cycle"] = cycler("color", color)
        fig = plt.figure(figsize=(10, 6))

        title = fill(
            r"Green's function $\langle$$S_{iz}$$S_{jz}$$\rangle_\omega$ " f"for on-site interaction $U = {_u25}$, $\delta = {_d:.2f}$, $n = {_n}$ sites with {_s_up} spin up electron(s), {_s_down} spin down electron(s) and hopping amplitude $t = 1$", width=80)
        plt.title(rf"{title}")
        plt.xlabel(r"$\omega$")
        plt.ylabel(G_SzSz_str)
        plt.grid(which="both", axis="both", linestyle="--",
                 color="black", alpha=0.4)

        for i in np.arange(_n // 2 + 1):
            plt.plot(w, G_Szi_Szj[i],
                     label=r"$\langle S_{1z}S_{"f"{i+1}"r"z}\rangle_\omega$")

        plt.legend(bbox_to_anchor=(1.0, 1), loc="upper left", ncol=1)
        return fig