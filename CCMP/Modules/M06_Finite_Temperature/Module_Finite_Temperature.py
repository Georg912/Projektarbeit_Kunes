from ..M04_Hubbard_Model.Module_Hubbard_Class import Hubbard, is_single_hopping_process, hop_sign
from ..M04_Hubbard_Model.Module_Cache_Decorator import Cach
from ..M05_Dynamical_Correlations.Module_Dynamical_Hubbard import Hubbard_NEW
from ..M06_Finite_Temperature.Module_Widgets import u25_Slider, T_range_Slider, N_Slider, bins_Slider

import matplotlib.pyplot as plt  	# Plotting
import matplotlib as mpl
from cycler import cycler  			# used for color cycles in mpl
from textwrap import fill  			# used for text wrapping in mpl
from fractions import Fraction  	# used for fractions in mpl
import numpy as np  				# for math and array handling

import numba as nb  # used for just in time compilation
from numba import jit, njit  # used for just in time compilation


class FiniteTemperature(Hubbard):

    def __init__(self, n=None, N=None) -> None:
        # call parent constructor
        super().__init__(n=n, s_up=None, s_down=None)

        # initialize all widgets
        self.N = N_Slider
        self.N.observe(self.on_change_N, names='value')

        self.T_range = T_range_Slider
        _num_T = (self.T_range.max -
                  self.T_range.min) / self.T_range.step
        # self.T_array = np.linspace(
        #     0.0001, self.T_range.max, int(_num_T) + 1, endpoint=True)
        self.T_array = np.exp(-0.05 * np.arange(_num_T + 1)
                              )[::-1] * self.T_range.max
        # self.T_range.observe(self.on_change_T, names='value')

        self.u25 = u25_Slider
        _num_u25 = (self.u25.max - self.u25.min) / self.u25.step
        self.u_array = np.linspace(
            self.u25.min, self.u25.max, num=int(_num_u25) + 1, endpoint=True)

        self.bins = bins_Slider

    def on_change_T(self, change):
        """
        This function is called when the temperature range slider is changed and updates the temperature array.
        """
        self.T_array = np.linspace(*self.T_range.value, num=400)

    def on_change_N(self, change):
        """
        This function is called when the filling slider is changed and updates the filling.
        """
        self.Reset()

    @Cach
    def Allowed_Sectors(self) -> np.ndarray:
        """
        Returns an array of all the possible sectors of a Hubbard model with `n` sites and `N` particle filling.

        Returns
        -------
        np.ndarray(n+1-abs(N-n), 2)
            An array of all the possible sectors of a Hubbard model with n sites and N particle filling.
        """
        _N = self.N.value
        _n = self.n.value
        if _N <= _n:
            _l = np.arange(_N+1)
            return np.vstack([_l, _l[::-1]]).T
        else:
            _l = np.arange(_N-_n, _n+1)
            return np.vstack([_l, _l[::-1]]).T

    @Cach
    def All_Hubbard_Sectors(self) -> list[Hubbard_NEW]:
        """
        Returns a list of Hubbard classes all the possible sectors of a Hubbard model with `n` sites and `N` particle filling.

        Returns
        -------
        list[Hubbard_NEW]
            A list of all the possible Hubbar class sectors of a Hubbard model with n sites and N particle filling.
        """
        _N = self.N.value
        _n = self.n.value
        _u_array = self.u_array
        return [Hubbard_NEW(n=_n, s_up=s_up, s_down=s_down, u_array=_u_array) for s_up, s_down in self.Allowed_Sectors]

    @Cach
    def All_Eigen_Energies(self) -> np.ndarray:
        """
        Calculate all possible eigenvalues of the Hubbard model with `n` sites and `N` particle filling. The eigenvalues are shifted such that the lowest eigenvalue is zero.
        """
        _E_list = []

        self.states_start = [0]
        self.states_end = []
        for sector in self.All_Hubbard_Sectors:
            _vals, _vecs = sector.All_Eigvals_and_Eigvecs
            _E_list.append(_vals)
            self.states_start.append(self.states_start[-1] + _vals.shape[1])
        _E = np.concatenate(_E_list, axis=-1)
        _E -= np.min(_E, axis=-1)[:, None]
        self.number_of_states = _E.shape[1]
        return _E

    @Cach
    def Elements_Sz_total(self):
        """
        """
        _Elements = []
        for sector in self.All_Hubbard_Sectors:
            _Elements.append(sector.Elements(sector.Op_Sz_total))
        return np.concatenate(_Elements, axis=-1)

    @Cach
    def Elements_Sz_total2(self):
        """
        """
        _Elements = []
        for sector in self.All_Hubbard_Sectors:
            _Elements.append(sector.Elements(sector.Op_Sz_total2))
        return np.concatenate(_Elements, axis=-1)

    @Cach
    def Elements_SzSz(self) -> list[np.ndarray]:
        """
        """
        _n = self.n.value
        _SzSz = []
        for i in np.arange(_n // 2 + 1):
            _Elements = []
            for sector in self.All_Hubbard_Sectors:
                _Elements.append(sector.Elements(sector.Op_SzSz(0, i)))
            _SzSz.append(np.concatenate(_Elements, axis=-1))
        return _SzSz

    @Cach
    def Z(self):
        """Cach wrapper for the partition function."""
        return self.Partition_Function_Z(self.T_array)

    @Cach
    def F(self):
        """Cach wrapper for the free energy."""
        _T = self.T_array
        return -_T * np.log(self.Z)

    @Cach
    def U(self):
        """Cach wrapper for the internal energy."""
        _E = self.All_Eigen_Energies
        _Z = self.Z
        _T = self.T_array
        return np.sum(_E[:, :, None]*np.exp(-_E[:, :, None]/_T[None, None, :]), axis=1)/_Z

    @Cach
    def S(self):
        """Cach wrapper for the entropy."""
        _Z = self.Z
        _E = self.All_Eigen_Energies
        _T = self.T_array
        return np.log(_Z) + np.sum(_E[:, :, None]*np.exp(-_E[:, :, None]/_T[None, None, :]), axis=1)/_Z / _T

    @Cach
    def Cv(self):
        """Cach wrapper for the specific heat."""
        _E = self.All_Eigen_Energies
        _Z = self.Z
        _U = self.U
        _T = self.T_array

        return np.sum((_E[:, :, None] - _U[:, None, :])**2*np.exp(-_E[:, :, None]/_T[None, None, :]), axis=1)/_Z/_T**2

    def Partition_Function_Z(self, T: np.array):
        """
        Calculate the partition function of the Hubbard model with `n` sites and `N` particle filling.

        Parameters
        ----------
        T : np.array
            The temperature array.

        Returns
        -------
        float
            The partition function.
        """
        _E = self.All_Eigen_Energies
        _Z = np.sum(np.exp(-_E[:, :, None]/T[None, None, :]), axis=1)
        return _Z

    def Free_Energy_F(self, T: np.array):
        """
        Calculate the free energy of the Hubbard model with `n` sites and `N` particle filling.

        Parameters
        ----------
        T : np.array
            The temperature array.

        Returns
        -------
        np.ndarray
            The free energy.
        """
        _Z = self.Partition_Function_Z(T)
        return -T*np.log(_Z)

    def Internal_Energy_U(self, T: np.array):
        """
        Calculate the internal energy of the Hubbard model with `n` sites and `N` particle filling.

        Parameters
        ----------
        T : np.array
            The temperature array.

        Returns
        -------
        np.ndarray
            The internal energy.
        """
        _E = self.All_Eigen_Energies
        _Z = self.Partition_Function_Z(T)
        return np.sum(_E[:, :, None]*np.exp(-_E[:, :, None]/T[None, None, :]), axis=1)/_Z

    def Entropy_S(self, T: np.array):
        """
        Calculate the entropy of the Hubbard model with `n` sites and `N` particle filling.

        Parameters
        ----------
        T : np.array
            The temperature array.

        Returns
        -------
        np.ndarray
            The entropy.
        """
        _Z = self.Partition_Function_Z(T)
        _E = self.All_Eigen_Energies
        return np.log(_Z) + np.sum(_E[:, :, None]*np.exp(-_E[:, :, None]/T[None, None, :]), axis=1)/_Z / T

    def Specific_Heat_Cv(self, T: np.array):
        """
        Calculate the specific heat of the Hubbard model with `n` sites and `N` particle filling.

        Parameters
        ----------
        T : np.array
            The temperature array.

        Returns
        -------
        np.ndarray
            The specific heat.
        """
        _E = self.All_Eigen_Energies
        _Z = self.Partition_Function_Z(T)
        _U = self.Internal_Energy_U(T)

        return np.sum((_E[:, :, None] - _U[:, None, :])**2*np.exp(-_E[:, :, None]/T[None, None, :]), axis=1)/_Z/T**2

    @jit(forceobj=True)
    def ExpVal_T(self, Op: np.ndarray):
        """
        Calculate the expectation value of an operator `Op` of the Hubbard model with `n` sites and `N` particle filling for a given temperature array `T`.

        Parameters
        ----------
        Op : np.ndarray
            The operator
        T : np.array
            The temperature array.

        Returns
        -------
        np.ndarray
            The expectation value of the operator `Op`.
        """
        _Z = self.Z
        _E = self.All_Eigen_Energies
        _T = self.T_array
        return np.sum(Op[:, :, None]*np.exp(-_E[:, :, None]/_T[None, None, :]), axis=1)/_Z

    @Cach
    def ExpVal_SzSz_T(self):
        """ TOdo write docstring """
        return [self.ExpVal_T(SzSz) for SzSz in self.Elements_SzSz]

    @Cach
    def ExpVal_Sz_T(self):
        """ TOdo write docstring """
        return self.ExpVal_T(self.Elements_Sz_total)

    @Cach
    def ExpVal_Sz2_T(self):
        """ TOdo write docstring """
        return self.ExpVal_T(self.Elements_Sz_total2)

    def Plot_Energy_Histogram(self, **kwargs) -> plt.figure:
        """TODO: Docstring for Plot_Energy_Histogram."""
        _n = self.n.value
        _N = self.N.value
        _bins = self.bins.value
        _u25 = self.u25.value
        _u_array = self.u_array

        u_idx = np.argmin(np.abs(_u_array - _u25))

        fig = plt.figure(figsize=(10, 6))

        title = fill(
            f"Histogram with {_bins} bins of all the Eigenenergies $E_\ell$ for a $n = {_n}$ site model with total filling of $N={_N}$ electrons " f"for on-site interaction $U = {_u25}$ and hopping amplitude $t = 1$", width=80)
        plt.title(rf"{title}")
        plt.xlabel(r"Energy $E$")
        plt.ylabel("Number of states")
        plt.grid(which="both", axis="both", linestyle="--",
                 color="black", alpha=0.4)
        plt.tight_layout()
        plt.hist(self.All_Eigen_Energies[
                 u_idx], bins=_bins, label=f"Total = {self.number_of_states}")
        plt.legend()

        return fig

    def Plot_Partition_Function_Z(self, **kwargs) -> plt.figure:
        """TODO: Docstring for Plot_Energy_Histogram."""
        _n = self.n.value
        _N = self.N.value
        _u25 = self.u25.value
        _u_array = self.u_array

        T_idx, T = self.find_indices_of_slider(self.T_array, self.T_range)
        u_idx = np.argmin(np.abs(_u_array - _u25))

        fig = plt.figure(figsize=(10, 6))

        title = fill(
            rf"Canonical Partition Function $Z(T)$ for a $n = {_n}$ site model with total filling of $N={_N}$ electrons " f"for on-site interaction $U = {_u25}$ and hopping amplitude $t = 1$", width=82)
        plt.title(title)
        plt.xlabel(r"Temperature $T$")
        plt.ylabel(r"Partition Function $Z(T)$")
        plt.grid(which="both", axis="both", linestyle="--",
                 color="black", alpha=0.4)
        plt.tight_layout()
        plt.plot(T, self.Z[u_idx][T_idx], ".-")

        return fig

    def Plot_Thermodynamic_Observables(self, **kwargs) -> plt.figure:
        """ TODO: Docstring for Plot_Thermodynamic_Observables."""
        _n = self.n.value
        _N = self.N.value
        _bins = self.bins.value
        _u25 = self.u25.value
        _u_array = self.u_array

        T_idx, T = self.find_indices_of_slider(self.T_array, self.T_range)
        u_idx = np.argmin(np.abs(_u_array - _u25))

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        title = fill(
            rf"Thermodynamic observables for a $n = {_n}$ site model with total filling of $N={_N}$ electrons " f"for on-site interaction $U = {_u25}$ and hopping amplitude $t = 1$", width=80)

        fig.suptitle(title)
        fig.set_tight_layout(True)

        axs[0, 0].set_ylabel(r"Free energy $F$")
        axs[0, 1].set_ylabel(r"Internal energy $U$")
        axs[1, 0].set_ylabel(r"Entropy $S$")
        axs[1, 1].set_ylabel(r"Specific heat $C_V$")
        axs[0, 0].set_xlabel(r"Temperature $T$")
        axs[0, 1].set_xlabel(r"Temperature $T$")
        axs[1, 0].set_xlabel(r"Temperature $T$")
        axs[1, 1].set_xlabel(r"Temperature $T$")
        for ax in axs.flatten():
            ax.grid(which="both", axis="both",
                    linestyle="--", color="black", alpha=0.4)

        color = mpl.cm.tab10(np.arange(0, 10))

        axs[0, 0].plot(T, self.F[u_idx][T_idx], ".-", color=color[0])
        axs[0, 1].plot(T, self.U[u_idx][T_idx], ".-", color=color[1])
        axs[1, 0].plot(T, self.S[u_idx][T_idx], ".-", color=color[2])
        axs[1, 1].plot(T, self.Cv[u_idx][T_idx], ".-", color=color[3])

        return fig

    def Plot_SzSz_T(self, **kwargs) -> plt.figure:
        """ TODO write docstring """
        _n = self.n.value
        _N = self.N.value
        _bins = self.bins.value
        _u25 = self.u25.value
        _u_array = self.u_array

        T_idx, T = self.find_indices_of_slider(self.T_array, self.T_range)
        u_idx = np.argmin(np.abs(_u_array - _u25))

        _SzSz = [S[u_idx][T_idx] for S in self.ExpVal_SzSz_T]

        SzSz_str = r"$\left\langle S_{iz} S_{jz} \right\rangle $"

        color = mpl.cm.tab10(np.arange(1, 10))
        mpl.pyplot.rcParams["axes.prop_cycle"] = cycler("color", color)
        fig = plt.figure(figsize=(10, 6))

        title = fill(
            r"Spin-spin correlation $\langle$$S_{iz}$$S_{jz}$$\rangle$ " f"as a function of temperature $T$ for a $n = {_n}$ site model with total filling of $N={_N}$ electrons " f"for on-site interaction $U = {_u25}$ and hopping amplitude $t = 1$", width=80)
        plt.title(rf"{title}")
        plt.xlabel(r"Temperature $T$")
        plt.ylabel(SzSz_str)
        plt.grid(which="both", axis="both", linestyle="--",
                 color="black", alpha=0.4)

        for i in np.arange(1, _n // 2 + 1):
            plt.plot(T, _SzSz[i], ".-",
                     label=r"$\langle S_{1z}S_{"f"{i+1}"r"z}\rangle$")

        plt.legend(bbox_to_anchor=(1.0, 1), loc="upper left", ncol=1)
        return fig

    def Plot_Sz_Sz2_T_Elements(self, **kwatgs) -> plt.figure:
        _n = self.n.value
        _N = self.N.value
        _u25 = self.u25.value
        _u_array = self.u_array

        u_idx = np.argmin(np.abs(_u_array - _u25))

        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        title = fill(
            rf"Matrix elements of the Spin operator $S_z$ and $S_z^2$ for a $n = {_n}$ site model with total filling of $N={_N}$ electrons ", width=80)

        fig.suptitle(title)
        fig.set_tight_layout(True)

        axs[0].set_ylabel(
            r"Entries of $\left\langle m\vert S_{z} \vert m \right\rangle $")
        axs[1].set_ylabel(
            r"Entries of $\left\langle m\vert S^2_{z} \vert m \right\rangle $")
        axs[0].set_xlabel(r"State $i$")
        axs[1].set_xlabel(r"State $i$")

        for ax in axs.flatten():
            ax.grid(which="both", axis="both",
                    linestyle="--", color="black", alpha=0.4)

        color = mpl.cm.tab10(np.arange(0, 10))

        self.All_Eigen_Energies
        x = np.arange(self.number_of_states)
        for idx, (start, end) in enumerate(zip(self.states_start[:-1], self.states_start[1:])):
            axs[0].plot(x[start:end], self.Elements_Sz_total[u_idx][start:end],
                        ".-", color=color[idx], label=f"{end - start}")
            axs[1].plot(x[start:end], self.Elements_Sz_total2[u_idx][start:end],
                        ".-", color=color[idx], label=f"{end - start}")
        axs[0].legend(bbox_to_anchor=(1.0, 1), loc="upper left",
                      ncol=1, title="# of elements")
        axs[1].legend(bbox_to_anchor=(1.0, 1), loc="upper left",
                      ncol=1, title="# of elements")

        return fig

    def Plot_Sz_Sz2_T(self, **kwatgs) -> plt.figure:
        _n = self.n.value
        _N = self.N.value
        _u25 = self.u25.value
        _u_array = self.u_array

        T_idx, T = self.find_indices_of_slider(self.T_array, self.T_range)
        u_idx = np.argmin(np.abs(_u_array - _u25))

        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        title = fill(
            rf"Expectation value of the Spin operator $S_z$ and $S_z^2$ for a $n = {_n}$ site model with total filling of $N={_N}$ electrons " f"for on-site interaction $U = {_u25}$ and hopping amplitude $t = 1$", width=80)

        fig.suptitle(title)
        fig.set_tight_layout(True)

        axs[0].set_ylabel(r"$\left\langle S_{z} \right\rangle $")
        axs[1].set_ylabel(r"$\left\langle S^2_{z} \right\rangle $")
        axs[0].set_xlabel(r"Temperature $T$")
        axs[1].set_xlabel(r"Temperature $T$")
        for ax in axs.flatten():
            ax.grid(which="both", axis="both",
                    linestyle="--", color="black", alpha=0.4)

        color = mpl.cm.tab10(np.arange(0, 10))

        axs[0].plot(T, np.round(self.ExpVal_Sz_T[u_idx]
                    [T_idx], 6), ".-", color=color[0])
        axs[1].plot(T, self.ExpVal_Sz2_T[u_idx][T_idx], ".-", color=color[1])

        return fig
