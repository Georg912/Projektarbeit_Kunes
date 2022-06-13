# Module for arbitrary hopping by user defined hopping and transition matrices

import numpy as np
from scipy.linalg import expm  # used for matrix exponentiation
import ipywidgets as widgets
import time
import math
import copy
from IPython.display import clear_output
from distutils.spawn import find_executable
import matplotlib.pyplot as plt  # Plotting
import matplotlib as mpl
from cycler import cycler  # used for color cycles in mpl
from ..General.Module_Widgets_and_Sliders import n_Slider, checkbox
from ..General.Module_Widgets_and_Sliders import button_to_add, button_to_undo, button_to_reset, button_to_show
from ..General.Module_Widgets_and_Sliders import i_IntText, j_IntText, p_BoundedFloatText
from ..General.Module_Widgets_and_Sliders import checkbox_periodic_boundary, p1_BoundedFloatText, p2_BoundedFloatText, p3_BoundedFloatText
from scipy.sparse import diags

mpl.rcParams['text.usetex'] = True

# possible TODOS:
# write function to store hopping matrices and load them
# clean class
# document module


class Hopping:

    def __init__(self, n=6):
        #n_Slider.value = n
        self.n = n_Slider
        self.n.observe(self.on_change_n)

        self.Reset_H_and_T()
        self.Add_All_Buttons()
        self.Add_ijp_widgets()

        self.old_Hs = []
        self.old_Ts = []

        self.pbc_checkbox = checkbox_periodic_boundary
        self.pbc_checkbox.observe(self.tick_boundary_checkbox, names="value")
        self.hide()

    def tick_boundary_checkbox(self, change):
        self.hide()
        self.Reset_H_and_T()

    def hide(self):
        if self.pbc_checkbox.value == True:
            self.i.layout.visibility = "hidden"
            self.j.layout.visibility = "hidden"
            self.p.layout.visibility = "hidden"
            self.p1.layout.visibility = "visible"
            self.p2.layout.visibility = "visible"
            self.p3.layout.visibility = "visible"
            self.button_to_undo.layout.visibility = "hidden"
        else:
            self.i.layout.visibility = "visible"
            self.j.layout.visibility = "visible"
            self.p.layout.visibility = "visible"
            self.p1.layout.visibility = "hidden"
            self.p2.layout.visibility = "hidden"
            self.p3.layout.visibility = "hidden"
            self.button_to_undo.layout.visibility = "visible"

    def is_checkbox(self):
        return self.checkbox.value

    def Add_All_Buttons(self):
        self.button_to_add = button_to_add
        self.button_to_undo = button_to_undo
        self.button_to_reset = button_to_reset
        self.button_to_show = button_to_show

        self.button_to_add.on_click(self.click_add)
        self.button_to_undo.on_click(self.click_undo)
        self.button_to_reset.on_click(self.click_reset)
        self.button_to_show.on_click(self.click_show)

        self.out = widgets.Output()
        self.checkbox = checkbox

    def Add_ijp_widgets(self):
        self.i = i_IntText
        self.j = j_IntText
        self.p = p_BoundedFloatText

        self.p1 = p1_BoundedFloatText
        self.p2 = p2_BoundedFloatText
        self.p3 = p3_BoundedFloatText

    def on_change_n(self, change):
        _n = change.new["value"]
        self.H = np.eye(_n)
        self.T = np.eye(_n)

    def Show_H_and_T(self):
        with self.out:
            clear_output()
            print(f"H = ", self.H, "", sep="\n")
            print(f"T = ", self.T, sep="\n")

    # TODO: check if n is large enough for p2 and p3
    def Add_Hop_PBC(self):
        _n = self.n.value
        _p1 = self.p1.value
        _p2 = self.p2.value
        _p3 = self.p3.value

        H1, H2, H3 = np.zeros((_n, _n)), np.zeros((_n, _n)), np.zeros((_n, _n))
        if self.p1.value > 0.:
            diagonal_entries = [np.ones(_n-1), np.ones(_n-1)]
            H1 = diags(diagonal_entries, [-1, 1]).toarray()
            if _n >= 3:
                H1[[0, _n-1], [_n-1, 0]] = 1
        if self.p2.value > 0.:
            diagonal_entries = [np.ones(2), np.ones(
                _n-2), np.ones(_n-2), np.ones(2)]
            if _n >= 5:
                H2 = diags(diagonal_entries, [-_n+2, -2, 2, _n-2]).toarray()
                H2[[0, _n-2], [_n-2, 0]] = 1
        if self.p3.value > 0.:
            if _n >= 7:
                diagonal_entries = [np.ones(3), np.ones(
                    _n-3), np.ones(_n-3), np.ones(3)]
                H3 = diags(diagonal_entries, [-_n+3, -3, 3, _n-3]).toarray()
        assert 2 * (_p1 + _p2 * (_n >= 5) + _p3 * (_n >= 7)) <= 1., self.out.append_stderr(
            f"Error negative probability. For n = {_n}, p0 = 1 - 2(p1{' + p2' if _n>=5 else ''}{' + p3' if _n>=7 else ''}) = 1 - 2 * ({_p1}{' + ' + str(_p2) if _n>=5 else ''}{' + ' +str(_p3) if _n>=7 else ''}) = {1 - 2 * (_p1 + (_p2 if _n>=5 else 0) + (_p3 if _n>=7 else 0)):.3f})")

        _H = H1 + H2 + H3
        _T = (1 - 2 * (_p1 + _p2 * (_n >= 5) + _p3 * (_n >= 7))) * \
            np.eye(_n) + _p1 * H1 + _p2 * H2 + _p3 * H3
        self.H = _H
        self.T = _T
        return
    # take care of the values not on the lower and upper main diagonal
        #H[[0, n-1], [n-1, 0]] = 1

    def AddHop(self):
        if self.pbc_checkbox.value == True:
            self.Add_Hop_PBC()
            return None
        with self.out:
            clear_output()
        _i = self.i.value - 1
        _j = self.j.value - 1
        _p = self.p.value
        _n_max = self.n.max

        if _i != _j:
            assert self.T[_i, _i] - _p >= 0, self.out.append_stderr(
                f"Error hopping would have caused T[{_i},{_i}] <=0.")

        self.H[_i, _j] = 1.
        self.H[_j, _i] = 1.
        self.T[_i, _j] += _p
        self.T[_j, _i] += _p
        self.T[_i, _i] -= _p
        self.T[_j, _j] -= _p

        # add to Ts and Hs after second hopping
        self.old_Hs.append(copy.deepcopy(self.H))
        self.old_Ts.append(copy.deepcopy(self.T))

    def Reset_H_and_T(self):
        _n = self.n.value
        self.H = np.eye(_n)
        self.T = np.eye(_n)

    # realy memory inefficent, work in progress
    def Undo_Hopping(self):
        if len(self.old_Hs) > 1:
            self.H = self.old_Hs[-2]
            self.T = self.old_Ts[-2]
            self.old_Hs.pop()
            self.old_Ts.pop()
        else:
            self.Reset_H_and_T()

    def click_undo(self, b):
        self.Undo_Hopping()

        if self.is_checkbox():
            with self.out:
                print("undone")
                time.sleep(2)
                clear_output()

    def click_add(self, b):
        self.AddHop()

        if self.is_checkbox():
            with self.out:
                print("added hopping")
                time.sleep(2)
                clear_output()

    def click_reset(self, b):
        self.Reset_H_and_T()

        if self.is_checkbox():
            with self.out:
                print("H and T reset")
                time.sleep(2)
                clear_output()

    def click_show(self, b):
        self.Show_H_and_T()

    def Calc_Markov(self, state=[1, 0, 0, 0, 0, 0], n_its=400):
        # Check if state is valid, i.e real, of unit length and compatible with `n`.
        _n = self.n.value
        assert len(
            state) == _n, f"Dimension of the state vector {state} = {len(state)} != n = {_n}."
        assert not any(isinstance(num, complex)
                       for num in state), f"Markovian evolution cannot deal with complex state {state}."
        assert math.isclose(sum(
            state), 1, rel_tol=1e-04), f"The norm of the state vector {state} = {sum(state)} != 1."
        assert any(math.isclose(num, 1, rel_tol=1e-04)
                   for num in state), f"In the beginning, particle must be at a definite site, e.g. [1, 0, 0, 0, 0, 0]. You have state = {state}."

        # check if `n_its` is a positive integer
        assert n_its >= 0, "n_its must be greater or equal to 0."
        assert type(n_its) == int, f"n_its must be an integer not {type(_n)}"

        state = np.array(state)
        observations = [state]
        for _ in np.arange(n_its):
            state = self.T @ state
            observations.append(state)
        return np.array(observations)

    def Plot_Markov(self, state=[1, 0, 0, 0, 0, 0], n_its=400):
        # Calculate states
        observations = self.Calc_Markov(state, n_its)
        _n = self.n.value
        fig = plt.figure(figsize=(10, 6))

        # make plot pretty
        plt.title(
            f"Markov evolution for graph with $n={_n}$ sites, with initial state {state}")
        plt.xlabel(r"Number of iterations $n_{\mathrm{its}}$")
        plt.ylabel(r"Probability of finding particle at site $i$")
        plt.grid()

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
                   bbox_to_anchor=(1.05, 1), loc="upper left", ncol=2)

        if self.find_latex():
            matrix = self.Latex_Matrix() % tuple(self.T.flatten())
            plt.annotate(text=f"T = {matrix}", xy=(0, 0), xytext=(
                1.02, 0.0), xycoords="axes fraction", horizontalalignment="left", verticalalignment="bottom")
        else:
            matrix = self.T.round(2)
            plt.annotate(text="T = ""\n"f"{matrix}", xy=(0, 0), xytext=(
                1.02, 0.0), xycoords="axes fraction", horizontalalignment="left", verticalalignment="bottom")

        plt.show()
        return fig

    def Latex_Matrix(self, precision=2):
        _p = precision
        _N = self.n.value

        beginning = r"$ \left( \begin{array}{"
        formatting = _N*r'c'+r'}'
        array_rows = (_N-1)*((_N-1)*rf"%.{_p}f & "+rf"%.{_p}f \\")
        final_row = (_N-1)*rf"%.{_p}f & "+rf"%.{_p}f "
        end = r"\end{array} \right) $"

        matrix = beginning+formatting+array_rows+final_row+end
        return matrix

    def Latex_Matrix2(self, precision=1):
        _p = precision
        _N = self.n.value

        beginning = r"$ \left( \begin{array}{"
        formatting = _N*r'c'+r'}'
        array_rows = (_N-1)*((_N-1)*rf"%.{_p}f \mathrm{{e}}^{{%.{_p}f \mathrm{{i}}}} & " +
                             rf"%.{_p}f \mathrm{{e}}^{{%.{_p}f \mathrm{{i}}}} \\")
        final_row = (_N-1)*rf"%.{_p}f \mathrm{{e}}^{{%.{_p}f \mathrm{{i}}}} & " + \
            rf"%.{_p}f \mathrm{{e}}^{{%.{_p}f \mathrm{{i}}}} "
        end = r"\end{array} \right) $"

        matrix = beginning+formatting+array_rows+final_row+end
        return matrix

    def Time_Evolution_Operator(self):
        return expm(-1j * self.T)

    def Calc_QM(self, state=[1, 0, 0, 0, 0, 0], n_its=400):
        """TODO: add doc string"""
        _n = self.n.value
        # Check if state is valid
        # TODO: possibly add optin of automatically normalizing state
        assert math.isclose(np.linalg.norm(
            state), 1, rel_tol=1e-04), f"The norm of the state vector {state} = {np.linalg.norm(state)} != 1"

        # check if `n_its` is a positive integer
        assert n_its >= 0, "n_its must be greater or equal to 0."
        assert type(n_its) == int, f"n_its must be an integer not {type(_n)}"

        U = self.Time_Evolution_Operator()

        state = np.array(state)
        observations = [state]
        for _ in np.arange(n_its):
            state = U @ state
            observations.append(state)
        return np.array(observations)

    def Plot_QM_Evolution(self, state=[1, 0, 0, 0, 0, 0], n_its=400):
        # TODO: write documentation
        observations = self.Calc_QM(state, n_its)
        _n = self.n.value

        fig = plt.figure(figsize=(10, 6))
        plt.title(
            f"QM evolution for graph with $n={_n}$ sites, with initial state {state}")
        plt.xlabel(r"Number of iterations $n_{\mathrm{its}}$")
        plt.ylabel(r"Probability of finding particle at site $i$")
        plt.grid()

        mpl.rcParams['axes.prop_cycle'] = cycler(
            "color", plt.cm.get_cmap("tab10").reversed().colors[-_n:])

        for site in np.arange(len(state))[::-1]:
            plt.plot(np.abs(observations[:, site])
                     ** 2, ".-", label=f"Site {site+1}", )
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles[::-1], labels[::-1],
                   bbox_to_anchor=(1.05, 1), loc="upper left", ncol=2)

        U = self.Time_Evolution_Operator()
        phase = np.angle(U)
        magnitude = np.abs(U)
        vals = tuple(np.dstack((magnitude, phase)).flatten())

        if self.find_latex():
            matrix = self.Latex_Matrix2() % vals
            plt.annotate(text=f"U = {matrix}", xy=(0, 0), xytext=(
                1.02, 0.0), xycoords="axes fraction", horizontalalignment="left", verticalalignment="bottom")
        else:
            matrix = U.round(2)
            plt.annotate(text="U = ""\n"f"{matrix}", xy=(0, 0), xytext=(
                1.02, 0.0), xycoords="axes fraction", horizontalalignment="left", verticalalignment="bottom")

        plt.show()
        return fig

    def find_latex(self):
        return find_executable('latex')
