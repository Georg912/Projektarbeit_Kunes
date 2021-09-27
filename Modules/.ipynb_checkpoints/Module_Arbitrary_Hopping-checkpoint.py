#Module for arbitrary hopping by user definde hopping and transition matrices

import numpy as np
import ipywidgets as widgets
import time, math
from IPython.display import clear_output
import matplotlib.pyplot as plt # Plotting
import matplotlib as mpl
from cycler import cycler #used for color cycles in mpl
from Module_Widgets_and_Sliders import n_Slider

class Hopping:
    
    def __init__(self, n=6):
        #n_Slider.value = n
        self.n = n_Slider
        self.H = np.zeros((n,n))
        self.T = np.eye(n)
        self.button_to_add = widgets.Button(description="A Button")
        self.button_to_add.on_click(self.test_button)
        self.out = widgets.Output()
        self.i = widgets.IntText(description="row", continuous_update=False, value=0)
        self.j = widgets.IntText(description="column", continuous_update=False, value=1)
        self.p = widgets.BoundedFloatText(description="probability", min=0., max=1., step=0.01, value=0.1, continuous_update=False)
        self.button_to_delete = widgets.Button(description="A Button to undo latest operation")
        self.button_to_delete.on_click(self.undo_button)
        self.n.observe(self.on_change_n)
    
    def on_change_n(self, change):
        _n = change.new["value"]
        self.H = np.zeros((_n,_n))
        self.T = np.eye(_n)
        
    
    def Change_n(self):
        pass
        
    def AddHop(self):
        _i = self.i.value
        _j = self.j.value
        _p = self.p.value
        
        self.H[_i, _j] = 1.
        self.H[_j, _i] = 1.
        self.T[_i, _j] = _p
        self.T[_j, _i] = _p
        self.T[_i, _i] -= _p
        self.T[_j, _j] -= _p
        
    def Reset_H_and_T(self):
        _n = self.n.value
        self.H = np.zeros((_n, _n))
        self.T = np.eye(_n)
    
    def set_previous_ij(self):
        self.old_i = self.i.value
        self.old_j = self.j.value
        
    def Undo_Hopping(self):
        self.H[self.old_i, self.old_j] = 0.
        self.T[self.old_i, self.old_j] = 0.
        
    def undo_button(self, but):
        self.Undo_Hopping()
        
        with self.out:
            print("undone")
            #print(self.H)
            #print(self.T)
            time.sleep(2)
            clear_output()
            
    def test_button(self, but):
        self.AddHop()
        self.set_previous_ij()
        with self.out:
            print("done")
            #print(self.H)
            #print(self.T)
            time.sleep(2)
            clear_output()
    
    def Calc_Markov(self, state=[1,0,0,0,0,0], n_its=400):
    ### Check if state is valid, i.e real, of unit length and compatible with `n`.
        _n = self.n.value
        assert len(state) == _n, f"Dimension of the state vector {state} = {len(state)} != n = {_n}."
        assert not any(isinstance(num, complex) for num in state), f"Markovian evolution cannot deal with complex state {state}."
        assert math.isclose(sum(state), 1, rel_tol=1e-04), f"The norm of the state vector {state} = {sum(state)} != 1."
        assert any(math.isclose(num, 1, rel_tol=1e-04) for num in state), f"In the beginning, particle must be at a definite site, e.g. [1, 0, 0, 0, 0, 0]. You have state = {state}."

        ### check if `n_its` is a positive integer
        assert n_its >= 0, "n_its must be greater or equal to 0."
        assert type(n_its) == int, f"n_its must be an integer not {type(_n)}"

        state = np.array(state)
        observations = [state]
        for _ in np.arange(n_its):
            state = self.T @ state
            observations.append(state)
        return np.array(observations)
    
    
    def Plot_Markov(self, state=[1,0,0,0,0,0], n_its=400):
    ### Calculate states
        observations = self.Calc_Markov(state, n_its)
        _n = self.n.value
        fig = plt.figure(figsize=(10,6))

        ### make plot pretty
        plt.title(f"Markov evolution for graph with $n={_n}$ sites, with initial state {state}")
        plt.xlabel(r"Number of iterations $n_{\mathrm{its}}$")
        plt.ylabel(r"Probability of finding particle at site $i$")
        plt.grid()

        ### Ensure color order is consistent with site number
        mpl.rcParams['axes.prop_cycle'] = cycler("color", plt.cm.get_cmap("tab10").reversed().colors[-_n:])
        colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']

        ### actual plotting
        for i, site in enumerate(np.arange(_n)[::-1]):
            plt.plot(observations[:, site], ".-", label=f"Site {site+1}", color=colors[i])
        legend = plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    
        #plt.annotate("hallo", xytext=(0.5,0.2))
        plt.annotate("T = ""\n"f"{self.T}", xy=(1, 1),
                xytext=(1.4, 0.3), textcoords='axes fraction',
                horizontalalignment='left', verticalalignment='top',
                )
        
        eq1 = (r"\begin{eqnarray*}"
       r"|\nabla\phi| &=& 1,\\"
       r"\frac{\partial \phi}{\partial t} + U|\nabla \phi| &=& 0 "
       r"\end{eqnarray*}")
        plt.text(1, 0.9, eq1, color="C2", fontsize=18,
        horizontalalignment="right", verticalalignment="top")

        plt.show()
        return fig