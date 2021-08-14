
import numpy as np 
from scipy.sparse import diags # Used for banded matrices

import matplotlib as mpl
import matplotlib.pyplot as plt # Plotting
from cycler import cycler


import seaborn as sns
plt.style.use('seaborn-dark')
plt.rcParams.update({'font.size':14})
#mpl.rcParams['figure.dpi']= 100

from IPython.display import display, Markdown, Latex, clear_output # used for printing Latex and Markdown output in code cells
from ipywidgets import Layout, fixed, HBox, VBox #interact, interactive, interact_manual, FloatSlider, , Label, Layout, Button, VBox
import ipywidgets as widgets

import functools
import time, math

n_Slider = widgets.BoundedIntText(
            min=2,
            max=10,
            step=1,
            value=6,
            layout=Layout(width = "3cm"),# height="80px"),#"auto"),
            description=r'$n$',
            style = {'description_width': 'initial'},
            continuous_update=False
            )

