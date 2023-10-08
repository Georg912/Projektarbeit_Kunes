# Module for Widgets and Sliders
import ipywidgets as widgets
from ipywidgets import Layout
from ..M04_Hubbard_Model.Module_Widgets import n_Slider

#################################
u25_Slider = widgets.FloatSlider(
    value=1,
    min=0,
    max=20,
    step=0.25,
    description=r'$U=$',
    continuous_update=False,
    orientation='horizontal',
    layout=Layout(width="10cm"),
    style={'description_width': 'initial'},
    readout=True,
    readout_format='.2f',
)

################################
T_range_Slider = widgets.FloatRangeSlider(
    value=[0, 5],
    min=0.0,
    max=20.0,
    step=0.1,
    description=r'$T=$',
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.2f',
    layout=Layout(width="10cm"),
    style={'description_width': 'initial'},
)

################################
N_Slider = widgets.BoundedIntText(
    value=n_Slider.value,
    min=0,
    max=2*n_Slider.value,
    step=1,
    description=r'$N=$',
    continuous_update=False,
    orientation='horizontal',
    layout=Layout(width="3cm"),
    style={'description_width': 'initial'},
    readout=True,
    readout_format='d',
)
################################
bins_Slider = widgets.IntSlider(
    value=50,
    min=5,
    max=200,
    step=5,
    description='Bins:',
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    layout=Layout(width="10cm"),
    style={'description_width': 'initial'},
    # readout_format='d'
)
################################
# connect n_Slider and N_Slider such that the maximum value of N_Slider is twiche n_Slider.value


def update_N_Slider(state):
    N_Slider.max = 2*n_Slider.value
    return N_Slider.max


widgets.dlink((n_Slider, 'value'), (N_Slider, 'max'), update_N_Slider)
