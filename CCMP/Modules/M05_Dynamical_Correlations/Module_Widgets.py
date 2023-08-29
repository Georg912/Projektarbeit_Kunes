# Module for Widgets and Sliders
import ipywidgets as widgets
from ipywidgets import Layout

#################################
u25_Slider = widgets.FloatSlider(
    value=1,
    min=0,
    max=5,
    step=0.25,
    description=r'$U=$',
    continuous_update=False,
    orientation='horizontal',
    style={'description_width': 'initial'},
    readout=True,
    readout_format='.2f',
)

#################################
delta_Slider = widgets.FloatSlider(
    value=0.02,
    min=0.0001,
    max=0.5,
    step=0.01,
    description=r'$\delta=$',
    continuous_update=False,
    orientation='horizontal',
    style={'description_width': 'initial'},
    readout=True,
    readout_format='.2f',
)

################################
omega_range_Slider = widgets.FloatRangeSlider(
    value=[0.5, 5],
    min=-5.,
    max=8.0,
    step=0.005,
    description=r'$\omega=$',
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',
    layout=Layout(width="10cm"),
    style={'description_width': 'initial'},
)
