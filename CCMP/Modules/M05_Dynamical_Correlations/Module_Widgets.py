# Module for Widgets and Sliders
import ipywidgets as widgets
from ipywidgets import Layout
from ..M04_Hubbard_Model.Module_Widgets import n_Slider

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
    min=-6.,
    max=6.0,
    step=0.005,
    description=r'$\omega=$',
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',
    layout=Layout(width="10cm"),
    style={'description_width': 'initial'},
)
################################
site_i_Slider = widgets.BoundedIntText(
    value=1,
    min=1,
    max=n_Slider.value,
    step=1,
    description=r'$i=$',
    continuous_update=False,
    orientation='horizontal',
    layout=Layout(width="3cm"),
    style={'description_width': 'initial'},
    readout=True,
    readout_format='d',
)
################################
site_j_Slider = widgets.BoundedIntText(
    value=1,
    min=1,
    max=n_Slider.value,
    step=1,
    description=r'$j=$',
    continuous_update=False,
    orientation='horizontal',
    layout=Layout(width="3cm"),
    style={'description_width': 'initial'},
    readout=True,
    readout_format='d',
)
################################
# connect the i and j sliders max value to the one of n
widgets.dlink((n_Slider, 'value'), (site_i_Slider, 'max'))
widgets.dlink((n_Slider, 'value'), (site_j_Slider, 'max'))
################################
mu_Slider = widgets.FloatSlider(
    value=u25_Slider.value/2.,
    min=0,
    max=5,
    step=0.05,
    description=r'$\mu=$',
    continuous_update=False,
    orientation='horizontal',
    style={'description_width': 'initial'},
    readout=True,
    readout_format='.2f',
)
