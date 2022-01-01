# Module for Widgets and Sliders
import ipywidgets as widgets
import time  # used for `sleep`
from ipywidgets import Layout

########################################
n_Slider = widgets.BoundedIntText(
    min=2,
    max=7,
    step=1,
    value=6,
    layout=Layout(width="3cm"),  # height="80px"),#"auto"),
    description=r'$n = $',
    style={'description_width': 'initial'},
    continuous_update=False
)
#####################################
s_up_Slider = widgets.BoundedIntText(
    min=0,
    max=n_Slider.value,
    step=1,
    value=1,
    layout=Layout(width="5cm"),  # height="80px"),#"auto"),
    description=r'$s_\mathrm{up} = $',
    style={'description_width': 'initial'},
    continuous_update=False
)

#####################################
s_down_Slider = widgets.BoundedIntText(
    min=0,
    max=n_Slider.value,
    step=1,
    value=1,
    layout=Layout(width="5cm"),  # height="80px"),#"auto"),
    description=r'$s_\mathrm{down} = $',
    style={'description_width': 'initial'},
    continuous_update=False
)
################################
u_Slider = widgets.FloatRangeSlider(
    value=[2, 7],
    min=0,
    max=40.0,
    step=0.1,
    description='todo:',
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',
)
################################
steps_Slider = widgets.IntSlider(
    value=10,
    min=1,
    max=40,
    step=1,
    description='Plotting steps:',
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    style={'description_width': 'initial'},
    # readout_format='d'
)
################################
t_Slider = widgets.FloatRangeSlider(
    value=[0, 1],
    min=0,
    max=5,
    step=0.1,
    description='todo:',
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',
)
################################

widgets.dlink((n_Slider, 'value'), (s_up_Slider, 'max'))
widgets.dlink((n_Slider, 'value'), (s_down_Slider, 'max'))