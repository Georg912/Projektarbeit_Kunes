# Module for Widgets and Sliders
import ipywidgets as widgets
import time  # used for `sleep`
from ipywidgets import Layout

########################################
n_Slider = widgets.BoundedIntText(
    min=2,
    max=10,
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


widgets.dlink((n_Slider, 'value'), (s_up_Slider, 'max'))
widgets.dlink((n_Slider, 'value'), (s_down_Slider, 'max'))
