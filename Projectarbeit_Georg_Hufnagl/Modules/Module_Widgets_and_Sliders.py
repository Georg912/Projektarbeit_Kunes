#Module for Widgets and Sliders
import ipywidgets as widgets
import time # used for `sleep`
from ipywidgets import Layout
import numpy as np
from Module_Markov import check_if_int

###########################################################################################################
Iterations_Slider = widgets.IntSlider(
            min=1,
            max=400,
            step=1,
            value=50,
            layout=Layout(width = "10cm"),# height="80px"),#"auto"),
            description=r"Number of iterations $n_\mathrm{its}$:",
            style = {'description_width': 'initial'},
            continuous_update=True,
            )

###########################################################################################################
p1_Slider = widgets.FloatSlider(
            min=0,
            max=1,
            step=0.01,
            value=0.1,
            layout=Layout(width = "10cm"),# height="80px"),#"auto"),
            description=r'$p_1$',
            style = {'description_width': 'initial'},
            continuous_update=False
            )

###########################################################################################################
p2_Slider = widgets.FloatSlider(
            min=0,
            max=1,
            step=0.01,
            value=0.,
            layout=Layout(width = "10cm"),# height="80px"),#"auto"),
            description=r'$p_2$',
            style = {'description_width': 'initial'},
            continuous_update=False
            )

#TODO: Fix weird error, where states_dict is not recognized as a list of 2
###########################################################################################################
states_dict = {
    2:[[1,0], [0,1], [0.8,0.2], [0.5,0.5]],
    3:[[1,0,0], [0,1,0], [0,0,1], [0.5,0.3,0.2]],
    4:[[1,0,0,0], [0,1,0,0], [0,0,1,0], [0.5,0.3,0.2,0]],
    5:[[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0.5,0.3,0.2,0,0]],
    6:[[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0], [0.5,0.3,0.2,0,0,0]],
    7:[[1,0,0,0,0,0,0], [0,1,0,0,0,0,0], [0,0,1,0,0,0,0], [0.5,0.3,0.2,0,0,0,0]],
    8:[[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0], [0.5,0.3,0.2,0,0,0,0,0]],
    9:[[1,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0], [0.5,0.3,0.2,0,0,0,0,0,0]],
    10:[[1,0,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0,0], [0.5,0.3,0.2,0,0,0,0,0,0,0]]
    }

#TODO: rename blub
###########################################################################################################
Initial_State = widgets.Dropdown(
    equals=np.array_equal, #otherwise "value" checks element wise
    options=[([1,0,0,0,0,0]),
        ('2', [0,1,0,0,0,0]),
        ("3", [0,0,1,0,0,0]),
        ("mix", [0.5,0.2,0.3,0,0,0])
        ],
    value=[1,0,0,0,0,0],#np.array([1,0,0,0,0,0]),
    description='Initial State:',
    style = {'description_width': 'initial'},
    continuous_update=False,
)


###########################################################################################################
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

#TODO: change to be more descriptive
###########################################################################################################
Text_Box = widgets.Text(continuous_update=False,
                 placeholder='Input Initial State',
                 description='User Input:',
                 value=str(Initial_State.value),
                 style = {'description_width': 'initial'},
                )


Save_Figure_Button = widgets.Button(
        layout=Layout(width = "5cm"),
        description="Click to save current figure",
        style = {'description_width': 'initial'}
        )

def set_filename(name):
    text = widgets.Text(continuous_update=False,
                 placeholder='enter filename',
                 description='Filename:',
                 value=name,
                 style = {'description_width': 'initial'},
                )
    return text



###########################################################################################################
#Linking Sliders *****************************************************************************
#TODO: delete print statements in "transform"
def transform(inp):
        #with output:
        #print(inp)
        #print(t.get_state)
        #print(Initial_State.options,  inp, type(inp))
        #print(Initial_State.options[0],  inp, type(inp))
    inp = list(map(check_if_int, inp.strip('][').split(',')))
        #print(Initial_State.value, type(Initial_State.value), inp, type(inp))
        #print("test")

    #Avoid duplicate entries in `Initial_State.options'
    if inp not in Initial_State.options:
        Initial_State.options += tuple([inp])
    Initial_State.value = inp
    return inp

###########################################################################################################
def Get_Options(n):
    Initial_State.options = states_dict[n]
    return Initial_State.value

###########################################################################################################
def Set_Initial_Userinput(n):
        #with output:

        #print(Initial_State.options)
        #print(Initial_State.value)
        #print(str(Initial_State.value))
    Text_Box.value = str(Initial_State.options[0])
    return Text_Box.value

###########################################################################################################
#TODO: rename test function
def test(state):
    Text_Box.value = str(Initial_State.value)
    return Text_Box.value

###########################################################################################################
initial_position_Slider = widgets.Dropdown(
    options=np.arange(1, n_Slider.value+1),
    value=1,
    description='Initial position:',
    style = {'description_width': 'initial'},
    continuous_update=True,
)

###########################################################################################################
seed_Slider = widgets.IntSlider(
            min=1,
            max=400,
            step=1,
            value=42,
            layout=Layout(width = "10cm"),# height="80px"),#"auto"),
            description=r"Random number generator seed:",
            style = {'description_width': 'initial'},
            continuous_update=False,
            )

###########################################################################################################
n_paths_Slider = widgets.IntSlider(
            min=100,
            max=10000,
            step=100,
            value=1000,
            layout=Layout(width = "10cm"),# height="80px"),#"auto"),
            description=r"Number of Paths:",
            style = {'description_width': 'initial'},
            continuous_update=False,
            )


def Set_Initial_Position_Range(n):
    initial_position_Slider.options = np.arange(1, n_Slider.value+1)
    return initial_position_Slider.value


###########################################################################################################
t_Slider = widgets.FloatSlider(
            min=0,
            max=1,
            step=0.01,
            value=0.1,
            layout=Layout(width = "10cm"),# height="80px"),#"auto"),
            description=r'$t$',
            style = {'description_width': 'initial'},
            continuous_update=False
            )



###########################################################################################################
widgets.dlink((Text_Box, 'value'), (Initial_State, 'value'), transform);
widgets.dlink((n_Slider, 'value'), (Initial_State, 'value'), Get_Options);
widgets.dlink((n_Slider, 'value'), (Text_Box, 'value'), Set_Initial_Userinput);
widgets.dlink((Initial_State, 'value'), (Text_Box, 'value'), test);
widgets.dlink((n_Slider, 'value'), (initial_position_Slider, 'value'), Set_Initial_Position_Range);
widgets.link((t_Slider, "value"), (p1_Slider, "value"));
