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
            description=r'$n = $',
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



phase_Dropdown = widgets.Dropdown(
    equals=np.array_equal, #otherwise "value" checks element wise
    options=[('[1, 2, 3, 4, 5, 6]', [1, 2, 3, 4, 5, 6]),
             ('[0, 2, 4, 6, 8, 10]', [0, 2, 4, 6, 8, 10]),
             ('[-1.0, 5.0, -2.0, 1.5, -0.83, 0.45]', [-1, 5, -2, 1.5, -0.83, 0.45]),
             ('[1.57, -1.57, 1.57, -1.57, 1.57, -1.57]', [np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2])],
    value=[1,2,3,4,5,6],#np.array([1,0,0,0,0,0]),
    description=r'Choose Phase Vector $\theta$:',
    style = {'description_width': 'initial'},
    continuous_update=False,
    layout=Layout(width = "15cm")
)


phase_Text_Box = widgets.Text(continuous_update=False,
                placeholder='Input Phase Vector',
                description=r'User Phase Vector $\theta$:',
                value=str(Initial_State.value),
                style = {'description_width': 'initial'},
                layout=Layout(width = "15cm"),
                )



# phase_dict = {
#     2:[[0]*2, [1,2], [0,2], [-1, 0.45], [np.pi/2, -np.pi/2], [1.234, 23.2235235]],
#     3:[[0]*3, [1,2,3], [0,2,4], [-1,5,-0.45], [np.pi/2, -np.pi/2, np.pi/2]],
#     4:[[0]*4, [1,2,3,4], [0,2,4,8], [-1,5,-2,0.45], [np.pi/2, -np.pi/2, np.pi/2, -np.pi/2]],
#     5:[[0]*5, [1,2,3,4,5], [0,2,4,6,8], [-1,5,-2,1.5,0.45], [np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2]],
#     6:[[0]*6, [1,2,3,4,5,6], [0,2,4,6,8,10], [-1,5,-2,1.5,-0.83,0.45], [np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2]],
#     7:[[0]*7, [1,2,3,4,5,6,7], [0,2,4,6,8,10,12], [-1,5,-2,1.5,-0.83,0.7,0.45], [np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2]],
#     8:[[0]*8, [1,2,3,4,5,6,7,8], [0,2,4,6,8,10,12,14], [-1,5,-2,1.5,-0.83,0.7,-2,0.45], [np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2]],
#     9:[[0]*9, [1,2,3,4,5,6,7,8,9], [0,2,4,6,8,10,12,14,16], [-1,5,-2,1.5,-0.83,0.7,4,-1,0.45], [np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2]],
#     10:[[0]*10, [1,2,3,4,5,6,7,8,9,10], [0,2,4,6,8,10,12,14,16,18], [-1,5,-2,1.5,-0.83,0.7,1,-3,-10,0.45], [np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2]]
#     }

phase_dict = {
    2:[[0]*2, list(np.arange(1,3)), list(np.arange(0,4,2)), [-1, 0.45], [np.pi/2, -np.pi/2], [1.234, 23.2235235]],
    3:[[0]*3, list(np.arange(1,4)), list(np.arange(0,6,2)), [-1,5,-0.45], [np.pi/2, -np.pi/2, np.pi/2]],
    4:[[0]*4, list(np.arange(1,5)), list(np.arange(0,8,2)), [-1,5,-2,0.45], [np.pi/2, -np.pi/2, np.pi/2, -np.pi/2]],
    5:[[0]*5, list(np.arange(1,6)), list(np.arange(0,10,2)), [-1,5,-2,1.5,0.45], [np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2]],
    6:[[0]*6, list(np.arange(1,7)), list(np.arange(0,12,2)), [-1,5,-2,1.5,-0.83,0.45], [np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2]],
    7:[[0]*7, list(np.arange(1,8)), list(np.arange(0,14,2)), [-1,5,-2,1.5,-0.83,0.7,0.45], [np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2]],
    8:[[0]*8, list(np.arange(1,9)), list(np.arange(0,16,2)), [-1,5,-2,1.5,-0.83,0.7,-2,0.45], [np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2]],
    9:[[0]*9, list(np.arange(1,10)), list(np.arange(0,18,2)), [-1,5,-2,1.5,-0.83,0.7,4,-1,0.45], [np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2]],
    10:[[0]*10, list(np.arange(1,11)), list(np.arange(0,20,2)), [-1,5,-2,1.5,-0.83,0.7,1,-3,-10,0.45], [np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2]]
    }

###########################################################################################################
#Linking Sliders *****************************************************************************
#TODO: delete print statements 
def cast_phase_text_to_array(user_phase):
    user_phase_list = list(map(check_if_int, user_phase.strip('][').split(',')))
    user_phase_tuple = (f"{list(np.round(user_phase_list, 2))}", user_phase_list)
    
    #Avoid duplicate entries in `phase_Dropdown.options'
    if user_phase_tuple[1] not in [b[1] for b in phase_Dropdown.options] and \
        str(user_phase_tuple[1]) not in [b[0] for b in phase_Dropdown.options]:
        phase_Dropdown.options += tuple([user_phase_tuple])
    return user_phase_list

###########################################################################################################
def set_phase_options(n):
    # `options` takes a list of key-value pair, where for estetic reasons `key` is the rounded array of `values`
    phase_Dropdown.options = [(f"{list(np.round(item, 2))}", item) for item in phase_dict[n]]
    #print(phase_Dropdown.options)
    return phase_Dropdown.value

###########################################################################################################
def set_initial_user_phase(n):
    phase_Text_Box.value = str(phase_Dropdown.value)
    return phase_Text_Box.value

###########################################################################################################
def change_phase_text(phase):
    phase_Text_Box.value = phase_Dropdown.label
    return phase_Text_Box.value





#TODO: change phase slider to phase dropdown naming 
widgets.dlink((n_Slider, 'value'), (phase_Dropdown, 'value'), set_phase_options);
widgets.dlink((n_Slider, 'value'), (phase_Text_Box, 'value'), set_initial_user_phase);
widgets.dlink((phase_Text_Box, 'value'), (phase_Dropdown, 'value'), cast_phase_text_to_array);
widgets.dlink((phase_Dropdown, "index"), (phase_Text_Box, 'value'), change_phase_text);

precision_Slider = widgets.IntSlider(
            min=0,
            max=10,
            step=1,
            value=2,
            layout=Layout(width = "10cm"),# height="80px"),#"auto"),
            description=r'Decimal Precision:',
            style = {'description_width': 'initial'},
            continuous_update=False
            )


turns_Slider = widgets.IntSlider(
            min=-20,
            max=20,
            step=1,
            value=1,
            layout=Layout(width = "10cm"),# height="80px"),#"auto"),
            description=r'Number of right rotations:',
            style = {'description_width': 'initial'},
            continuous_update=False
            )

axis2_Slider = widgets.IntSlider(
            min=1,
            max=6,
            step=1,
            value=2,
            layout=Layout(width = "10cm"),# height="80px"),#"auto"),
            description=r'Second axis of reflection:',
            style = {'description_width': 'initial'},
            continuous_update=False
            )

axis_Slider = widgets.IntSlider(
            min=1,
            max=6,
            step=1,
            value=1,
            layout=Layout(width = "10cm"),# height="80px"),#"auto"),
            description=r'Axis of reflection:',
            style = {'description_width': 'initial'},
            continuous_update=False
            )


widgets.dlink((n_Slider, 'value'), (axis_Slider, 'max'));
widgets.dlink((n_Slider, 'value'), (axis2_Slider, 'max'));