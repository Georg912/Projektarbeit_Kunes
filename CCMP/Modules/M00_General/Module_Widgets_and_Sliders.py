# Module for Widgets and Sliders
from itertools import permutations
import ipywidgets as widgets
import time  # used for `sleep`
from ipywidgets import Layout
import numpy as np
from IPython.display import clear_output
from .Module_Utilities import check_if_int
from ..M03_Gauge_Symmetry.Module_Symmetry_and_Gauge import Hopping_Matrix_with_Phase as H
from ..M03_Gauge_Symmetry.Module_Symmetry_and_Gauge import Right_Translation_Matrix as T
from ..M03_Gauge_Symmetry.Module_Symmetry_and_Gauge import Magnetic_Flux_Matrix as M
from ..M03_Gauge_Symmetry.Module_Symmetry_and_Gauge import Reflection_Matrix as R

out = widgets.Output()
###########################################################################################################
Iterations_Slider = widgets.IntSlider(
    min=1,
    max=400,
    step=1,
    value=50,
    layout=Layout(width="10cm"),  # height="80px"),#"auto"),
    description=r"Number of iterations $n_\mathrm{its}$:",
    style={'description_width': 'initial'},
    continuous_update=True,
)

###########################################################################################################
p1_Slider = widgets.FloatSlider(
    min=0,
    max=1,
    step=0.01,
    value=0.1,
    layout=Layout(width="10cm"),  # height="80px"),#"auto"),
    description=r'$p_1$',
    style={'description_width': 'initial'},
    continuous_update=False
)

###########################################################################################################
p2_Slider = widgets.FloatSlider(
    min=0,
    max=1,
    step=0.01,
    value=0.,
    layout=Layout(width="10cm"),  # height="80px"),#"auto"),
    description=r'$p_2$',
    style={'description_width': 'initial'},
    continuous_update=False
)

# TODO: Fix weird error, where states_dict is not recognized as a list of 2
###########################################################################################################
states_dict = {
    2: [[1, 0], [0, 1], [0.8, 0.2], [0.5, 0.5]],
    3: [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.3, 0.2]],
    4: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0.5, 0.3, 0.2, 0]],
    5: [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0.5, 0.3, 0.2, 0, 0]],
    6: [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0.5, 0.3, 0.2, 0, 0, 0]],
    7: [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0.5, 0.3, 0.2, 0, 0, 0, 0]],
    8: [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0.5, 0.3, 0.2, 0, 0, 0, 0, 0]],
    9: [[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0.5, 0.3, 0.2, 0, 0, 0, 0, 0, 0]],
    10: [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0.5, 0.3, 0.2, 0, 0, 0, 0, 0, 0, 0]]
}

states_dict = dict(zip(np.arange(2, 11), [list(map(list, sorted(set(permutations(
    list(np.pad([1], (0, n-1), constant_values=0)))))))[::-1] for n in np.arange(2, 11)]))

# TODO: rename blub


###########################################################################################################
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


###########################################################################################################
Initial_State = widgets.Dropdown(
    equals=np.array_equal,  # otherwise "value" checks element wise
    options=states_dict[n_Slider.value],
    value=states_dict[n_Slider.value][0],  # np.array([1,0,0,0,0,0]),
    description='Initial State:',
    style={'description_width': 'initial'},
    continuous_update=False,
)

# TODO: change to be more descriptive
###########################################################################################################
Text_Box = widgets.Text(continuous_update=False,
                        placeholder='Input Initial State',
                        description='User Input:',
                        value=str(Initial_State.value),
                        style={'description_width': 'initial'},
                        )


Save_Figure_Button = widgets.Button(
    layout=Layout(width="5cm"),
    description="Save Current Figure",
    style={'description_width': 'initial'}
)


def set_filename(name):
    text = widgets.Text(continuous_update=False,
                        placeholder='enter filename',
                        description='Filename:',
                        value=name,
                        style={'description_width': 'initial'},
                        )
    return text


###########################################################################################################
# Linking Sliders *****************************************************************************
# TODO: delete print statements in "transform"
def transform(inp):
    # with output:
    # print(inp)
    # print(t.get_state)
    # print(Initial_State.options,  inp, type(inp))
    # print(Initial_State.options[0],  inp, type(inp))
    inp = list(map(check_if_int, inp.strip('][').split(',')))
    # print(Initial_State.value, type(Initial_State.value), inp, type(inp))
    # print("test")
    # Avoid duplicate entries in `Initial_State.options'
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
    # with output:

    # print(Initial_State.options)
    # print(Initial_State.value)
    # print(str(Initial_State.value))
    Text_Box.value = str(Initial_State.options[0])
    return Text_Box.value

###########################################################################################################
# TODO: rename test function


def test(state):
    Text_Box.value = str(Initial_State.value)
    return Text_Box.value


###########################################################################################################
initial_position_Slider = widgets.Dropdown(
    options=np.arange(1, n_Slider.value+1),
    value=1,
    description='Initial position:',
    style={'description_width': 'initial'},
    continuous_update=True,
)

###########################################################################################################
seed_Slider = widgets.IntSlider(
    min=1,
    max=400,
    step=1,
    value=42,
    layout=Layout(width="10cm"),  # height="80px"),#"auto"),
    description=r"Random number generator seed:",
    style={'description_width': 'initial'},
    continuous_update=False,
)

###########################################################################################################
n_paths_Slider = widgets.IntSlider(
    min=100,
    max=10000,
    step=100,
    value=1000,
    layout=Layout(width="10cm"),  # height="80px"),#"auto"),
    description=r"Number of Paths:",
    style={'description_width': 'initial'},
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
    layout=Layout(width="10cm"),  # height="80px"),#"auto"),
    description=r'$t$',
    style={'description_width': 'initial'},
    continuous_update=False
)


###########################################################################################################
widgets.dlink((Text_Box, 'value'), (Initial_State, 'value'), transform)
widgets.dlink((n_Slider, 'value'), (Initial_State, 'value'), Get_Options)
widgets.dlink((n_Slider, 'value'), (Text_Box, 'value'), Set_Initial_Userinput)
widgets.dlink((Initial_State, 'value'), (Text_Box, 'value'), test)
widgets.dlink((n_Slider, 'value'), (initial_position_Slider,
              'value'), Set_Initial_Position_Range)
widgets.link((t_Slider, "value"), (p1_Slider, "value"))


phase_Dropdown = widgets.Dropdown(
    equals=np.array_equal,  # otherwise "value" checks element wise
    options=[('[1, 2, 3, 4, 5, 6]', [1, 2, 3, 4, 5, 6]),
             ('[0, 2, 4, 6, 8, 10]', [0, 2, 4, 6, 8, 10]),
             ('[-1.0, 5.0, -2.0, 1.5, -0.83, 0.45]',
              [-1, 5, -2, 1.5, -0.83, 0.45]),
             ('[1.57, -1.57, 1.57, -1.57, 1.57, -1.57]', [np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2])],
    value=[1, 2, 3, 4, 5, 6],  # np.array([1,0,0,0,0,0]),
    description=r'Choose Phase Vector $\theta$:',
    style={'description_width': 'initial'},
    continuous_update=False,
    layout=Layout(width="15cm")
)


phase_Text_Box = widgets.Text(continuous_update=False,
                              placeholder='Input Phase Vector',
                              description=r'User Phase Vector $\theta$:',
                              value=str(Initial_State.value),
                              style={'description_width': 'initial'},
                              layout=Layout(width="15cm"),
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
    2: [[0]*2, list(np.arange(1, 3)), list(np.arange(0, 4, 2)), [-1, 0.45], [np.pi/2, -np.pi/2], [1.234, 23.2235235]],
    3: [[0]*3, list(np.arange(1, 4)), list(np.arange(0, 6, 2)), [-1, 5, -0.45], [np.pi/2, -np.pi/2, np.pi/2]],
    4: [[0]*4, list(np.arange(1, 5)), list(np.arange(0, 8, 2)), [-1, 5, -2, 0.45], [np.pi/2, -np.pi/2, np.pi/2, -np.pi/2]],
    5: [[0]*5, list(np.arange(1, 6)), list(np.arange(0, 10, 2)), [-1, 5, -2, 1.5, 0.45], [np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2]],
    6: [[0]*6, list(np.arange(1, 7)), list(np.arange(0, 12, 2)), [-1, 5, -2, 1.5, -0.83, 0.45], [np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2]],
    7: [[0]*7, list(np.arange(1, 8)), list(np.arange(0, 14, 2)), [-1, 5, -2, 1.5, -0.83, 0.7, 0.45], [np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2]],
    8: [[0]*8, list(np.arange(1, 9)), list(np.arange(0, 16, 2)), [-1, 5, -2, 1.5, -0.83, 0.7, -2, 0.45], [np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2]],
    9: [[0]*9, list(np.arange(1, 10)), list(np.arange(0, 18, 2)), [-1, 5, -2, 1.5, -0.83, 0.7, 4, -1, 0.45], [np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2]],
    10: [[0]*10, list(np.arange(1, 11)), list(np.arange(0, 20, 2)), [-1, 5, -2, 1.5, -0.83, 0.7, 1, -3, -10, 0.45], [np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2]]
}

###########################################################################################################
# Linking Sliders *****************************************************************************
# TODO: delete print statements


def cast_phase_text_to_array(user_phase, dropdown):
    user_phase_list = list(
        map(check_if_int, user_phase.strip('][').split(',')))
    user_phase_tuple = (
        f"{list(np.round(user_phase_list, 2))}", user_phase_list)

    # Avoid duplicate entries in `phase_Dropdown.options'
    if user_phase_tuple[1] not in [b[1] for b in dropdown.options] and \
            str(user_phase_tuple[1]) not in [b[0] for b in dropdown.options]:
        dropdown.options += tuple([user_phase_tuple])
    return user_phase_list

###########################################################################################################


def set_phase_options(n, dropdown, _dict):
    # `options` takes a list of key-value pair, where for estetic reasons `key` is the rounded array of `values`
    dropdown.options = [(f"{list(np.round(item, 2))}", item)
                        for item in _dict[n]]
    # print(phase_Dropdown.options)
    return dropdown.value

###########################################################################################################


def set_initial_user_phase(n, dropdown, text):
    text.value = str(dropdown.value)
    return text.value


# TODO: change phase slider to phase dropdown naming
widgets.dlink((n_Slider, 'value'), (phase_Dropdown, 'value'),
              lambda x: set_phase_options(n=x, dropdown=phase_Dropdown, _dict=phase_dict))
widgets.dlink((n_Slider, 'value'), (phase_Text_Box, 'value'), lambda x: set_initial_user_phase(
    n=x, dropdown=phase_Dropdown, text=phase_Text_Box))
widgets.dlink((phase_Text_Box, 'value'), (phase_Dropdown, 'value'),
              lambda x: cast_phase_text_to_array(user_phase=x, dropdown=phase_Dropdown))
widgets.dlink((phase_Dropdown, "label"), (phase_Text_Box, 'value'))

precision_Slider = widgets.IntSlider(
    min=0,
    max=10,
    step=1,
    value=2,
    layout=Layout(width="10cm"),  # height="80px"),#"auto"),
    description=r'Decimal Precision:',
    style={'description_width': 'initial'},
    continuous_update=False
)


turns_Slider = widgets.IntSlider(
    min=-20,
    max=20,
    step=1,
    value=1,
    layout=Layout(width="10cm"),  # height="80px"),#"auto"),
    description=r'Number of right rotations:',
    style={'description_width': 'initial'},
    continuous_update=False
)

axis2_Slider = widgets.IntSlider(
    min=1,
    max=6,
    step=1,
    value=2,
    layout=Layout(width="10cm"),  # height="80px"),#"auto"),
    description=r'Second axis of reflection:',
    style={'description_width': 'initial'},
    continuous_update=False
)

axis_Slider = widgets.IntSlider(
    min=1,
    max=6,
    step=1,
    value=1,
    layout=Layout(width="10cm"),  # height="80px"),#"auto"),
    description=r'Axis of reflection:',
    style={'description_width': 'initial'},
    continuous_update=False
)


def change_min_turns(turns):
    turns_Slider.min = -turns_Slider.max
    return turns_Slider.min


widgets.dlink((n_Slider, 'value'), (axis_Slider, 'max'))
widgets.dlink((n_Slider, 'value'), (axis2_Slider, 'max'))
widgets.dlink((n_Slider, 'value'), (turns_Slider, 'max'))
widgets.dlink((turns_Slider, 'max'), (turns_Slider, 'min'), change_min_turns)


#################################################################################
# Widgets for Arbitrary Hopping
button_to_add = widgets.Button(
    layout=Layout(width="5cm"),
    description="Add hopping",
    style={'description_width': 'initial'}
)


button_to_undo = widgets.Button(
    layout=Layout(width="5cm"),
    description="Undo last hopping",
    style={'description_width': 'initial'}
)

button_to_reset = widgets.Button(
    layout=Layout(width="5cm"),
    description="Reset H and T",
    style={'description_width': 'initial'}
)

button_to_show = widgets.Button(
    layout=Layout(width="5cm"),
    description="Show current H and T",
    style={'description_width': 'initial'}
)

i_IntText = widgets.BoundedIntText(
    min=1,
    max=n_Slider.value,
    step=1,
    value=0,
    layout=Layout(width="5cm"),  # height="80px"),#"auto"),
    description=r"Hopping from site $i$ = ",
    style={'description_width': 'initial'},
    continuous_update=False
)


j_IntText = widgets.BoundedIntText(
    min=1,
    max=n_Slider.value,
    step=1,
    value=2,
    layout=Layout(width="5cm"),  # height="80px"),#"auto"),
    description=r'To site $j$ =',
    style={'description_width': 'initial'},
    continuous_update=False
)

p_BoundedFloatText = widgets.BoundedFloatText(
    min=0,
    max=1.,
    step=0.01,
    value=0.1,
    style={'description_width': 'initial'},
    layout=Layout(width="5cm"),  # height="80px"),#"auto"),
    description=r"With ${p_{ij}}$ = ",
    continuous_update=False
)

widgets.dlink((n_Slider, "value"), (i_IntText, "max"))
widgets.dlink((n_Slider, "value"), (j_IntText, "max"))

checkbox = widgets.Checkbox(
    value=False,
    description="Display output messages",
    style={'description_width': 'initial'},
    layout=Layout(width="5cm"))


#########################################################################################
def Click_Save_Figure(b, widget, name_widget, output, path="Figures/"):
    widget.result.savefig(f"{path}{name_widget.value}", bbox_inches='tight')

    with output:
        print(" Done")
        time.sleep(2)
        clear_output()


#############################################################
Hamilton_Dropdown = widgets.Dropdown(
    equals=np.array_equal,  # otherwise "value" checks element wise
    options=(("H", H), ("M", M)),
    value=H,  # np.array([1,0,0,0,0,0]),
    description='Matrix:',
    style={'description_width': 'initial'},
    continuous_update=False,
)

Matrix1_Dropdown = widgets.Dropdown(
    equals=np.array_equal,  # otherwise "value" checks element wise
    options=((H.name, H), (M.name, M), (T.name, T), (R.name, R)),
    value=H,  # np.array([1,0,0,0,0,0]),
    description='Matrix 1:',
    style={'description_width': 'initial'},
    continuous_update=False,
)

Matrix2_Dropdown = widgets.Dropdown(
    equals=np.array_equal,  # otherwise "value" checks element wise
    options=((H.name, H), (M.name, M), (T.name, T), (R.name, R)),
    value=H,  # np.array([1,0,0,0,0,0]),
    description='Matrix 2:',
    style={'description_width': 'initial'},
    continuous_update=False,
)


#################################################################
# phi slider options

phi_Dropdown = widgets.Dropdown(
    equals=np.array_equal,  # otherwise "value" checks element wise
    options=[('[1, 2, 3, 4, 5, 6]', [1, 2, 3, 4, 5, 6])],
    value=[1, 2, 3, 4, 5, 6],  # np.array([1,0,0,0,0,0]),
    description=r'Choose Phase Vector $\varphi$:',
    style={'description_width': 'initial'},
    continuous_update=False,
    layout=Layout(width="15cm")
)


phi_Text_Box = widgets.Text(continuous_update=False,
                            placeholder='Input Phase Vector',
                            description=r'User Phase Vector $\varphi$:',
                            value=str(Initial_State.value),
                            style={'description_width': 'initial'},
                            layout=Layout(width="15cm"),
                            )

entries = 10
d1 = dict(zip(np.arange(2, entries+1),
          [list(np.arange(1, x)) for x in np.arange(3, entries+2)]))
d2 = dict(zip(np.arange(2, entries+1), [list(np.pad([int(x * (x-1) / 2)], (0, x-2),
          "constant", constant_values=int(0))) for x in np.arange(3, entries+2)]))
d3 = dict(zip(np.arange(2, entries+1),
          [list(np.random.rand(x-1).round(2)) for x in np.arange(3, entries+2)]))
d4 = dict(zip(np.arange(2, entries+1),
          [list((np.resize([1, -1], x-1) * np.pi/2).round(2)) for x in np.arange(3, entries+2)]))
ks = list(d1.keys())
phi_dict = {k: [d1[k], d2[k], d3[k], d4[k]] for k in ks}

widgets.dlink((n_Slider, 'value'), (phi_Dropdown, 'value'),
              lambda x: set_phase_options(n=x, dropdown=phi_Dropdown, _dict=phi_dict))
widgets.dlink((n_Slider, 'value'), (phi_Text_Box, 'value'),
              lambda x: set_initial_user_phase(n=x, dropdown=phi_Dropdown, text=phi_Text_Box))
widgets.dlink((phi_Text_Box, 'value'), (phi_Dropdown, 'value'),
              lambda x: cast_phase_text_to_array(user_phase=x, dropdown=phi_Dropdown))
widgets.dlink((phi_Dropdown, "label"), (phi_Text_Box, 'value'))

button_permute = widgets.Button(
    layout=Layout(width="5cm"),
    description=r'permute phase',
    style={'description_width': 'initial'}
)


def cast_permutation_to_dropdown_tuple(user_phase, dropdown):
    user_phase_tuple = (f"{list(np.round(user_phase, 2))}", user_phase)
    # Avoid duplicate entries in `phase_Dropdown.options'
    if user_phase_tuple[1] not in [b[1] for b in dropdown.options] and \
            str(user_phase_tuple[1]) not in [b[0] for b in dropdown.options]:
        dropdown.options += tuple([user_phase_tuple])


def click_permute(b):
    new_phi = list(np.random.permutation(phi_Dropdown.value))
    cast_permutation_to_dropdown_tuple(new_phi, phi_Dropdown)
    phi_Dropdown.value = new_phi


button_permute.on_click(click_permute)


###############################################
checkbox_periodic_boundary = widgets.Checkbox(
    value=False,
    description="Use periodic isotropic n-chain",
    style={'description_width': 'initial'},
    layout=Layout(width="7cm"))


p1_BoundedFloatText = widgets.BoundedFloatText(
    min=0,
    max=0.5,
    step=0.01,
    value=0.1,
    style={'description_width': 'initial'},
    layout=Layout(width="5cm"),  # height="80px"),#"auto"),
    description=r"1st: $p_1$ = ",
    continuous_update=False
)

p2_BoundedFloatText = widgets.BoundedFloatText(
    min=0,
    max=0.5,
    step=0.001,
    value=0.,
    style={'description_width': 'initial'},
    layout=Layout(width="5cm"),  # height="80px"),#"auto"),
    description=r"2nd: $p_2$ = ",
    continuous_update=False
)

p3_BoundedFloatText = widgets.BoundedFloatText(
    min=0,
    max=0.5,
    step=0.0001,
    value=0.,
    layout=Layout(width="5cm"),  # height="80px"),#"auto"),
    description=r"3rd: $p_3$ = ",
    continuous_update=False,
    style={'description_width': 'initial'},
)

#########################################################################################################
