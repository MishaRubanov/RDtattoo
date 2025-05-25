import typing
from typing import Any, Union

import numpy as np
import streamlit as st

from rdtattoo import tattoo_plotter as tp
from rdtattoo.initial_conditions import (
    InitialCondition,
    InitialConditionType,
    PillarGaussianParams,
    RandomGaussianParams,
)
from rdtattoo.plotly_colorscales import oslo, turku
from rdtattoo.rd_defaults import (
    brusselator_default,
    fitzhugh_nagumo_default,
    grayscott_bubble_default,
    grayscott_coral_default,
    grayscott_worm_default,
    model_descriptions,
)
from rdtattoo.tattoo_functions import FloatArrayType, RDSimulator

st.set_page_config(layout="wide")
st.sidebar.header("Simulation Parameters")
st.title("Reaction-Diffusion Simulator :sewing_needle:")


def run_simulation(
    sim: RDSimulator, a_initial: FloatArrayType, b_initial: FloatArrayType
):
    st.session_state["simulation_results"] = sim.run(a_initial, b_initial)
    st.write("Simulation completed")


typed_oslo = typing.cast(list[tuple[float, str]], oslo)
typed_turku = typing.cast(list[tuple[float, str]], turku)

# Define available default simulators
default_simulators = {
    "Fitzhugh-Nagumo": fitzhugh_nagumo_default,
    "Gray-Scott (Worm)": grayscott_worm_default,
    "Gray-Scott (Bubble)": grayscott_bubble_default,
    "Gray-Scott (Coral)": grayscott_coral_default,
    "Brusselator": brusselator_default,
}


selected_model = st.selectbox(
    "Reaction-Diffusion Model and Default Parameters",
    options=list(default_simulators.keys()),
    index=0,  # Default to Fitzhugh-Nagumo
)

st.markdown(model_descriptions[selected_model])

default_sim = default_simulators[selected_model]


def generate_number_inputs(default_sim: RDSimulator) -> dict[str, Any]:
    """
    Generate number inputs for each parameter of the simulator.

    Parameters:
    -----------
    default_sim : RDSimulator
        The default simulator to use as a base

    Returns:
    --------
    dict[str, Any]
        Dictionary of parameter values from user inputs
    """
    default_params = default_sim.model_dump()
    user_inputs = {}

    param_labels = {
        "Da": "D_u (Diffusion coefficient of activator)",
        "Db": "D_v (Diffusion coefficient of inhibitor)",
        "alpha": "α (First reaction parameter)",
        "beta": "β (Second reaction parameter)",
        "dx": "Δx (Spatial step size)",
        "dt": "Δt (Time step size)",
        "width": "Width (Grid points in x-direction)",
        "height": "Height (Grid points in y-direction)",
        "steps": "Steps (Total simulation steps)",
        "frames": "Frames (Output frames to save)",
    }

    for param, value in default_params.items():
        if isinstance(value, (int, float)):
            if value <= 0:
                value = 1e-10  # Small positive value

            if value != 0:
                min_val = value / 100
                max_val = value * 100
            else:
                min_val = 1e-10
                max_val = 1e10

            label = f"{param_labels.get(param, param)} (default: {value})"

            if isinstance(value, int):
                user_inputs[param] = st.sidebar.number_input(
                    label=label,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(value),
                    step=1.0,
                )
            elif isinstance(value, float):
                # For floats, use smaller step
                user_inputs[param] = st.sidebar.number_input(
                    label=label,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(value),
                    step=0.1,
                )
        else:
            # For non-numeric parameters, just store the value
            user_inputs[param] = value

    return user_inputs


user_inputs = generate_number_inputs(default_sim)
# Create a new RDSimulator with the selected parameters
sim = RDSimulator(**user_inputs)

# Add initial condition selection
initial_condition_type = st.selectbox(
    "Initial Condition",
    options=[ic.value for ic in InitialConditionType],
    index=0,
)

# Create appropriate parameter object based on selected type
params: Union[RandomGaussianParams, PillarGaussianParams]
if InitialConditionType(initial_condition_type) == InitialConditionType.RANDOM_GAUSSIAN:
    params = RandomGaussianParams()
else:
    params = PillarGaussianParams()

# Create initial condition object
initial_condition = InitialCondition(
    condition_type=InitialConditionType(initial_condition_type),
    height=sim.height,
    width=sim.width,
    params=params,
)

# Generate initial conditions
a_initial, b_initial = initial_condition.generate()

st.write("Adjust the parameters in the sidebar and click 'Run Simulation'")

# Create a centered container for the button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("Run Simulation", use_container_width=True):
        run_simulation(sim=sim, a_initial=a_initial, b_initial=b_initial)

if "simulation_results" not in st.session_state:
    st.session_state["simulation_results"] = (
        0,
        np.zeros((1, sim.height, sim.width)),
        np.zeros((1, sim.height, sim.width)),
    )

elapsed_time, a_frames, b_frames = st.session_state["simulation_results"]

st.write(
    f"Simulation was run for {elapsed_time:.2f} steps" if elapsed_time != 0 else ""
)

fig1 = tp.create_plotly_figure(a_frames, typed_oslo, initial_frame=0)
fig2 = tp.create_plotly_figure(b_frames, typed_turku, initial_frame=0)

col1, col2 = st.columns(2)

with col1:
    st.write("State A")
    st.plotly_chart(fig1)

with col2:
    st.write("State B")
    st.plotly_chart(fig2)
