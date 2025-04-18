import plotly.graph_objects as go
import streamlit as st
from matplotlib import pyplot as plt

from plotly_colorscales import oslo, turku
from tattoo_functions import RDSimulatorBase

# Initialize the RDSimulatorBase with defaults
default_sim = RDSimulatorBase(
    Da=1.0,
    Db=199,
    alpha=-0.005,
    beta=10,
    dx=1,
    dt=0.001,
    width=100,
    height=100,
    steps=10000,
)

# Streamlit inputs for dynamic parameters
st.sidebar.header("Simulation Parameters")
Da = st.sidebar.slider("Da", 0.0, 5.0, default_sim.Da)
Db = st.sidebar.slider("Db", 0.0, 5.0, default_sim.Db)
alpha = st.sidebar.slider("Alpha", -2.0, 2.0, default_sim.alpha)
beta = st.sidebar.slider("Beta", 0.0, 10.0, default_sim.beta)
dx = st.sidebar.slider("dx", 0.01, 1.0, default_sim.dx)
dt = st.sidebar.slider("dt", 0.01, 2.0, default_sim.dt)
width = st.sidebar.number_input(
    "Width", min_value=10, max_value=500, value=default_sim.width
)
height = st.sidebar.number_input(
    "Height", min_value=10, max_value=500, value=default_sim.height
)
steps = st.sidebar.number_input(
    "Steps", min_value=1, max_value=100000, value=default_sim.steps
)

# Create a new RDSimulatorBase with the selected parameters
sim = RDSimulatorBase(
    Da=Da,
    Db=Db,
    alpha=alpha,
    beta=beta,
    dx=dx,
    dt=dt,
    width=width,
    height=height,
    steps=steps,
)

# UI elements
st.title("Reaction-Diffusion Simulator")
st.write("Adjust the parameters in the sidebar and click 'Run Simulation'")

# Initialize arrays
a_initial = sim.generate_normal_array(0, 0.05)
b_initial = sim.generate_normal_array(0, 0.05)

if st.button("Run Simulation"):
    st.write("Running simulation...")
    a_result, b_result, elapsed_time = sim.run(a_initial, b_initial)
    st.write(f"Simulation took {elapsed_time:.2f} steps")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Final 'a' state")
        fig1 = go.Figure()

        fig1.add_trace(
            go.Heatmap(
                z=a_result,
                colorscale=oslo,
                zmin=0,
                zmax=1,
                showscale=False,
                xaxis="x",
                yaxis="y",
            )
        )
        st.plotly_chart(fig1)

    with col2:
        st.write("Final 'b' state")
        fig2 = go.Figure()

        fig2.add_trace(
            go.Heatmap(
                z=a_result,
                colorscale=turku,
                zmin=0,
                zmax=1,
                showscale=False,
                xaxis="x",
                yaxis="y",
            )
        )
        st.plotly_chart(fig2)
