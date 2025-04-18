import numpy as np
import plotly.graph_objects as go
import streamlit as st

from plotly_colorscales import oslo, turku
from tattoo_functions import FloatArrayType, RDSimulatorBase, normalize

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
    frames=100,
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
frames = st.sidebar.number_input(
    "Frames", min_value=1, max_value=100, value=default_sim.frames
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
    frames=frames,
)

# UI elements
st.title("Tattoo RD Simulator :sewing_needle:")
st.write("Adjust the parameters in the sidebar and click 'Run Simulation'")

# Initialize arrays
a_initial = sim.generate_normal_array(0, 0.05)
b_initial = sim.generate_normal_array(0, 0.05)

if st.button("Run Simulation"):
    st.session_state["simulation_results"] = sim.run(a_initial, b_initial)
    st.write("Simulation completed")

# Initialize placeholders to avoid key errors before running the simulation
if "simulation_results" not in st.session_state:
    st.session_state["simulation_results"] = (
        0,
        np.zeros((1, sim.height, sim.width)),
        np.zeros((1, sim.height, sim.width)),
    )

elapsed_time, a_frames, b_frames = st.session_state["simulation_results"]

st.write(f"Simulation took {elapsed_time:.2f} steps" if elapsed_time != 0 else "")


# Helper function to create animation frames for Plotly with annotations
def create_animation_frames(
    frames_data: FloatArrayType, colorscale: str
) -> list[go.Frame]:
    frames = []
    for i, frame_data in enumerate(frames_data):
        normalized_data = normalize(frame_data)
        frames.append(
            go.Frame(
                data=[
                    go.Heatmap(
                        z=normalized_data,
                        colorscale=colorscale,
                        showscale=False,
                    ),
                ],
                name=str(i),
                layout=go.Layout(
                    annotations=[
                        go.layout.Annotation(
                            text=f"Frame: {i}",
                            x=0.5,
                            y=1.05,
                            showarrow=False,
                            xref="paper",
                            yref="paper",
                            align="center",
                            font=dict(size=12),
                        )
                    ]
                ),
            )
        )
    return frames


# Helper function to create the Plotly figure with play/pause buttons and frame annotation
def create_plotly_figure(
    frames_data: FloatArrayType, colorscale: str, initial_frame: int
) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=normalize(frames_data[initial_frame]),
                colorscale=colorscale,
                showscale=False,
                xaxis="x",
                yaxis="y",
            ),
        ],
        layout=go.Layout(
            annotations=[
                go.layout.Annotation(
                    text=f"Frame: {initial_frame}",
                    x=0.5,
                    y=1.05,
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    align="center",
                    font=dict(size=12),
                )
            ],
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {"duration": 50, "redraw": True},
                                    "fromcurrent": True,
                                    "mode": "immediate",
                                },
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                },
                            ],
                        ),
                    ],
                )
            ],
        ),
        frames=create_animation_frames(frames_data, colorscale),
    )
    return fig


# Create figures with animation frames and annotations
fig1 = create_plotly_figure(a_frames, oslo, initial_frame=0)
fig2 = create_plotly_figure(b_frames, turku, initial_frame=0)

col1, col2 = st.columns(2)

with col1:
    st.write("State 'a'")
    st.plotly_chart(fig1)

with col2:
    st.write("State 'b'")
    st.plotly_chart(fig2)
