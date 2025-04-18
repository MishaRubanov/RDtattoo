import plotly.graph_objects as go

from tattoo_functions import FloatArrayType, RDSimulatorBase, normalize


# Helper function to create animation frames for Plotly with annotations
def create_animation_frames(
    frames_data: list[FloatArrayType], colorscale: str
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
