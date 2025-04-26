import plotly.graph_objects as go

from tattoo_functions import FloatArrayType, normalize


# Helper function to create animation frames for Plotly with annotations
def create_animation_frames(
    frames_data: FloatArrayType, colorscale: list[tuple[float, str]]
) -> list[go.Frame]:
    """
    Create animation frames for a Plotly heatmap with annotations.

    Parameters:
        frames_data (FloatArrayType): A 3D array where each sub-array represents a frame
            of data to be animated. Make sure that the first dimension of the array is the index for the list
            - as output by RDSimulator.run().
        colorscale (list of tuple): A list of tuples representing the colorscale for the heatmap.
            Each tuple contains a float and a string representing a color.

    Returns:
        list of go.Frame: A list of Plotly Frame objects that can be used to create an animated heatmap.
    """
    frames: list[go.Frame] = []
    for i, frame_data in enumerate(frames_data):
        assert frame_data.ndim == 2
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
    frames_data: FloatArrayType,
    colorscale: list[tuple[float, str]],
    initial_frame: int,
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
            width=600,  # Set fixed width
            height=600,  # Set fixed height to match width for square aspect ratio
            paper_bgcolor="rgb(30, 30, 30)",  # Dark background for the entire figure
            plot_bgcolor="rgb(30, 30, 30)",  # Dark background for the plot area
            font=dict(color="white"),  # White text for better contrast
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
            ),
            annotations=[
                go.layout.Annotation(
                    text=f"Frame: {initial_frame}",
                    x=0.5,
                    y=1.05,
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    align="center",
                    font=dict(size=12, color="white"),
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
