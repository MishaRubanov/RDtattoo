from pathlib import Path
from typing import Any, Callable, Optional

import cmcrameri.cm as cmc  # type: ignore[import]
import matplotlib
import matplotlib.axes
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import scipy  # type: ignore[import]
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pydantic import BaseModel

FloatArrayType = npt.NDArray[np.float64]

FloatFunction = Callable[[FloatArrayType, FloatArrayType, float], FloatArrayType]


def normalize(image: FloatArrayType) -> FloatArrayType:
    """
    Normalize the image to have values between 0 and 1.

    Parameters:
    image (numpy.ndarray): The input image to be normalized.

    Returns:
    numpy.ndarray: The normalized image.
    """
    min_val = image.min()
    max_val = image.max()
    normalized_image = (image - min_val) / (max_val - min_val)
    return normalized_image


def generate_2random_2darrays(
    height: int, width: int
) -> tuple[FloatArrayType, FloatArrayType]:
    return (
        np.random.normal(loc=0, scale=0.05, size=(height, width)),
        np.random.normal(loc=0, scale=0.05, size=(height, width)),
    )


def laplacian2D(a: FloatArrayType, dx: float) -> FloatArrayType:
    # Laplacian kernel
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    # Convolve the input array with the Laplacian kernel
    laplacian_a: FloatArrayType = scipy.ndimage.convolve(
        a, laplacian_kernel, mode="reflect"
    )

    # Normalize by dx^2
    laplacian_b: FloatArrayType = (laplacian_a / dx**2).astype(np.float64)

    return laplacian_b


def Ra(a: FloatArrayType, b: FloatArrayType, alpha: float) -> FloatArrayType:
    r: FloatArrayType = a - a**3 - b + alpha
    return r


def Rb(a: FloatArrayType, b: FloatArrayType, beta: float) -> FloatArrayType:
    r: FloatArrayType = (a - b) * beta
    return r


class RDSimulatorBase(BaseModel):
    Da: float
    Db: float
    Ra: FloatFunction
    Rb: FloatFunction
    alpha: float
    beta: float
    dx: float
    dt: float
    width: int
    height: int
    steps: int
    t: float = 0
    a: Optional[FloatArrayType] = None
    b: Optional[FloatArrayType] = None

    class Config:
        arbitrary_types_allowed = True

    def generate_normal_array(self, loc: float, scale: float) -> FloatArrayType:
        return np.random.normal(loc=loc, scale=scale, size=(self.height, self.width))

    def model_post_init(self, __context: Any) -> None:
        if self.a is None:
            self.a = self.generate_normal_array(loc=0, scale=0.05)
        if self.b is None:
            self.b = self.generate_normal_array(loc=0, scale=0.05)

    def update(self):
        for _ in range(self.steps):
            self.t += self.dt
            self._update()

    def _update(self):
        assert self.a is not None
        assert self.b is not None
        La = laplacian2D(self.a, self.dx)
        Lb = laplacian2D(self.b, self.dx)

        delta_a = self.dt * (self.Da * La + self.Ra(self.a, self.b, self.alpha))
        delta_b = self.dt * (self.Db * Lb + self.Rb(self.a, self.b, self.beta))
        self.a += delta_a
        self.b += delta_b

    def draw(self, ax: tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]):
        ax[0].clear()
        ax[1].clear()
        assert isinstance(self.a, np.ndarray), "self.a must be a numpy array"
        assert isinstance(self.b, np.ndarray), "self.b must be a numpy array"

        ax[0].imshow(X=self.a, cmap="jet")  # type: ignore[reportUnknownMemberType]
        ax[1].imshow(self.b, cmap="brg")  # type: ignore[reportUnknownMemberType]

        ax[0].set_title("A, t = {:.2f}".format(self.t))  # type: ignore[reportUnknownMemberType]
        ax[1].set_title("B, t = {:.2f}".format(self.t))  # type: ignore[reportUnknownMemberType]

    def initialise_figure(self):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))  # type: ignore[reportUnknownMemberType]
        return fig, ax

    def plot_evolution_outcome(self, filename: Path, n_steps: int):
        """
        Evolves and save the outcome of evolving the system for n_steps
        """
        fig, ax = self.initialise_figure()

        for _ in range(n_steps):
            self.update()

        self.draw(ax)
        fig.savefig(filename)  # type: ignore[reportUnknownMemberType]
        plt.close()

    def plot_side_by_side(
        self,
        title1: str = "molecule a",
        title2: str = "molecule b",
    ) -> tuple[Figure, matplotlib.axes.Axes]:
        """
        Plot two images side by side.

        Args:
            array1 (FloatArrayType): First image array.
            array2 (FloatArrayType): Second image array.
            title1 (str): Title for the first image.
            title2 (str): Title for the second image.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(self.a, cmap=cmc.oslo)
        axes[0].set_title(title1)
        axes[0].axis("off")  # Hide axis

        axes[1].imshow(self.b, cmap=cmc.lajolla)
        axes[1].set_title(title2)
        axes[1].axis("off")  # Hide axis

        plt.tight_layout()
        plt.show()
        return fig, axes

    def plot_side_by_side_animation(self, n_frames: int, interval: int = 100):

        fig = go.Figure()

        frames = []
        for i in range(n_frames):
            for _ in range(self.steps):
                self.update()  # Update the simulator state
            frame = go.Frame(
                data=[
                    go.Heatmap(
                        z=self.a,
                        colorscale="Inferno",
                        zmin=0,
                        zmax=1,
                        showscale=False,
                        xaxis="x",
                        yaxis="y",
                    ),
                    go.Heatmap(
                        z=self.b,
                        colorscale="Inferno",
                        zmin=0,
                        zmax=1,
                        showscale=False,
                        xaxis="x2",
                        yaxis="y2",
                    ),
                ],
                name=f"frame_{i}",
            )
            frames.append(frame)

        fig.add_trace(
            go.Heatmap(
                z=self.a,
                colorscale="Inferno",
                zmin=0,
                zmax=1,
                showscale=False,
                xaxis="x",
                yaxis="y",
            )
        )

        fig.add_trace(
            go.Heatmap(
                z=self.b,
                colorscale="Inferno",
                zmin=0,
                zmax=1,
                showscale=False,
                xaxis="x2",
                yaxis="y2",
            )
        )

        fig.update_layout(
            title="Side by Side Animation",
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [
                                None,
                                {
                                    "frame": {"duration": interval, "redraw": True},
                                    "fromcurrent": True,
                                },
                            ],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top",
                }
            ],
            sliders=[
                {
                    "active": 0,
                    "yanchor": "top",
                    "xanchor": "left",
                    "currentvalue": {
                        "font": {"size": 20},
                        "prefix": "Frame:",
                        "visible": True,
                        "xanchor": "right",
                    },
                    "transition": {"duration": interval, "easing": "cubic-in-out"},
                    "pad": {"b": 10, "t": 50},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [
                                [f"frame_{k}"],
                                {
                                    "frame": {"duration": interval, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": interval},
                                },
                            ],
                            "label": str(k),
                            "method": "animate",
                        }
                        for k in range(n_frames)
                    ],
                }
            ],
            xaxis=dict(domain=[0.0, 0.45]),
            yaxis=dict(domain=[0.0, 0.9]),
            xaxis2=dict(domain=[0.55, 1.0]),
            yaxis2=dict(domain=[0.0, 0.9]),
            showlegend=False,
        )

        fig.frames = frames
        fig.show()
