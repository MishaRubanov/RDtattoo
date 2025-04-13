from pathlib import Path
from typing import Any, Callable

import matplotlib
import matplotlib.axes
import numpy as np
import numpy.typing as npt
import scipy  # type: ignore[import]
from matplotlib import pyplot as plt
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


# def laplacian2D(a, dx):
#     return (
#         -4 * a
#         + np.roll(a, 1, axis=0)
#         + np.roll(a, -1, axis=0)
#         + np.roll(a, +1, axis=1)
#         + np.roll(a, -1, axis=1)
#     ) / (dx**2)


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
    _a: FloatArrayType
    _b: FloatArrayType

    def model_post_init(self, __context: Any) -> None:
        self._a, self._b = generate_2random_2darrays(self.height, self.width)

    def update(self):
        for _ in range(self.steps):
            self.t += self.dt
            self._update()

    def _update(self):
        assert self._a is not None
        assert self._b is not None
        La = laplacian2D(self._a, self.dx)
        Lb = laplacian2D(self._b, self.dx)

        delta_a = self.dt * (self.Da * La + self.Ra(self._a, self._b, self.alpha))
        delta_b = self.dt * (self.Db * Lb + self.Rb(self._a, self._b, self.beta))
        self._a += delta_a
        self._b += delta_b

    def draw(self, ax: tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]):
        ax[0].clear()
        ax[1].clear()
        assert type(self._a) is FloatArrayType
        assert type(self._b) is FloatArrayType
        assert isinstance(self._a, np.ndarray), "self._a must be a numpy array"
        assert isinstance(self._b, np.ndarray), "self._b must be a numpy array"

        ax[0].imshow(X=self._a, dtype=np.float64, cmap="jet")  # type: ignore[reportUnknownMemberType]
        ax[1].imshow(self._b, cmap="brg")  # type: ignore[reportUnknownMemberType]

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
