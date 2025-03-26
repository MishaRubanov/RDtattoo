from typing import Any

import numpy as np
import numpy.typing as npt
import scipy
from pydantic import BaseModel


def normalize(image: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
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
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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


def laplacian2Dconv(
    a: npt.NDArray[np.float64], dx: float
) -> np.ndarray[tuple[int, ...], np.dtype[np.floating[Any]]]:
    # Laplacian kernel
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    # Convolve the input array with the Laplacian kernel
    laplacian_a: npt.NDArray[np.floating[Any]] = scipy.ndimage.convolve(
        a, laplacian_kernel
    )

    # Normalize by dx^2
    laplacian_a /= dx**2

    return laplacian_a


class RDSimulatorBase(BaseModel):
    Da: float
    Db: float
    alpha: float
    beta: float
    dx: float
    dt: float
    width: int
    height: int
    steps: int

    def __post_init__(self):
        self.a, self.b = generate_2random_2darrays(self.height, self.width)
