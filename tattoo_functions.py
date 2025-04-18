import numpy as np
import numpy.typing as npt
import scipy  # type: ignore[import]
from pydantic import BaseModel

FloatArrayType = npt.NDArray[np.float64]


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
    normalized_image: FloatArrayType = (image - min_val) / (max_val - min_val)
    return normalized_image


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

    def Ra(self, a: FloatArrayType, b: FloatArrayType) -> FloatArrayType:
        r: FloatArrayType = a - a**3 - b + self.alpha
        return r

    def Rb(self, a: FloatArrayType, b: FloatArrayType) -> FloatArrayType:
        r: FloatArrayType = (a - b) * self.beta
        return r

    def generate_normal_array(self, loc: float, scale: float) -> FloatArrayType:
        return np.random.normal(loc=loc, scale=scale, size=(self.height, self.width))

    def run(
        self, a: FloatArrayType, b: FloatArrayType
    ) -> tuple[FloatArrayType, FloatArrayType, float]:
        run_a = np.array(a)
        run_b = np.array(b)
        t: float = 0
        for _ in range(self.steps):
            t += self.dt
            run_a, run_b = self._run(run_a, run_b)
        return run_a, run_b, t

    def _run(
        self, a: FloatArrayType, b: FloatArrayType
    ) -> tuple[FloatArrayType, FloatArrayType]:
        La = laplacian2D(a, self.dx)
        Lb = laplacian2D(b, self.dx)

        delta_a = self.dt * (self.Da * La + self.Ra(a, b))
        delta_b = self.dt * (self.Db * Lb + self.Rb(a, b))
        a += delta_a
        b += delta_b
        return a, b
