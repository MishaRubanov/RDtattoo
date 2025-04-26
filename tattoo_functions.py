from enum import Enum
from typing import Any, Protocol, Self, Tuple, runtime_checkable

import numpy as np
import numpy.typing as npt
import scipy  # type: ignore[import]
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

FloatArrayType = npt.NDArray[np.float64]


@runtime_checkable
class ReactionFunction(Protocol):
    def __call__(
        self, a: FloatArrayType, b: FloatArrayType, alpha: float, beta: float
    ) -> FloatArrayType: ...


class ReactionType(Enum):
    BRUSSELATOR = 1
    FITZHUGH_NAGUMO = 2
    GRAY_SCOTT = 3

    @classmethod
    def get_reaction_functions(
        cls, reaction_type: "ReactionType"
    ) -> Tuple[ReactionFunction, ReactionFunction]:
        if reaction_type == cls.BRUSSELATOR:
            return BrusselatorA(), BrusselatorB()
        elif reaction_type == cls.FITZHUGH_NAGUMO:
            return FitzHughNagumoA(), FitzHughNagumoB()
        elif reaction_type == cls.GRAY_SCOTT:
            return GrayScottA(), GrayScottB()


class BrusselatorA(ReactionFunction):
    def __call__(
        self, a: FloatArrayType, b: FloatArrayType, alpha: float, beta: float
    ) -> FloatArrayType:
        return alpha - (1 + beta) * a + b * (a**2)


class BrusselatorB(ReactionFunction):
    def __call__(
        self, a: FloatArrayType, b: FloatArrayType, alpha: float, beta: float
    ) -> FloatArrayType:
        return b * beta - beta * (b**2)


class FitzHughNagumoA(ReactionFunction):
    def __call__(
        self, a: FloatArrayType, b: FloatArrayType, alpha: float, beta: float
    ) -> FloatArrayType:
        return a - a**3 - b + alpha


class FitzHughNagumoB(ReactionFunction):
    def __call__(
        self, a: FloatArrayType, b: FloatArrayType, alpha: float, beta: float
    ) -> FloatArrayType:
        return beta * (a - b)


class GrayScottA(ReactionFunction):
    def __call__(
        self, a: FloatArrayType, b: FloatArrayType, alpha: float, beta: float
    ) -> FloatArrayType:
        return (-a * (b**2)) + (alpha * (1 - a))


class GrayScottB(ReactionFunction):
    def __call__(
        self, a: FloatArrayType, b: FloatArrayType, alpha: float, beta: float
    ) -> FloatArrayType:
        return (a * (b**2)) - ((alpha + beta) * b)


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
    # Avoid division by zero
    if max_val == min_val:
        return np.zeros_like(image)
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


class RDSimulator(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    Da: float = Field(
        title="Diffusion Coefficient A",
        description="Diffusion coefficient for species A",
    )
    Db: float = Field(
        title="Diffusion Coefficient B",
        description="Diffusion coefficient for species B",
    )
    alpha: float = Field(
        title="Alpha Parameter",
        description="First reaction parameter, controls the production rate of species A",
    )
    beta: float = Field(
        title="Beta Parameter",
        description="Second reaction parameter, controls the production rate of species B",
    )
    dx: float = Field(
        title="Spatial Step Size",
        description="Spatial step size for the discretization",
    )
    dt: float = Field(
        title="Time Step Size",
        description="Temporal step size for the time integration",
    )
    width: int = Field(
        title="Grid Width", description="Number of grid points in the x-direction"
    )
    height: int = Field(
        title="Grid Height", description="Number of grid points in the y-direction"
    )
    steps: int = Field(
        title="Simulation Steps", description="Total number of time steps to simulate"
    )
    frames: int = Field(
        title="Output Frames",
        description="Number of frames to save during the simulation",
    )
    reaction_type: ReactionType = Field(
        title="Reaction Type",
        default=ReactionType.FITZHUGH_NAGUMO,
        description="Type of reaction-diffusion system to simulate",
    )
    _reaction_a: ReactionFunction = PrivateAttr(default_factory=FitzHughNagumoA)
    _reaction_b: ReactionFunction = PrivateAttr(default_factory=FitzHughNagumoB)

    @model_validator(mode="after")
    def validate_frames_steps(self) -> Self:
        if self.frames >= self.steps:
            raise ValueError("frames must be lower than steps")
        return self

    def model_post_init(self, context: Any) -> None:
        self._initialize_reactions()

    def _initialize_reactions(self) -> None:
        self._reaction_a, self._reaction_b = ReactionType.get_reaction_functions(
            self.reaction_type
        )

    @property
    def reaction_a(self) -> ReactionFunction:
        return self._reaction_a

    @property
    def reaction_b(self) -> ReactionFunction:
        return self._reaction_b

    def generate_normal_array(self, loc: float, scale: float) -> FloatArrayType:
        return np.random.normal(loc=loc, scale=scale, size=(self.height, self.width))

    def run(
        self, a: FloatArrayType, b: FloatArrayType
    ) -> tuple[float, FloatArrayType, FloatArrayType]:
        run_a = np.array(a)
        run_b = np.array(b)
        t: float = 0

        # Calculate the frame interval and round to the nearest integer
        frame_interval = max(1, round(self.steps / self.frames))

        # Initialize the 3D arrays to store the frames
        a_frames = np.zeros((self.frames, self.height, self.width), dtype=np.float64)
        b_frames = np.zeros((self.frames, self.height, self.width), dtype=np.float64)

        frame_index: int = 0

        for step in range(self.steps):
            t += self.dt
            run_a, run_b = self._run(run_a, run_b)

            # Store the frame every frame_interval steps
            if step % frame_interval == 0 and frame_index < self.frames:
                a_frames[frame_index, :, :] = run_a
                b_frames[frame_index, :, :] = run_b
                frame_index += 1

        return t, a_frames, b_frames

    def _run(
        self, a: FloatArrayType, b: FloatArrayType
    ) -> tuple[FloatArrayType, FloatArrayType]:
        La = laplacian2D(a, self.dx)
        Lb = laplacian2D(b, self.dx)

        delta_a = self.dt * (
            self.Da * La + self.reaction_a(a, b, self.alpha, self.beta)
        )
        delta_b = self.dt * (
            self.Db * Lb + self.reaction_b(a, b, self.alpha, self.beta)
        )
        a += delta_a
        b += delta_b
        return a, b
