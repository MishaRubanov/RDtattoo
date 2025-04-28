"""
This module implements a reaction-diffusion system simulator with various reaction types.
It provides classes and functions for simulating different reaction-diffusion patterns
that can be used to generate artistic patterns or study chemical reactions.

To add a new reaction type:
1. Add a new enum value to ReactionType
2. Create two new classes implementing ReactionFunction (one for each species)
3. Add the new reaction functions to ReactionType.get_reaction_functions()
4. Add default parameters to rd_defaults.py

Example of adding a new reaction type:
```python
# In ReactionType enum:
MY_REACTION = 4

# Create reaction function classes:
class MyReactionA(ReactionFunction):
    def __call__(self, a, b, alpha, beta):
        return ...  # Your reaction equation

class MyReactionB(ReactionFunction):
    def __call__(self, a, b, alpha, beta):
        return ...  # Your reaction equation

# Add to get_reaction_functions:
@classmethod
def get_reaction_functions(cls, reaction_type):
    if reaction_type == cls.MY_REACTION:
        return MyReactionA(), MyReactionB()
    ...

# In rd_defaults.py:
MY_REACTION_DEFAULTS = {
    "Da": 0.1,
    "Db": 0.1,
    "alpha": 0.5,
    "beta": 0.5,
    ...
}
"""

from enum import Enum
from typing import Any, Protocol, Self, Tuple, runtime_checkable

import numpy as np
import numpy.typing as npt
import scipy  # type: ignore[import]
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

FloatArrayType = npt.NDArray[np.float64]


@runtime_checkable
class ReactionFunction(Protocol):
    """Protocol defining the interface for reaction functions.

    A reaction function calculates the rate of change for a chemical species
    based on the current concentrations of both species and reaction parameters.

    Methods:
        __call__: Calculate the reaction rate for a species.
    """

    def __call__(
        self, a: FloatArrayType, b: FloatArrayType, alpha: float, beta: float
    ) -> FloatArrayType: ...


class ReactionType(Enum):
    """Enumeration of available reaction-diffusion system types.

    Each reaction type represents a different chemical reaction system with its own
    mathematical equations and behavior patterns.

    Values:
        BRUSSELATOR: The Brusselator model, a theoretical model for a type of
            autocatalytic reaction.
        FITZHUGH_NAGUMO: The FitzHugh-Nagumo model, a simplified model of
            neuron behavior.
        GRAY_SCOTT: The Gray-Scott model, a reaction-diffusion system that can
            produce various patterns.
    """

    BRUSSELATOR = 1
    FITZHUGH_NAGUMO = 2
    GRAY_SCOTT = 3

    @classmethod
    def get_reaction_functions(
        cls, reaction_type: "ReactionType"
    ) -> Tuple[ReactionFunction, ReactionFunction]:
        """Get the reaction functions for a specific reaction type.

        Args:
            reaction_type: The type of reaction system to get functions for.

        Returns:
            A tuple containing two ReactionFunction instances, one for each
            chemical species (A and B) in the reaction system.

        Raises:
            ValueError: If an invalid reaction type is provided.
        """
        if reaction_type == cls.BRUSSELATOR:
            return BrusselatorA(), BrusselatorB()
        elif reaction_type == cls.FITZHUGH_NAGUMO:
            return FitzHughNagumoA(), FitzHughNagumoB()
        elif reaction_type == cls.GRAY_SCOTT:
            return GrayScottA(), GrayScottB()
        else:
            raise ValueError(f"Unknown reaction type: {reaction_type}")


class BrusselatorA(ReactionFunction):
    """Reaction function for species A in the Brusselator model.

    The Brusselator is a theoretical model for a type of autocatalytic reaction.
    This class implements the reaction function for species A:
    dA/dt = α - (1 + β)A + BA²
    """

    def __call__(
        self, a: FloatArrayType, b: FloatArrayType, alpha: float, beta: float
    ) -> FloatArrayType:
        return alpha - ((1 + beta) * a) + (b * (a**2))


class BrusselatorB(ReactionFunction):
    """Reaction function for species B in the Brusselator model.

    This class implements the reaction function for species B:
    dB/dt = βA - BA²
    """

    def __call__(
        self, a: FloatArrayType, b: FloatArrayType, alpha: float, beta: float
    ) -> FloatArrayType:
        return (beta * a) - ((a**2) * b)


class FitzHughNagumoA(ReactionFunction):
    """Reaction function for species A in the FitzHugh-Nagumo model.

    The FitzHugh-Nagumo model is a simplified model of neuron behavior.
    This class implements the reaction function for species A:
    dA/dt = A - A³ - B + α
    """

    def __call__(
        self, a: FloatArrayType, b: FloatArrayType, alpha: float, beta: float
    ) -> FloatArrayType:
        return a - a**3 - b + alpha


class FitzHughNagumoB(ReactionFunction):
    """Reaction function for species B in the FitzHugh-Nagumo model.

    This class implements the reaction function for species B:
    dB/dt = β(A - B)
    """

    def __call__(
        self, a: FloatArrayType, b: FloatArrayType, alpha: float, beta: float
    ) -> FloatArrayType:
        return beta * (a - b)


class GrayScottA(ReactionFunction):
    """Reaction function for species A in the Gray-Scott model.

    The Gray-Scott model is a reaction-diffusion system that can produce
    various patterns. This class implements the reaction function for species A:
    dA/dt = -AB² + α(1 - A)
    """

    def __call__(
        self, a: FloatArrayType, b: FloatArrayType, alpha: float, beta: float
    ) -> FloatArrayType:
        return (-a * (b**2)) + (alpha * (1 - a))


class GrayScottB(ReactionFunction):
    """Reaction function for species B in the Gray-Scott model.

    This class implements the reaction function for species B:
    dB/dt = AB² - (α + β)B
    """

    def __call__(
        self, a: FloatArrayType, b: FloatArrayType, alpha: float, beta: float
    ) -> FloatArrayType:
        return (a * (b**2)) - ((alpha + beta) * b)


def normalize(image: FloatArrayType) -> FloatArrayType:
    """Normalize the image to have values between 0 and 1.

    This function scales the input array so that its values span the range [0, 1].
    If all values are the same, returns an array of zeros.

    Args:
        image: The input array to be normalized.

    Returns:
        The normalized array with values between 0 and 1.
    """
    min_val = image.min()
    max_val = image.max()
    # Avoid division by zero
    if max_val == min_val:
        return np.zeros_like(image)
    normalized_image: FloatArrayType = (image - min_val) / (max_val - min_val)
    return normalized_image


def laplacian2D(a: FloatArrayType, dx: float) -> FloatArrayType:
    """Calculate the 2D Laplacian of an array.

    This function computes the discrete Laplacian operator using a 3x3 kernel
    and normalizes by the spatial step size squared.

    Args:
        a: The input array to compute the Laplacian for.
        dx: The spatial step size.

    Returns:
        The Laplacian of the input array.
    """
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
    """A simulator for reaction-diffusion systems.

    This class implements a numerical solver for reaction-diffusion systems
    using the finite difference method. It supports multiple reaction types
    and can save simulation frames for visualization.

    Attributes:
        Da: Diffusion coefficient for species A.
        Db: Diffusion coefficient for species B.
        alpha: First reaction parameter.
        beta: Second reaction parameter.
        dx: Spatial step size for the discretization.
        dt: Temporal step size for the time integration.
        width: Number of grid points in the x-direction.
        height: Number of grid points in the y-direction.
        steps: Total number of time steps to simulate.
        frames: Number of frames to save during the simulation.
        reaction_type: Type of reaction-diffusion system to simulate.
    """

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
        """Validate that the number of frames is less than the number of steps.

        Returns:
            The validated model instance.

        Raises:
            ValueError: If frames >= steps.
        """
        if self.frames >= self.steps:
            raise ValueError("frames must be lower than steps")
        return self

    def model_post_init(self, context: Any) -> None:
        """Initialize the reaction functions after model initialization."""
        self._initialize_reactions()

    def _initialize_reactions(self) -> None:
        """Initialize the reaction functions based on the selected reaction type."""
        self._reaction_a, self._reaction_b = ReactionType.get_reaction_functions(
            self.reaction_type
        )

    @property
    def reaction_a(self) -> ReactionFunction:
        """Get the reaction function for species A."""
        return self._reaction_a

    @property
    def reaction_b(self) -> ReactionFunction:
        """Get the reaction function for species B."""
        return self._reaction_b

    def run(
        self, a: FloatArrayType, b: FloatArrayType
    ) -> tuple[float, FloatArrayType, FloatArrayType]:
        """Run the reaction-diffusion simulation.

        Args:
            a: Initial concentration field for species A.
            b: Initial concentration field for species B.

        Returns:
            A tuple containing:
            - The final simulation time
            - Array of saved frames for species A
            - Array of saved frames for species B
        """
        run_a = np.array(a)
        run_b = np.array(b)
        t: float = 0

        frame_interval = max(1, round(self.steps / self.frames))

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
        """Perform a single time step of the simulation.

        Args:
            a: Current concentration field for species A.
            b: Current concentration field for species B.

        Returns:
            Updated concentration fields for both species.
        """
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
