from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Union

import numpy as np

from rdtattoo import array_generator as ag
from rdtattoo.tattoo_functions import FloatArrayType


class InitialConditionType(Enum):
    PILLAR_GAUSSIAN = "Pillar + Gaussian"
    RANDOM_GAUSSIAN = "Random Gaussian"


@dataclass
class RandomGaussianParams:
    """Parameters for random Gaussian initial conditions."""

    loc: float = 0.0
    scale: float = 0.05


@dataclass
class PillarGaussianParams:
    """Parameters for pillar with Gaussian noise initial conditions."""

    pillar_size: int = 40
    pillar_value_a: float = 0.5
    pillar_value_b: float = 0.25
    noise_level: float = 0.05
    background_value_a: float = 1.0
    background_value_b: float = 0.0


@dataclass
class InitialCondition:
    """A dataclass representing the initial conditions for a reaction-diffusion simulation.

    Uses a discriminated union pattern where each condition type has its own parameter class.
    """

    condition_type: InitialConditionType
    height: int
    width: int
    params: Union[RandomGaussianParams, PillarGaussianParams]

    def generate(self) -> Tuple[FloatArrayType, FloatArrayType]:
        """Generate the initial conditions for both species A and B.

        Returns:
            Tuple[FloatArrayType, FloatArrayType]: A tuple containing the initial conditions
            for species A and B respectively.
        """
        if self.condition_type == InitialConditionType.RANDOM_GAUSSIAN:
            if not isinstance(self.params, RandomGaussianParams):
                raise TypeError(
                    "Random Gaussian condition requires RandomGaussianParams"
                )
            a_initial = ag.random_normal_array(
                self.params.loc, self.params.scale, self.height, self.width
            )
            b_initial = ag.random_normal_array(
                self.params.loc, self.params.scale, self.height, self.width
            )
        else:  # PILLAR_GAUSSIAN
            if not isinstance(self.params, PillarGaussianParams):
                raise TypeError(
                    "Pillar Gaussian condition requires PillarGaussianParams"
                )
            a_initial = ag.generate_centered_pillar_with_noise(
                height=self.height,
                width=self.width,
                pillar_size=self.params.pillar_size,
                pillar_value=self.params.pillar_value_a,
                noise_level=self.params.noise_level,
                background_value=self.params.background_value_a,
            )
            b_initial = ag.generate_centered_pillar_with_noise(
                height=self.height,
                width=self.width,
                pillar_size=self.params.pillar_size,
                pillar_value=self.params.pillar_value_b,
                noise_level=self.params.noise_level,
                background_value=self.params.background_value_b,
            )

        return a_initial, b_initial
