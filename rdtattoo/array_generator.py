import numpy as np

from rdtattoo.tattoo_functions import FloatArrayType


def random_normal_array(
    loc: float, scale: float, height: int, width: int
) -> FloatArrayType:
    return np.random.normal(loc=loc, scale=scale, size=(height, width))


def random_uniform_array(
    low: float, high: float, height: int, width: int
) -> FloatArrayType:
    return np.random.uniform(low=low, high=high, size=(height, width))


def generate_centered_pillar(
    height: int,
    width: int,
    pillar_size: int,
    pillar_value: float,
    background_value: float = 0.0,
) -> FloatArrayType:
    """
    Generate an array with a centered pillar of specified size and value.

    Args:
        height: Height of the output array
        width: Width of the output array
        pillar_size: Size of the pillar (in pixels)
        pillar_value: Value to fill the pillar with

    Returns:
        A 2D numpy array with a centered pillar
    """
    array = np.ones((height, width)) * background_value

    # Calculate the center position
    center = height // 2
    half_pillar = pillar_size // 2

    # Calculate start and end positions for the pillar
    start = center - half_pillar
    end = (
        center + half_pillar + (pillar_size % 2)
    )  # Add remainder for odd-sized pillars

    # Ensure the pillar stays within bounds
    start = max(0, start)
    end = min(height, end)

    # Place the pillar
    array[start:end, start:end] = pillar_value

    return array


def generate_centered_pillar_with_noise(
    height: int,
    width: int,
    pillar_size: int,
    pillar_value: float,
    background_value: float = 0.0,
    noise_level: float = 0.05,
) -> FloatArrayType:
    """
    Generate an array with a centered pillar of specified size and value, with added noise.

    Args:
        height: Height of the output array
        width: Width of the output array
        pillar_size: Size of the pillar (in pixels)
        pillar_value: Value to fill the pillar with
        noise_level: Standard deviation of the Gaussian noise to add (default: 0.5)

    Returns:
        A 2D numpy array with a centered pillar and added noise
    """
    # First generate the base pillar
    array = generate_centered_pillar(
        height, width, pillar_size, pillar_value, background_value
    )

    # Add Gaussian noise to the entire array
    noise = np.random.normal(scale=noise_level, size=array.shape)

    # Add the noise to the array
    return array + noise
