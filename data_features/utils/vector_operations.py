import numpy as np
from typing import Union, Tuple


def calculate_angle(point1: np.ndarray, point2: np.ndarray, point3: np.ndarray) -> float:
    """
    Calculate angle at point2 formed by three 3D points.

    Args:
        point1: First point (x, y, z)
        point2: Vertex point (x, y, z)
        point3: Third point (x, y, z)

    Returns:
        Angle in degrees
    """
    vector1 = subtract_vectors(point1, point2)
    vector2 = subtract_vectors(point3, point2)

    vector1_norm = normalize_vector(vector1)
    vector2_norm = normalize_vector(vector2)

    dot_product = np.dot(vector1_norm, vector2_norm)
    dot_product = np.clip(dot_product, -1.0, 1.0)

    angle_radians = np.arccos(dot_product)
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees


def subtract_vectors(vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
    """
    Calculate the difference between two vectors.

    Args:
        vector1: First vector (x, y, z)
        vector2: Second vector (x, y, z)

    Returns:
        Resulting vector (vector1 - vector2)
    """
    return vector1 - vector2


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    Normalize a vector of any dimension.

    Args:
        vector: Input vector of n dimensions

    Returns:
        Normalized vector (unit vector)
    """
    magnitude = np.linalg.norm(vector)

    if magnitude == 0:
        return vector

    return vector / magnitude
