"""Geometric utility functions for 3D transformations."""

import math
import numpy as np
from ..config import DBL_MAX


def normalize_vector(in_vector):
    """Normalize a 3D vector to unit length.

    Args:
        in_vector: 3D vector to normalize

    Returns:
        np.array: Normalized unit vector, or zero vector if input is zero
    """
    if in_vector[0] == 0.0 and in_vector[1] == 0.0 and in_vector[2] == 0.0:
        return in_vector

    root = math.sqrt(in_vector[0] * in_vector[0] + in_vector[1] * in_vector[1] + in_vector[2] * in_vector[2])
    result = np.array([0.0, 0.0, 0.0])
    result[0] = in_vector[0] / root
    result[1] = in_vector[1] / root
    result[2] = in_vector[2] / root
    return result


def line_plane_intersection(plane_point, plane_normal, line_point, line_direction):
    """Compute intersection point of a line and a plane.

    Args:
        plane_point: Point on the plane
        plane_normal: Normal vector to the plane
        line_point: Point on the line
        line_direction: Direction vector of the line

    Returns:
        np.array: Intersection point, or [DBL_MAX, DBL_MAX, DBL_MAX] if no intersection
    """
    r = np.array([DBL_MAX, DBL_MAX, DBL_MAX])
    d = np.dot(plane_normal, normalize_vector(line_direction))
    if math.isclose(d, 0):
        return r

    if abs(d) > 0.0000001:
        w = line_point - plane_point
        scalar = -np.dot(plane_normal, w) / d
        u = scalar * normalize_vector(line_direction)
        r = line_point + u

    return r


def get_quaternion_from_euler(euler: np.array):
    """Convert Euler angles to quaternion.

    Args:
        euler: Euler angles [pitch, roll, yaw] in radians

    Returns:
        np.array: Quaternion [w, x, y, z]
    """
    cy = math.cos(euler[2] * 0.5)
    sy = math.sin(euler[2] * 0.5)
    cp = math.cos(euler[0] * 0.5)
    sp = math.sin(euler[0] * 0.5)
    cr = math.cos(euler[1] * 0.5)
    sr = math.sin(euler[1] * 0.5)

    q = np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy]
    )

    return q


def get_euler_from_quaternion(q):
    """Convert quaternion to Euler angles.

    Args:
        q: Quaternion [w, x, y, z]

    Returns:
        np.array: Euler angles [pitch, roll, yaw] in radians
    """
    r = np.array([0.0, 0.0, 0.0])

    # roll (x-axis rotation)
    sinr_cosp = 2 * (q[0] * q[1] + q[2] * q[3])
    cosr_cosp = 1 - 2 * (q[1] * q[1] + q[2] * q[2])
    r[1] = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (q[0] * q[2] - q[3] * q[1])
    if abs(sinp) >= 1:
        r[0] = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        r[0] = math.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (q[0] * q[3] + q[1] * q[2])
    cosy_cosp = 1 - 2 * (q[2] * q[2] + q[3] * q[3])
    r[2] = math.atan2(siny_cosp, cosy_cosp)

    return r


def create_matrix_from_quaternion(q):
    """Create a 3x3 rotation matrix from a quaternion.

    Args:
        q: Quaternion [w, x, y, z]

    Returns:
        np.ndarray: 3x3 rotation matrix
    """
    r = np.zeros((3, 3))

    r[0][0] = 2.0 * (q[0] * q[0] + q[1] * q[1]) - 1.0
    r[0][1] = 2.0 * (q[1] * q[2] - q[0] * q[3])
    r[0][2] = 2.0 * (q[1] * q[3] + q[0] * q[2])

    r[1][0] = 2.0 * (q[1] * q[2] + q[0] * q[3])
    r[1][1] = 2.0 * (q[0] * q[0] + q[2] * q[2]) - 1.0
    r[1][2] = 2.0 * (q[2] * q[3] - q[0] * q[1])

    r[2][0] = 2.0 * (q[1] * q[3] - q[0] * q[2])
    r[2][1] = 2.0 * (q[2] * q[3] + q[0] * q[1])
    r[2][2] = 2.0 * (q[0] * q[0] + q[3] * q[3]) - 1.0

    return r
