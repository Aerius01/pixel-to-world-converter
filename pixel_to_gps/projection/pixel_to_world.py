"""Pixel-to-world coordinate projection functions."""

import numpy as np
from ..config import MAX_VELOCITY, RAY_PARALLEL_TOLERANCE


def pixel_to_world_coordinates(pixel_pos, camera_extrinsics, K, K_inv=None):
    """
    Convert 2D pixel coordinates to 3D world coordinates using generalized Z-scaling.

    This generalizes the original method to work with arbitrary camera orientations.
    The old method used a simplified formula that only worked when pitch=0, which made
    R_inv[2,:] = [0,0,1]. For arbitrary pitch, we compute the correct scale factor by
    solving for where the projected ray intersects the ocean surface (z=0).

    Args:
        pixel_pos: 2D pixel position [x, y]
        camera_extrinsics: 4x4 camera extrinsic matrix
        K: 3x3 camera intrinsic matrix
        K_inv: (optional) Pre-computed inverse of K for efficiency

    Returns:
        np.array: 3D world position [x, y, z] at ocean surface (z≈0) in meters
    """
    # Convert pixel to homogeneous coordinates
    p = np.array([pixel_pos[0], pixel_pos[1], 1])

    # Use pre-computed inverse if available, otherwise compute it
    if K_inv is None:
        K_inv = np.linalg.inv(K)

    # Get normalized camera coordinates (at unit depth in camera frame)
    p_camera_normalized = K_inv @ p

    # Apply NED transformation (camera frame to body frame)
    # This is part of the coordinate system convention used throughout the codebase
    p_camera_ned_normalized = np.array([-p_camera_normalized[1],
                                        p_camera_normalized[0],
                                        p_camera_normalized[2]])

    # Extract rotation matrix and camera position from extrinsics
    R = camera_extrinsics[:3, :3]
    # R is orthogonal, so R^-1 = R^T (much faster than explicit inverse)
    R_inv = R.T
    camera_position = camera_extrinsics[:3, -1]

    # Transform normalized direction to world coordinates
    direction_world = R_inv @ p_camera_ned_normalized

    # Compute scale factor such that p_world[2] = 0
    # Old method used: scale = -camera_extrinsics[2,3] (only works when pitch=0)
    # General method: scale = -camera_position[2] / direction_world[2]

    if abs(direction_world[2]) < RAY_PARALLEL_TOLERANCE:
        # Ray nearly parallel to ocean - shouldn't happen with downward camera
        # Fallback: project straight down
        scale = camera_position[2]
    else:
        scale = -camera_position[2] / direction_world[2]

    # Apply scale and transform to world coordinates
    # This is equivalent to: p_world = camera_position + scale * direction_world
    p_camera_ned_scaled = scale * p_camera_ned_normalized
    p_world = (R_inv @ p_camera_ned_scaled) + camera_position

    return p_world


def compute_world_velocity(current_position, last_position, time_per_frame_ms):
    """
    Compute world velocity from position delta, with velocity capping.

    Args:
        current_position: Current 3D world position [x, y, z]
        last_position: Previous 3D world position [x, y, z]
        time_per_frame_ms: Time between frames in milliseconds

    Returns:
        np.array: Velocity vector [vx, vy, vz] in m/s (capped at MAX_VELOCITY)
    """
    current_velocity = (current_position - last_position) * 1000 / time_per_frame_ms

    # Cap velocity magnitude at MAX_VELOCITY
    velocity_magnitude = np.linalg.norm(current_velocity)
    if velocity_magnitude > MAX_VELOCITY:
        current_velocity = (MAX_VELOCITY / velocity_magnitude) * current_velocity

    return current_velocity
