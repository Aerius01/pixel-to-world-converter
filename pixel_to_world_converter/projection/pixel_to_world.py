"""
Pixel-to-world coordinate projection via ray-plane intersection.

This module implements the core geometric projection that converts 2D pixel coordinates
to 3D world coordinates. The conversion assumes:

1. Pinhole camera model (ideal projection with no lens distortion)
2. Ocean surface is a flat plane at altitude = 0 (sea level)
3. All detected objects are on the ocean surface

The projection works by:
    1. Unprojecting the pixel to a 3D ray in camera coordinates
    2. Transforming the ray to world coordinates using camera extrinsics
    3. Finding where the ray intersects the ocean plane (z = 0)
"""

import numpy as np
from ..config import MAX_VELOCITY, RAY_PARALLEL_TOLERANCE


def pixel_to_world_coordinates(pixel_pos, camera_extrinsics, K, K_inv=None):
    """
    Convert pixel coordinates to world coordinates via ray-plane intersection.

    Algorithm:
        1. Convert pixel [u, v] to normalized camera coordinates using K^{-1}
        2. Transform to NED camera frame (swap axes to match gimbal orientation)
        3. Rotate ray direction to world frame using R^{-1} (rotation part of extrinsics)
        4. Compute intersection of ray with ocean plane (z = 0)
        5. Scale ray and add camera position to get world intersection point

    Ray-Plane Intersection:
        Ray: p(t) = camera_pos + t * direction
        Plane: z = 0
        Solve for t: camera_pos.z + t * direction.z = 0
                  => t = -camera_pos.z / direction.z

    Special Case:
        If ray is parallel to ocean surface (|direction.z| < tolerance), use
        camera altitude directly as scale factor.

    Args:
        pixel_pos: ndarray of shape (2,) with [x, y] pixel coordinates
        camera_extrinsics: 4x4 extrinsic matrix [R|t; 0|1]
        K: 3x3 intrinsic matrix
        K_inv: 3x3 inverse intrinsic matrix (optional, computed if not provided)

    Returns:
        ndarray of shape (3,) with [x, y, z] world coordinates in meters (ENU)
        where z should be approximately 0 (on ocean surface)
    """
    # Convert pixel to homogeneous coordinates [u, v, 1]
    p = np.array([pixel_pos[0], pixel_pos[1], 1])

    # Compute inverse intrinsic matrix if not provided
    if K_inv is None:
        K_inv = np.linalg.inv(K)

    # Unproject to normalized camera coordinates: p_norm = K^{-1} * p
    p_camera_normalized = K_inv @ p

    # Transform to NED camera frame (matches gimbal coordinate system)
    # Standard camera: x=right, y=down, z=forward
    # NED camera: x=forward, y=right, z=down
    p_camera_ned_normalized = np.array([-p_camera_normalized[1],  # -y -> forward
                                        p_camera_normalized[0],   # x -> right
                                        p_camera_normalized[2]])  # z unchanged

    # Extract rotation and translation from extrinsic matrix
    R = camera_extrinsics[:3, :3]
    R_inv = R.T  # Inverse rotation = transpose (orthogonal matrix)
    camera_position = camera_extrinsics[:3, -1]

    # Rotate ray direction to world frame
    direction_world = R_inv @ p_camera_ned_normalized

    # Compute ray-plane intersection scale factor
    # If ray is parallel to ocean (direction.z ≈ 0), use camera altitude
    if abs(direction_world[2]) < RAY_PARALLEL_TOLERANCE:
        scale = camera_position[2]
    else:
        # Solve for t: camera_z + t * direction_z = 0 (ocean at z=0)
        scale = -camera_position[2] / direction_world[2]

    # Scale the normalized ray in camera frame
    p_camera_ned_scaled = scale * p_camera_ned_normalized

    # Transform to world and add camera position
    p_world = (R_inv @ p_camera_ned_scaled) + camera_position

    return p_world


def compute_world_velocity(current_position, last_position, time_per_frame_ms):
    """
    Compute velocity from consecutive world positions with magnitude clamping.

    Velocity is computed as finite difference: v = (p_current - p_last) / dt
    Large velocities (> MAX_VELOCITY) are clamped to MAX_VELOCITY to filter
    out spurious detections or tracking errors.

    Args:
        current_position: ndarray of shape (3,) with current world position [x, y, z] in meters
        last_position: ndarray of shape (3,) with previous world position [x, y, z] in meters
        time_per_frame_ms: Time between frames in milliseconds

    Returns:
        ndarray of shape (3,) with velocity [vx, vy, vz] in meters/second,
        clamped to MAX_VELOCITY if magnitude exceeds threshold
    """
    # Compute velocity: v = Δp / Δt (convert ms to seconds)
    current_velocity = (current_position - last_position) * 1000 / time_per_frame_ms

    # Clamp velocity magnitude to MAX_VELOCITY (filter outliers)
    velocity_magnitude = np.linalg.norm(current_velocity)
    if velocity_magnitude > MAX_VELOCITY:
        # Scale down to MAX_VELOCITY while preserving direction
        current_velocity = (MAX_VELOCITY / velocity_magnitude) * current_velocity

    return current_velocity
