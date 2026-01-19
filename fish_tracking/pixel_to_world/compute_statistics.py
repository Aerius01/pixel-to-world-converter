from fish_tracking.common.globals import *
#from utils.common import *
from fish_tracking.common.math_utils import *
import math
import copy
import pandas as pd

def pixel_to_world_coordinates(pixel_pos, camera_extrinsics, K):
    """
    Convert 2D pixel coordinates to 3D world coordinates.

    Args:
        pixel_pos: 2D pixel position [x, y]
        camera_extrinsics: 4x4 camera extrinsic matrix for this frame
        K: 3x3 camera intrinsic matrix

    Returns:
        np.array: 3D world position [x, y, z] in meters
    """
    p = np.array([pixel_pos[0], pixel_pos[1], 1])

    # Compute the inverse of the intrinsic matrix
    K_inv = np.linalg.inv(K)

    # Convert the image point to camera coordinates
    Z_scale_new = -camera_extrinsics[2, 3]
    p_camera = np.dot(K_inv, p) * Z_scale_new  # [fX/Z, fY/Z, 1] -> [fX, fY, Z]

    # convert to NED
    p_camera_ned = [-p_camera[1], p_camera[0], p_camera[2]]

    # Invert the rotation matrix
    R = camera_extrinsics[:3, :3]
    R_inv = np.linalg.inv(R)

    # Calculate the translation vector in world coordinates
    t = camera_extrinsics[:3, -1]

    # Rotate then Translate
    p_world = R_inv.dot(p_camera_ned) + t

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




