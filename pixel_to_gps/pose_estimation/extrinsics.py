"""
Camera calibration matrix computation (intrinsic and extrinsic).

This module computes the transformation matrices needed for projecting between
pixel coordinates and world coordinates:

1. Intrinsic matrix (K): Maps 3D camera coordinates to 2D pixel coordinates
2. Extrinsic matrix (E): Maps 3D world coordinates to 3D camera coordinates

Together, these form the full projection pipeline:
    World coords --[Extrinsic]--> Camera coords --[Intrinsic]--> Pixel coords
"""

import numpy as np
from scipy.spatial.transform import Rotation as Rot


def compute_extrinsic_matrix(angles, camera_position):
    """
    Compute 4x4 camera extrinsic matrix from gimbal angles and position.

    The extrinsic matrix transforms world coordinates to camera coordinates:
        [x_cam]   [R | t] [x_world]
        [y_cam] = [--+--] [y_world]
        [z_cam]   [0 | 1] [z_world]
        [  1  ]            [   1   ]

    Where R is a 3x3 rotation matrix and t is a 3x1 translation vector.

    Rotation Convention:
        - Applies rotations in XYZ (roll-pitch-yaw) order
        - Yaw is negated to match coordinate system convention
        - Input angles are in degrees

    Args:
        angles: ndarray of shape (3,) with [roll, pitch, yaw] in degrees
        camera_position: ndarray of shape (3,) with [x, y, z] in meters (ENU)

    Returns:
        ndarray of shape (4, 4) containing the extrinsic matrix:
            [[R00, R01, R02, tx],
             [R10, R11, R12, ty],
             [R20, R21, R22, tz],
             [  0,   0,   0,  1]]
    """
    # Convert gimbal angles from degrees to radians (negate yaw for coordinate convention)
    rotation = Rot.from_euler('xyz', [np.deg2rad(angles[0]), np.deg2rad(angles[1]), -np.deg2rad(angles[2])])
    rotation_matrix = rotation.as_matrix()

    # Translation vector (camera position in world frame)
    translation = np.array(camera_position).reshape(3, 1)

    # Combine rotation and translation into 3x4 matrix [R|t]
    rotation_translation = np.hstack((rotation_matrix, translation))

    # Add homogeneous row [0, 0, 0, 1] to make 4x4 matrix
    extrinsic = np.vstack((rotation_translation, [0, 0, 0, 1]))

    return extrinsic


def compute_intrinsic_matrix(focal_length, pixel_size, image_width, image_height):
    """
    Compute 3x3 camera intrinsic matrix from physical camera parameters.

    The intrinsic matrix K maps 3D points in camera coordinates to 2D pixel coordinates:
        [u]       [x_cam]
        [v] = K * [y_cam]  where K = [[f_x,   0, c_x],
        [1]       [z_cam]              [  0, f_y, c_y],
                  [  1  ]              [  0,   0,   1]]

    Parameters:
        - f_x, f_y: Focal lengths in pixel units (focal_length / pixel_size)
        - c_x, c_y: Principal point (optical center, assumed at image center)

    Args:
        focal_length: Physical focal length in meters (distance from aperture to sensor)
        pixel_size: Tuple (pixel_width, pixel_height) in meters
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        ndarray of shape (3, 3) containing the intrinsic matrix K
    """
    # Convert focal length from meters to pixels
    f_x = focal_length / pixel_size[0]  # Focal length in horizontal pixels
    f_y = focal_length / pixel_size[1]  # Focal length in vertical pixels

    # Principal point at image center (pixel coordinates)
    c_x = image_width / 2
    c_y = image_height / 2

    # Construct intrinsic matrix (pinhole camera model)
    K = np.array([
        [f_x, 0, c_x],
        [0, f_y, c_y],
        [0, 0, 1]
    ])

    return K
