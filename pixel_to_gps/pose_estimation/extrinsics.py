"""Camera extrinsic and intrinsic matrix computation."""

import numpy as np
from scipy.spatial.transform import Rotation as Rot


def compute_extrinsic_matrix(angles, camera_position):
    """
    Compute the extrinsic matrix from gimbal angles and camera position.

    Args:
        angles (array-like): Gimbal angles in degrees where:
            - angles[0]: roll
            - angles[1]: pitch
            - angles[2]: yaw
        camera_position (array-like): Camera position [x, y, z] in meters

    Returns:
        np.ndarray: 4x4 extrinsic matrix
    """
    rotation = Rot.from_euler('xyz', [np.deg2rad(angles[0]), np.deg2rad(angles[1]), -np.deg2rad(angles[2])])
    rotation_matrix = rotation.as_matrix()
    # center world coordinate system at initial drone pose
    translation = np.array(camera_position).reshape(3, 1)
    # Combine rotation_matrix and translation into a 3x4 matrix
    rotation_translation = np.hstack((rotation_matrix, translation))
    # Create the 4x4 extrinsic matrix
    extrinsic = np.vstack((rotation_translation, [0, 0, 0, 1]))

    return extrinsic


def compute_intrinsic_matrix(focal_length, pixel_size, image_width, image_height):
    """
    Compute the camera intrinsic matrix that maps 3D camera coordinates to 2D pixel coordinates.

    The intrinsic matrix encodes the camera's internal optical properties (lens and sensor),
    independent of where the camera is positioned or pointing in the world.

    Args:
        focal_length (float): Physical focal length of the lens in meters
        pixel_size (tuple): Physical size of one pixel (width, height) in meters
        image_width (int): Image width in pixels
        image_height (int): Image height in pixels

    Returns:
        np.ndarray: 3x3 intrinsic matrix K with structure:
            [[f_x,   0, c_x],
             [  0, f_y, c_y],
             [  0,   0,   1]]

            where:
            - f_x, f_y: focal lengths in pixel units (controls zoom/magnification)
            - c_x, c_y: principal point (where optical axis hits image plane)
            - zeros: no skew between pixel axes (pixels are rectangular)
            - 1: homogeneous coordinate for projection math
    """
    # Convert focal length from physical units (meters) to pixel units
    # This tells us how many pixels correspond to the camera's focal length
    f_x = focal_length / pixel_size[0]  # Horizontal focal length in pixels
    f_y = focal_length / pixel_size[1]  # Vertical focal length in pixels

    # The principal point is where the camera's optical axis intersects the image plane.
    # For most cameras, this is very close to the image center.
    c_x = image_width / 2   # Horizontal center
    c_y = image_height / 2  # Vertical center

    # Construct the 3x3 intrinsic matrix K
    # This matrix is used in the pinhole camera model to project 3D points to 2D pixels
    K = np.array([
        [f_x, 0, c_x],  # Row 1: x-axis scaling and offset
        [0, f_y, c_y],  # Row 2: y-axis scaling and offset
        [0, 0, 1]       # Row 3: homogeneous coordinates
    ])

    return K
