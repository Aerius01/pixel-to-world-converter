"""Ground-plane homography computation and bounding box projection."""

import numpy as np
from numpy import ndarray

from ..config import RAY_PARALLEL_TOLERANCE


def compute_ground_homography(
    K: ndarray,
    extrinsic: ndarray,
    ground_z: float = 0.0,
    K_inv: ndarray | None = None,
) -> ndarray:
    """
    Compute 3x3 homography mapping pixel coordinates to ENU world coordinates on a flat ground plane.

    Derived from the same ray-plane intersection used in pixel_to_world_coordinates:
    shoots a ray through each pixel, intersects it with the plane z = ground_z, and
    encodes the full transform as a projective matrix H such that:
        [X, Y, w]^T = H @ [u, v, 1]^T,  world_x = X/w, world_y = Y/w

    Args:
        K: (3, 3) camera intrinsic matrix
        extrinsic: (4, 4) camera extrinsic matrix [R|t; 0|1] in ENU frame
        ground_z: altitude of the ground plane in meters (default 0.0 = sea level)
        K_inv: (3, 3) inverse of K; computed internally if not provided

    Returns:
        ndarray of shape (3, 3) — the ground-plane homography
    """
    M = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    if K_inv is None:
        K_inv = np.linalg.inv(K)
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    A = R.T @ M @ K_inv
    dz = ground_z - t[2]
    return np.stack([
        t[0] * A[2] + dz * A[0],
        t[1] * A[2] + dz * A[1],
        A[2],
    ])


def project_bbox(cx: float, cy: float, w: float, h: float, H: ndarray) -> ndarray:
    """
    Project a pixel-space bounding box through a ground-plane homography.

    Args:
        cx: bounding box center x in pixels
        cy: bounding box center y in pixels
        w: bounding box width in pixels
        h: bounding box height in pixels
        H: (3, 3) ground-plane homography from compute_ground_homography

    Returns:
        ndarray of shape (4, 2) — world (East, North) coordinates of the four corners
        in ENU meters, ordered: top-left, top-right, bottom-right, bottom-left
    """
    half_w, half_h = w / 2.0, h / 2.0
    corners = np.array([
        [cx - half_w, cy - half_h],
        [cx + half_w, cy - half_h],
        [cx + half_w, cy + half_h],
        [cx - half_w, cy + half_h],
    ], dtype=float)
    corners_h = np.column_stack([corners, np.ones(4)])
    world_h = (H @ corners_h.T).T
    denominators = world_h[:, 2]
    if np.any(np.abs(denominators) < RAY_PARALLEL_TOLERANCE) or not (
        np.all(denominators > 0) or np.all(denominators < 0)
    ):
        raise ValueError(
            "Degenerate projection: bounding box corners produce rays that are parallel to "
            "or straddle the ground plane. Check camera orientation."
        )
    return world_h[:, :2] / world_h[:, 2:3]
