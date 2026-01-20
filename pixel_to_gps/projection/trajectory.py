"""Trajectory processor for converting pixel tracks to world coordinates."""

import numpy as np
import pandas as pd
import math
from collections import deque

from ..config import POSITION_BUFFER_SIZE, VELOCITY_BUFFER_SIZE, RAD2DEG
from .pixel_to_world import pixel_to_world_coordinates, compute_world_velocity


def create_rolling_buffer(size):
    """
    Create a rolling average buffer for 3D vectors using deque with O(1) incremental averaging.

    Args:
        size (int): Size of the rolling buffer

    Returns:
        tuple: (update_func, get_value_func) where:
            - update_func(new_value): Updates buffer with new 3D vector and returns new average
            - get_value_func(): Returns current average
    """
    buffer = deque(maxlen=size)
    avg = np.zeros(3)
    running_sum = np.zeros(3)

    def update(new_value):
        nonlocal avg, running_sum

        # Update running sum incrementally for O(1) complexity
        if len(buffer) == size:
            # Buffer is full - remove oldest value from sum
            running_sum -= buffer[0]

        buffer.append(new_value)
        running_sum += new_value

        # Compute average from running sum (O(1) instead of O(n))
        avg = running_sum / len(buffer)
        return avg

    def get_value():
        return avg

    return update, get_value


class TrajectoryProcessor:
    """
    Processes pixel tracks into world coordinate trajectories.

    This class encapsulates the logic for converting pixel-space detections
    into world-space coordinates, computing velocities, and maintaining
    temporal smoothing buffers.
    """

    def __init__(self, camera_extrinsics, camera_intrinsic, camera_intrinsic_inv, video_fps, gps_reference):
        """
        Initialize the trajectory processor.

        Args:
            camera_extrinsics (np.ndarray): Array of 4x4 extrinsic matrices (one per frame)
            camera_intrinsic (np.ndarray): 3x3 camera intrinsic matrix
            camera_intrinsic_inv (np.ndarray): Pre-computed inverse of camera_intrinsic
            video_fps (float): Video frames per second
            gps_reference (list): Reference GPS coordinates [latitude, longitude, altitude]
        """
        self.camera_extrinsics = camera_extrinsics
        self.camera_intrinsic = camera_intrinsic
        self.camera_intrinsic_inv = camera_intrinsic_inv
        self.time_per_frame_ms = 1000.0 / video_fps
        self.gps_reference = gps_reference

    def process_target(self, target_id, pixel_tracks, species_label):
        """
        Process a single target's pixel tracks into world coordinates.

        Args:
            target_id (int): Unique identifier for this target
            pixel_tracks (pd.DataFrame): DataFrame with columns [x, y, frame]
            species_label (str): Species classification label for this target

        Returns:
            list: List of dictionaries containing trajectory data for each detection
        """
        # Initialize temporal averaging buffers for this target
        # Isolated per-target to prevent cross-contamination between tracks
        update_position, get_position = create_rolling_buffer(POSITION_BUFFER_SIZE)
        update_velocity, get_velocity = create_rolling_buffer(VELOCITY_BUFFER_SIZE)

        trajectory_records = []

        # Convert to numpy arrays for faster iteration (avoid DataFrame .iloc overhead)
        pixel_x = pixel_tracks['x'].values
        pixel_y = pixel_tracks['y'].values
        frame_nums = pixel_tracks['frame'].values
        n_detections = len(pixel_tracks)

        # Process all detections for this target
        for i in range(n_detections):
            # Get current detection
            pixel_pos = np.array([pixel_x[i], pixel_y[i]])
            frame_num = frame_nums[i]

            # Get next detection info if available (for velocity computation)
            next_detection = None
            if i < n_detections - 1:
                next_detection = {
                    'pixel_pos': np.array([pixel_x[i + 1], pixel_y[i + 1]]),
                    'frame_num': frame_nums[i + 1]
                }

            # Compute world position and velocity (pure function)
            world_pos, world_velocity = self._process_detection(
                pixel_pos=pixel_pos,
                frame_num=frame_num,
                next_detection=next_detection
            )

            # Update temporal buffers
            update_position(world_pos)
            if world_velocity is not None:
                update_velocity(world_velocity)

            # Get temporally averaged values
            avg_position = get_position()
            avg_velocity = get_velocity()

            # Compute heading angle from averaged velocity
            avg_angle = math.atan2(avg_velocity[1], avg_velocity[0]) * RAD2DEG

            # Create trajectory record
            trajectory_record = self._create_trajectory_record(
                pixel_pos=pixel_pos,
                frame_num=frame_num,
                avg_position=avg_position,
                avg_velocity=avg_velocity,
                avg_angle=avg_angle,
                target_id=target_id,
                species_label=species_label
            )

            trajectory_records.append(trajectory_record)

        return trajectory_records

    def _process_detection(self, pixel_pos, frame_num, next_detection):
        """
        Pure computation function: converts pixel coordinates to world coordinates.

        Args:
            pixel_pos (np.ndarray): Pixel position [x, y]
            frame_num (int): Frame number for camera extrinsic lookup
            next_detection (dict): Next detection info for velocity computation, or None if last frame
                                  Contains 'pixel_pos' and 'frame_num'

        Returns:
            tuple: (world_pos, world_velocity) where:
                - world_pos (np.ndarray): 3D world position [x, y, z]
                - world_velocity (np.ndarray or None): 3D velocity vector, or None if no next frame
        """
        # Convert pixel position to world coordinates
        world_pos = pixel_to_world_coordinates(
            pixel_pos,
            self.camera_extrinsics[frame_num],
            self.camera_intrinsic,
            self.camera_intrinsic_inv
        )

        # Compute velocity using forward differences (current to next frame)
        world_velocity = None
        if next_detection is not None:
            next_world_pos = pixel_to_world_coordinates(
                next_detection['pixel_pos'],
                self.camera_extrinsics[next_detection['frame_num']],
                self.camera_intrinsic,
                self.camera_intrinsic_inv
            )

            # Compute velocity vector from current to next position
            world_velocity = compute_world_velocity(
                next_world_pos,
                world_pos,
                self.time_per_frame_ms
            )

        return world_pos, world_velocity

    def _create_trajectory_record(
        self,
        pixel_pos,
        frame_num,
        avg_position,
        avg_velocity,
        avg_angle,
        target_id,
        species_label
    ):
        """
        Create a trajectory record dictionary from computed values.

        Args:
            pixel_pos (np.ndarray): Original pixel position [x, y]
            frame_num (int): Frame number
            avg_position (np.ndarray): Averaged world position [x, y, z]
            avg_velocity (np.ndarray): Averaged velocity vector [vx, vy, vz]
            avg_angle (float): Heading angle in degrees
            target_id (int): Target identifier
            species_label (str): Species classification

        Returns:
            dict: Trajectory record with all required fields
        """
        return {
            'avg_pos_x': avg_position[0],
            'avg_pos_y': avg_position[1],
            'avg_vel_x': avg_velocity[0],
            'avg_vel_y': avg_velocity[1],
            'avg_vel': np.linalg.norm(avg_velocity),
            'avg_pixel_pos_x': pixel_pos[0],
            'avg_pixel_pos_y': pixel_pos[1],
            'angle': avg_angle,
            'frame': frame_num,
            'target_id': target_id,
            'species_label': species_label,
            'ref_latitude': self.gps_reference[0],
            'ref_longitude': self.gps_reference[1],
            'ref_altitude': self.gps_reference[2]
        }


def prepare_pixel_tracks(target_df: pd.DataFrame):
    """Prepare pixel track data for world coordinate conversion.

    Args:
        target_df: DataFrame already filtered to a single target_id with columns: id, x, y, frame, label

    Returns:
        tracks_df: DataFrame with columns x, y, frame (one row per detection)
    """
    # Return DataFrame with only the columns needed for tracking
    # Round x, y to integers to match original Particle class behavior
    tracks_df = target_df[['x', 'y', 'frame']].copy()
    tracks_df['x'] = tracks_df['x'].round().astype(int)
    tracks_df['y'] = tracks_df['y'].round().astype(int)
    tracks_df['frame'] = tracks_df['frame'].astype(int)

    return tracks_df


def extract_species_label(target_rows: pd.DataFrame) -> str:
    """
    Extract species label from target rows using majority voting.

    Args:
        target_rows: DataFrame rows for a single target

    Returns:
        str: Species label (most common label, or "Unknown" if unavailable)
    """
    if 'label' not in target_rows.columns or len(target_rows) == 0:
        return "Unknown"

    mode_result = target_rows['label'].mode()
    return mode_result.iloc[0] if len(mode_result) > 0 else "Unknown"
