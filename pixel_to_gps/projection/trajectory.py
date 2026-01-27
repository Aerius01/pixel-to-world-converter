"""
Trajectory processing with temporal averaging and smoothing.

This module handles per-target trajectory processing, converting sequences of pixel
detections into smoothed world trajectories. Each target (unique ID) is processed
independently with its own temporal buffers to prevent cross-contamination.

Key Features:
- Rolling buffers for position and velocity smoothing
- Per-target state isolation (no mixing between different fish)
- Functional design for deterministic output
- Species label extraction via majority voting
"""

import numpy as np
import pandas as pd
import math
from collections import deque

from ..config import POSITION_BUFFER_SIZE, VELOCITY_BUFFER_SIZE, RAD2DEG
from .pixel_to_world import pixel_to_world_coordinates, compute_world_velocity


def create_rolling_buffer(size):
    """
    Create a closure-based rolling buffer for temporal averaging.

    Returns two functions (update, get_value) that share an internal state:
    - buffer: deque with maxlen=size (automatically discards old values)
    - running_sum: Efficiently tracks sum without recomputing each time
    - avg: Current average value

    This design avoids class overhead while maintaining encapsulated state.

    Args:
        size: Maximum buffer size (number of values to average)

    Returns:
        tuple: (update_fn, get_value_fn)
            - update_fn(new_value): Add value to buffer and return new average
            - get_value_fn(): Return current average without modification
    """
    # Internal state (shared via closure)
    buffer = deque(maxlen=size)
    avg = np.zeros(3)
    running_sum = np.zeros(3)

    def update(new_value):
        """Add new value to buffer and update running average."""
        nonlocal avg, running_sum

        # If buffer is full, subtract the value that will be evicted
        if len(buffer) == size:
            running_sum -= buffer[0]

        # Add new value (deque automatically evicts oldest if full)
        buffer.append(new_value)
        running_sum += new_value

        # Compute average over current buffer size
        avg = running_sum / len(buffer)
        return avg

    def get_value():
        """Return current average without modification."""
        return avg

    return update, get_value


class TrajectoryProcessor:
    """
    Process pixel tracks for multiple targets into smoothed world trajectories.

    This class maintains camera parameters and GPS reference, then processes each
    target's detections independently. Each target gets its own temporal buffers
    to prevent mixing state between different tracked objects.

    Processing Flow:
        For each target:
            1. Create fresh position and velocity buffers
            2. For each detection:
                a. Convert pixel to world coordinates
                b. Compute velocity from consecutive positions
                c. Update temporal buffers
                d. Create trajectory record with averaged values

    Design Philosophy:
        - Pure functional processing in _process_detection (no side effects)
        - Per-target buffer isolation (prevents cross-contamination)
        - Deterministic output (same input always produces same output)

    Attributes:
        camera_extrinsics: ndarray of shape (num_frames, 4, 4) with per-frame extrinsic matrices
        camera_intrinsic: 3x3 intrinsic matrix K
        camera_intrinsic_inv: 3x3 inverse intrinsic matrix K^{-1}
        time_per_frame_ms: Time between frames in milliseconds
        gps_reference: List [latitude, longitude, altitude] of reference origin
    """
    def __init__(self, camera_extrinsics, camera_intrinsic, camera_intrinsic_inv, video_fps, gps_reference):
        """
        Initialize trajectory processor with camera parameters.

        Args:
            camera_extrinsics: ndarray of shape (num_frames, 4, 4) with per-frame extrinsic matrices
            camera_intrinsic: 3x3 intrinsic matrix K
            camera_intrinsic_inv: 3x3 inverse intrinsic matrix K^{-1} (pre-inverted for efficiency)
            video_fps: Video frame rate (frames per second)
            gps_reference: List [latitude, longitude, altitude] of GPS origin
        """
        self.camera_extrinsics = camera_extrinsics
        self.camera_intrinsic = camera_intrinsic
        self.camera_intrinsic_inv = camera_intrinsic_inv
        self.time_per_frame_ms = 1000.0 / video_fps
        self.gps_reference = gps_reference

    def process_target(self, target_id, pixel_tracks, species_label):
        """
        Process all detections for a single target into smoothed trajectory records.

        Creates fresh temporal buffers for this target, then iterates through all
        detections in chronological order. Each detection is converted to world
        coordinates, velocity is computed, and values are smoothed via rolling buffers.

        Buffer Configuration:
            - Position buffer: POSITION_BUFFER_SIZE frames (typically 1 = no smoothing)
            - Velocity buffer: VELOCITY_BUFFER_SIZE frames (typically 60 = 2 seconds at 30fps)

        Args:
            target_id: Unique identifier for this tracked object
            pixel_tracks: DataFrame with columns [x, y, frame] containing detections
            species_label: Species classification string (e.g., "marlin", "shark")

        Returns:
            List of trajectory record dictionaries, one per detection
        """
        # Create fresh buffers for this target (per-target isolation)
        update_position, get_position = create_rolling_buffer(POSITION_BUFFER_SIZE)
        update_velocity, get_velocity = create_rolling_buffer(VELOCITY_BUFFER_SIZE)

        trajectory_records = []

        # Pre-extract arrays for efficiency (avoid repeated DataFrame access)
        pixel_x = pixel_tracks['x'].values
        pixel_y = pixel_tracks['y'].values
        frame_nums = pixel_tracks['frame'].values
        n_detections = len(pixel_tracks)

        # Process each detection in chronological order
        for i in range(n_detections):
            pixel_pos = np.array([pixel_x[i], pixel_y[i]])
            frame_num = frame_nums[i]

            # Prepare next detection for velocity computation
            next_detection = None
            if i < n_detections - 1:
                next_detection = {
                    'pixel_pos': np.array([pixel_x[i + 1], pixel_y[i + 1]]),
                    'frame_num': frame_nums[i + 1]
                }

            # Convert pixel to world and compute velocity (pure function)
            world_pos, world_velocity = self._process_detection(
                pixel_pos=pixel_pos,
                frame_num=frame_num,
                next_detection=next_detection
            )

            # Update temporal buffers with new values
            update_position(world_pos)
            if world_velocity is not None:
                update_velocity(world_velocity)

            # Get smoothed values from buffers
            avg_position = get_position()
            avg_velocity = get_velocity()

            # Compute heading angle in degrees (0=East, 90=North)
            avg_angle = math.atan2(avg_velocity[1], avg_velocity[0]) * RAD2DEG

            # Create trajectory record for this detection
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
        Convert pixel detection to world position and velocity (pure function).

        This function has no side effects and is deterministic - same inputs always
        produce same outputs. It performs the core pixel-to-world projection and
        optionally computes velocity from consecutive detections.

        Args:
            pixel_pos: ndarray of shape (2,) with [x, y] pixel coordinates
            frame_num: Frame number (integer)
            next_detection: Dictionary with {'pixel_pos', 'frame_num'} or None if last detection

        Returns:
            tuple: (world_pos, world_velocity)
                - world_pos: ndarray of shape (3,) with [x, y, z] in meters (ENU)
                - world_velocity: ndarray of shape (3,) with [vx, vy, vz] in m/s, or None if no next detection
        """
        # Convert current pixel to world coordinates
        world_pos = pixel_to_world_coordinates(
            pixel_pos,
            self.camera_extrinsics[frame_num],
            self.camera_intrinsic,
            self.camera_intrinsic_inv
        )

        # Compute velocity if next detection exists
        world_velocity = None
        if next_detection is not None:
            # Convert next pixel to world coordinates
            next_world_pos = pixel_to_world_coordinates(
                next_detection['pixel_pos'],
                self.camera_extrinsics[next_detection['frame_num']],
                self.camera_intrinsic,
                self.camera_intrinsic_inv
            )

            # Compute velocity as finite difference
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
        Create trajectory record dictionary from processed detection data.

        This function packages all relevant information about a detection into a
        dictionary that will become one row in the output CSV.

        Args:
            pixel_pos: Original pixel position [x, y]
            frame_num: Frame number
            avg_position: Smoothed world position [x, y, z]
            avg_velocity: Smoothed velocity [vx, vy, vz]
            avg_angle: Heading angle in degrees
            target_id: Target identifier
            species_label: Species classification

        Returns:
            Dictionary with keys matching WORLD_TRAJECTORY_SCHEMA output columns
        """
        return {
            'avg_pos_x': avg_position[0],      # Smoothed world X (East, meters)
            'avg_pos_y': avg_position[1],      # Smoothed world Y (North, meters)
            'avg_vel_x': avg_velocity[0],      # Smoothed velocity X (East, m/s)
            'avg_vel_y': avg_velocity[1],      # Smoothed velocity Y (North, m/s)
            'avg_vel': np.linalg.norm(avg_velocity),  # Velocity magnitude (m/s)
            'avg_pixel_pos_x': pixel_pos[0],   # Original pixel x
            'avg_pixel_pos_y': pixel_pos[1],   # Original pixel y
            'angle': avg_angle,                # Heading (degrees, 0=East, 90=North)
            'frame': frame_num,                # Frame number
            'target_id': target_id,            # Target identifier
            'species_label': species_label,    # Species classification
            'ref_latitude': self.gps_reference[0],   # GPS reference latitude
            'ref_longitude': self.gps_reference[1],  # GPS reference longitude
            'ref_altitude': self.gps_reference[2]    # GPS reference altitude (0 at sea level)
        }


def prepare_pixel_tracks(target_df: pd.DataFrame):
    """
    Prepare pixel tracks DataFrame for trajectory processing.

    Extracts required columns and converts to appropriate dtypes:
    - x, y: Rounded to nearest integer pixel
    - frame: Integer frame number

    Args:
        target_df: DataFrame with at least columns [x, y, frame]

    Returns:
        DataFrame with columns [x, y, frame] as integers, sorted by frame
    """
    # Extract required columns
    tracks_df = target_df[['x', 'y', 'frame']].copy()

    # Round pixel coordinates to nearest integer
    tracks_df['x'] = tracks_df['x'].round().astype(int)
    tracks_df['y'] = tracks_df['y'].round().astype(int)
    tracks_df['frame'] = tracks_df['frame'].astype(int)

    return tracks_df


def extract_species_label(target_rows: pd.DataFrame) -> str:
    """
    Extract species label for a target using majority voting.

    If a target has multiple different labels across its detections (due to
    classification uncertainty), selects the most common label (mode). Falls
    back to "Unknown" if no labels are present.

    Args:
        target_rows: DataFrame with 'label' column containing species classifications

    Returns:
        Most common species label as string, or "Unknown" if unavailable
    """
    # Check if label column exists and has data
    if 'label' not in target_rows.columns or len(target_rows) == 0:
        return "Unknown"

    # Compute mode (most common value)
    mode_result = target_rows['label'].mode()

    # Return mode if it exists, otherwise "Unknown"
    return mode_result.iloc[0] if len(mode_result) > 0 else "Unknown"
