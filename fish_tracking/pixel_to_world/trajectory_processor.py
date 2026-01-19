"""
Trajectory processor for converting pixel tracks to world coordinates.

This module provides the TrajectoryProcessor class which handles the conversion
of pixel-space detections into world-space trajectories with velocity estimation.
"""

import numpy as np
import pandas as pd
import math

from fish_tracking.common.globals import *
from fish_tracking.pixel_to_world.compute_statistics import pixel_to_world_coordinates, compute_world_velocity


def create_rolling_buffer(size):
    """
    Create a rolling average buffer for 3D vectors.
    
    Args:
        size (int): Size of the rolling buffer
    
    Returns:
        tuple: (update_func, get_value_func) where:
            - update_func(new_value): Updates buffer with new 3D vector and returns new average
            - get_value_func(): Returns current average
    """
    buffer = np.zeros((size, 3))
    idx = [0]  # Mutable container for closure
    avg = np.zeros(3)
    
    def update(new_value):
        nonlocal avg
        buffer[idx[0]] = new_value
        idx[0] = (idx[0] + 1) % size
        avg = buffer.mean(axis=0)
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
    
    def __init__(self, camera_extrinsics, camera_intrinsic, video_fps, gps_reference):
        """
        Initialize the trajectory processor.
        
        Args:
            camera_extrinsics (np.ndarray): Array of 4x4 extrinsic matrices (one per frame)
            camera_intrinsic (np.ndarray): 3x3 camera intrinsic matrix
            video_fps (float): Video frames per second
            gps_reference (list): Reference GPS coordinates [latitude, longitude, altitude]
        """
        self.camera_extrinsics = camera_extrinsics
        self.camera_intrinsic = camera_intrinsic
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
        
        # Process all detections for this target
        for i in range(len(pixel_tracks)):
            # Get current detection
            detection_row = pixel_tracks.iloc[i]
            pixel_pos = np.array([detection_row['x'], detection_row['y']])
            frame_num = detection_row['frame']
            
            # Get next detection info if available (for velocity computation)
            next_detection = None
            if i < len(pixel_tracks) - 1:
                next_row = pixel_tracks.iloc[i + 1]
                next_detection = {
                    'pixel_pos': np.array([next_row['x'], next_row['y']]),
                    'frame_num': next_row['frame']
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
            self.camera_intrinsic
        )
        
        # Compute velocity using forward differences (current to next frame)
        world_velocity = None
        if next_detection is not None:
            next_world_pos = pixel_to_world_coordinates(
                next_detection['pixel_pos'],
                self.camera_extrinsics[next_detection['frame_num']],
                self.camera_intrinsic
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
            'vel_pixel_pos_x': 0.0,  # Not used in current implementation
            'vel_pixel_pos_y': 0.0,  # Not used in current implementation
            'angle': avg_angle,
            'frame': frame_num,
            'target_id': target_id,
            'species_label': species_label,
            'ref_latitude': self.gps_reference[0],
            'ref_longitude': self.gps_reference[1],
            'ref_altitude': self.gps_reference[2]
        }
