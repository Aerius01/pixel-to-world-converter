"""
CSV schema definitions for input and output data formats.

This module defines the required column names and validation logic for:
- Drone sensor logs (GPS, IMU, gimbal data)
- Image tracking data (pixel coordinates and labels)
- World trajectory output (GPS-converted coordinates and velocities)

These schemas ensure data consistency across the pipeline and provide
clear error messages when required columns are missing.
"""

from dataclasses import dataclass
from typing import List
import pandas as pd


@dataclass
class DroneLogSchema:
    """
    Schema definition for drone sensor CSV files.

    Drone logs must contain synchronized GPS, IMU velocity, and gimbal orientation
    data. All timestamps are in milliseconds, positions in WGS84 geodetic coordinates,
    velocities in meters per second (NED frame), and gimbal angles in raw units
    (which get divided by CONVERSION_FACTOR=10.0 during processing).

    Attributes:
        time_col: Timestamp in milliseconds since epoch
        lat_col: Latitude in decimal degrees (WGS84)
        lon_col: Longitude in decimal degrees (WGS84)
        alt_col: Altitude in meters above sea level
        vel_x_col: Velocity in North direction (NED frame, meters/second)
        vel_y_col: Velocity in East direction (NED frame, meters/second)
        vel_z_col: Velocity in Down direction (NED frame, meters/second)
        gimbal_pitch_col: Raw gimbal pitch angle (needs CONVERSION_FACTOR and offset)
        gimbal_roll_col: Raw gimbal roll angle (needs CONVERSION_FACTOR)
        gimbal_yaw_col: Raw gimbal yaw angle (needs CONVERSION_FACTOR)
    """
    time_col: str = 'time(millisecond)'
    lat_col: str = 'latitude'
    lon_col: str = 'longitude'
    alt_col: str = 'altitude(m)'
    vel_x_col: str = 'velocityX(mps)'
    vel_y_col: str = 'velocityY(mps)'
    vel_z_col: str = 'velocityZ(mps)'
    gimbal_pitch_col: str = 'gimbalPitchRaw'
    gimbal_roll_col: str = 'gimbalRollRaw'
    gimbal_yaw_col: str = 'gimbalYawRaw'

    def get_required_columns(self) -> List[str]:
        """
        Return list of all required column names for drone log CSV.

        Returns:
            List of column name strings
        """
        return [
            self.time_col,
            self.lat_col,
            self.lon_col,
            self.alt_col,
            self.vel_x_col,
            self.vel_y_col,
            self.vel_z_col,
            self.gimbal_pitch_col,
            self.gimbal_roll_col,
            self.gimbal_yaw_col
        ]

    def validate(self, df: pd.DataFrame) -> List[str]:
        """
        Check if DataFrame contains all required columns.

        Args:
            df: DataFrame to validate (typically just the header row)

        Returns:
            List of missing column names (empty list if all present)
        """
        required = self.get_required_columns()
        return [col for col in required if col not in df.columns]


@dataclass
class ImageTracksSchema:
    """
    Schema definition for image tracking CSV files.

    Image tracks contain pixel coordinates of detected objects across video frames.
    Each row represents one detection with its pixel location, frame number,
    unique trajectory ID, and species classification.

    Attributes:
        id_col: Unique trajectory/target identifier (same ID = same tracked object)
        x_col: Pixel x-coordinate (horizontal, 0 = left edge)
        y_col: Pixel y-coordinate (vertical, 0 = top edge)
        frame_col: Frame number (0-indexed)
        label_col: Species classification label (e.g., "marlin", "shark")
    """
    id_col: str = 'id'
    x_col: str = 'x'
    y_col: str = 'y'
    frame_col: str = 'frame'
    label_col: str = 'label'

    def get_required_columns(self) -> List[str]:
        """
        Return list of all required column names for image tracks CSV.

        Returns:
            List of column name strings
        """
        return [
            self.id_col,
            self.x_col,
            self.y_col,
            self.frame_col,
            self.label_col
        ]

    def validate(self, df: pd.DataFrame) -> List[str]:
        """
        Check if DataFrame contains all required columns.

        Args:
            df: DataFrame to validate (typically just the header row)

        Returns:
            List of missing column names (empty list if all present)
        """
        required = self.get_required_columns()
        return [col for col in required if col not in df.columns]


@dataclass
class WorldTrajectorySchema:
    """
    Schema definition for world trajectory output CSV files.

    Output trajectories contain smoothed world positions and velocities in the
    ENU (East-North-Up) coordinate system, relative to a GPS reference point.
    Each row represents one detection after GPS conversion with temporal averaging.

    Coordinate System:
        - Origin: First GPS position from drone log (ref_latitude, ref_longitude)
        - X-axis: East (meters)
        - Y-axis: North (meters)
        - Z-axis: Up (altitude relative to sea level = 0)

    Attributes:
        avg_pos_x_col: Smoothed world position X (East, meters)
        avg_pos_y_col: Smoothed world position Y (North, meters)
        avg_vel_x_col: Smoothed velocity X component (East, meters/second)
        avg_vel_y_col: Smoothed velocity Y component (North, meters/second)
        avg_vel_col: Smoothed velocity magnitude (meters/second)
        avg_pixel_pos_x_col: Original pixel x-coordinate
        avg_pixel_pos_y_col: Original pixel y-coordinate
        angle_col: Heading angle in degrees (0 = East, 90 = North)
        frame_col: Frame number
        target_id_col: Trajectory identifier (matches input 'id')
        species_label_col: Species classification
        ref_latitude_col: GPS reference latitude (origin)
        ref_longitude_col: GPS reference longitude (origin)
        ref_altitude_col: GPS reference altitude (always 0 at sea level)
    """
    avg_pos_x_col: str = 'avg_pos_x'
    avg_pos_y_col: str = 'avg_pos_y'
    avg_vel_x_col: str = 'avg_vel_x'
    avg_vel_y_col: str = 'avg_vel_y'
    avg_vel_col: str = 'avg_vel'
    avg_pixel_pos_x_col: str = 'avg_pixel_pos_x'
    avg_pixel_pos_y_col: str = 'avg_pixel_pos_y'
    angle_col: str = 'angle'
    frame_col: str = 'frame'
    target_id_col: str = 'target_id'
    species_label_col: str = 'species_label'
    ref_latitude_col: str = 'ref_latitude'
    ref_longitude_col: str = 'ref_longitude'
    ref_altitude_col: str = 'ref_altitude'

    def get_output_columns(self) -> List[str]:
        """
        Return list of all output column names for world trajectory CSV.

        Returns:
            List of column name strings in the order they appear in output
        """
        return [
            self.avg_pos_x_col,
            self.avg_pos_y_col,
            self.avg_vel_x_col,
            self.avg_vel_y_col,
            self.avg_vel_col,
            self.avg_pixel_pos_x_col,
            self.avg_pixel_pos_y_col,
            self.angle_col,
            self.frame_col,
            self.target_id_col,
            self.species_label_col,
            self.ref_latitude_col,
            self.ref_longitude_col,
            self.ref_altitude_col
        ]


# Global schema instances used throughout the pipeline
DRONE_LOG_SCHEMA = DroneLogSchema()
IMAGE_TRACKS_SCHEMA = ImageTracksSchema()
WORLD_TRAJECTORY_SCHEMA = WorldTrajectorySchema()
