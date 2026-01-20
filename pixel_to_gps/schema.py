"""Data structure schemas and validation for GPS converter input/output."""

from dataclasses import dataclass
from typing import List
import pandas as pd


@dataclass
class DroneLogSchema:
    """Schema for drone sensor log CSV files."""

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
        """Return list of all required columns."""
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
        Validate that DataFrame contains all required columns.

        Args:
            df: DataFrame to validate

        Returns:
            List of missing column names (empty if valid)
        """
        required = self.get_required_columns()
        return [col for col in required if col not in df.columns]


@dataclass
class ImageTracksSchema:
    """Schema for image tracking CSV files."""

    id_col: str = 'id'
    x_col: str = 'x'
    y_col: str = 'y'
    frame_col: str = 'frame'
    label_col: str = 'label'

    def get_required_columns(self) -> List[str]:
        """Return list of all required columns."""
        return [
            self.id_col,
            self.x_col,
            self.y_col,
            self.frame_col,
            self.label_col
        ]

    def validate(self, df: pd.DataFrame) -> List[str]:
        """
        Validate that DataFrame contains all required columns.

        Args:
            df: DataFrame to validate

        Returns:
            List of missing column names (empty if valid)
        """
        required = self.get_required_columns()
        return [col for col in required if col not in df.columns]


@dataclass
class WorldTrajectorySchema:
    """Schema for output world trajectory CSV files."""

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
        """Return list of all output columns."""
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


# Default schema instances
DRONE_LOG_SCHEMA = DroneLogSchema()
IMAGE_TRACKS_SCHEMA = ImageTracksSchema()
WORLD_TRAJECTORY_SCHEMA = WorldTrajectorySchema()
