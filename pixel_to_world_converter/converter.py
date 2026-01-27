"""
Main conversion interface for pixel-to-world coordinate transformation.

This module provides the programmatic API for converting pixel coordinates from
drone video tracking to world GPS coordinates. It handles validation, video property
extraction, and orchestrates the conversion pipeline.
"""

import os
import cv2
import pandas as pd

from .pipeline import convert_pixel_to_world
from .schema import DRONE_LOG_SCHEMA, IMAGE_TRACKS_SCHEMA


class PixelToWorldError(Exception):
    """Base exception for all pixel-to-world conversion errors."""
    pass


class ValidationError(PixelToWorldError):
    """Raised when input file validation fails (missing files, permissions, etc.)."""
    pass


class VideoError(PixelToWorldError):
    """Raised when video file cannot be opened or read."""
    pass


class DataLoadError(PixelToWorldError):
    """Raised when CSV data is missing required columns or contains invalid values."""
    pass


def validate_file_exists(filepath, file_description):
    """
    Verify that a file exists and is readable.

    Args:
        filepath: Path to the file to validate
        file_description: Human-readable description for error messages

    Raises:
        ValidationError: If file does not exist or is not readable
    """
    if not os.path.isfile(filepath):
        raise ValidationError(f"{file_description} not found: {filepath}")
    if not os.access(filepath, os.R_OK):
        raise ValidationError(f"{file_description} is not readable: {filepath}")


def validate_output_path(output_path):
    """
    Ensure the output directory exists, creating it if necessary.

    Args:
        output_path: Full path where output file will be written

    Raises:
        ValidationError: If directory cannot be created due to permissions or other OS errors
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            raise ValidationError(f"Cannot create output directory {output_dir}: {e}")


def validate_csv_schema(csv_path, schema, file_description):
    """
    Validate that a CSV file contains all required columns.

    This function reads only the header row (nrows=0) for efficiency, then checks
    for the presence of all required columns defined in the schema.

    Args:
        csv_path: Path to the CSV file to validate
        schema: Schema object with validate() and get_required_columns() methods
        file_description: Human-readable description for error messages

    Raises:
        DataLoadError: If required columns are missing from the CSV
    """
    csv_preview = pd.read_csv(csv_path, nrows=0)
    missing_columns = schema.validate(csv_preview)
    if missing_columns:
        error_msg = f"{file_description} is missing required columns: {', '.join(missing_columns)}\n"
        error_msg += f"Required columns: {', '.join(schema.get_required_columns())}"
        raise DataLoadError(error_msg)


def extract_video_properties(video_path):
    """
    Extract essential properties from a video file using OpenCV.

    Opens the video file and reads metadata properties required for the GPS conversion
    pipeline: frame rate, dimensions, and total frame count. The video is immediately
    closed after reading properties to free resources.

    Args:
        video_path: Path to the video file

    Returns:
        tuple: (fps, width, height, num_frames)
            - fps (float): Frames per second
            - width (int): Frame width in pixels
            - height (int): Frame height in pixels
            - num_frames (int): Total number of frames in the video

    Raises:
        VideoError: If video file cannot be opened by OpenCV
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    return fps, width, height, num_frames


def convert(
    image_tracks_path,
    drone_log_path,
    video_path,
    output_path,
    verbose=True
):
    """
    Convert pixel coordinates from drone video tracking to world GPS coordinates.

    This is the main programmatic interface for the pixel-to-world conversion pipeline.
    It validates inputs, extracts video properties, loads data, and orchestrates the
    full conversion process.

    Args:
        image_tracks_path: Path to CSV file containing pixel coordinates (columns: id, x, y, frame, label)
        drone_log_path: Path to pre-filtered drone sensor data CSV file
        video_path: Path to video file (for extracting FPS and dimensions)
        output_path: Path where output CSV will be written
        verbose: If True, prints progress messages to stdout (default: True)

    Returns:
        pd.DataFrame: Trajectory data with world GPS coordinates

    Raises:
        ValidationError: If input files don't exist or aren't readable
        VideoError: If video file cannot be opened
        DataLoadError: If CSV files are missing required columns or contain invalid data
        PixelToWorldError: For other conversion-related errors

    Example:
        >>> from pixel_to_world_converter import convert
        >>> df = convert(
        ...     image_tracks_path='tracks.csv',
        ...     drone_log_path='drone.csv',
        ...     video_path='video.mp4',
        ...     output_path='world_coords.csv'
        ... )
        >>> print(f"Converted {len(df)} trajectory records")
    """
    def log(message):
        """Helper to conditionally print messages."""
        if verbose:
            print(message)

    log("=" * 80)
    log("GPS CONVERTER - Pixel to World Coordinate Transformation")
    log("=" * 80)

    # Step 1: Validate input files
    log("\n[1/6] Validating input files...")
    validate_file_exists(image_tracks_path, "Image tracks file")
    validate_file_exists(drone_log_path, "Drone log file")
    validate_file_exists(video_path, "Video file")
    validate_output_path(output_path)
    log("      All input files validated successfully")

    # Step 2: Extract video properties
    log("\n[2/6] Extracting video properties...")
    video_fps, video_width, video_height, video_num_frames = extract_video_properties(video_path)
    log(f"      Video: {video_width}x{video_height} @ {video_fps:.2f} fps ({video_num_frames} frames)")

    # Step 3: Validate CSV schemas
    log("\n[3/6] Validating CSV schemas...")
    validate_csv_schema(image_tracks_path, IMAGE_TRACKS_SCHEMA, "Image tracks file")
    validate_csv_schema(drone_log_path, DRONE_LOG_SCHEMA, "Drone log file")
    log("      All required columns present")

    # Step 4: Load data
    log("\n[4/6] Loading CSV data...")
    try:
        image_tracks_df = pd.read_csv(image_tracks_path, usecols=IMAGE_TRACKS_SCHEMA.get_required_columns())
    except Exception as e:
        raise DataLoadError(f"Failed to load image tracks CSV: {e}")
    
    try:
        sensor_data_df = pd.read_csv(drone_log_path, usecols=DRONE_LOG_SCHEMA.get_required_columns(), dtype=float)
    except Exception as e:
        raise DataLoadError(f"Failed to load drone log CSV: {e}")

    num_tracks = len(image_tracks_df)
    num_targets = image_tracks_df['id'].nunique()
    num_sensor_records = len(sensor_data_df)
    log(f"      Loaded {num_tracks} detections across {num_targets} targets")
    log(f"      Loaded {num_sensor_records} sensor records")

    # Step 5: Validate velocity data (required for Kalman filter)
    log("\n[5/6] Validating sensor data completeness...")
    velocity_columns = [DRONE_LOG_SCHEMA.vel_x_col, DRONE_LOG_SCHEMA.vel_y_col, DRONE_LOG_SCHEMA.vel_z_col]
    for col in velocity_columns:
        if sensor_data_df[col].isna().any():
            missing_count = sensor_data_df[col].isna().sum()
            raise DataLoadError(
                f"Drone log contains {missing_count} missing values in {col}. "
                f"All velocity measurements are required for GPS conversion."
            )
    log("      All velocity measurements present (required for Kalman filter)")

    # Step 6: Run conversion pipeline
    log("\n[6/6] Running GPS conversion pipeline...")
    log("      This may take a few minutes depending on video length...")
    try:
        trajectory_df = convert_pixel_to_world(
            image_tracks_df=image_tracks_df,
            sensor_data_df=sensor_data_df,
            video_fps=video_fps,
            video_width=video_width,
            video_height=video_height,
            video_num_frames=video_num_frames
        )
    except Exception as e:
        raise PixelToWorldError(f"Pipeline conversion failed: {e}")

    # Save output
    try:
        trajectory_df.to_csv(output_path, index=False)
    except Exception as e:
        raise PixelToWorldError(f"Failed to write output CSV to {output_path}: {e}")
    
    num_trajectory_records = len(trajectory_df)
    log(f"      Conversion complete! Generated {num_trajectory_records} trajectory records")

    log(f"\n{'=' * 80}")
    log(f"SUCCESS: Output written to {output_path}")
    log(f"{'=' * 80}\n")

    return trajectory_df
