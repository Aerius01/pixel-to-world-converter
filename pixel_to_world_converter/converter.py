"""
Main conversion interface for pixel-to-world coordinate transformation.

This module provides the programmatic API for converting pixel coordinates from
drone video tracking to world GPS coordinates. It handles validation, video property
extraction, and orchestrates the conversion pipeline.
"""

import os
import cv2
import numpy as np
import pandas as pd

from .errors import DataLoadError, PixelToWorldError, ValidationError, VideoError
from .pipeline import convert_pixel_to_world, build_frame_homographies


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
    log("\n[1/4] Validating input files...")
    validate_file_exists(image_tracks_path, "Image tracks file")
    validate_file_exists(drone_log_path, "Drone log file")
    validate_file_exists(video_path, "Video file")
    validate_output_path(output_path)
    log("      All input files validated successfully")

    # Step 2: Extract video properties
    log("\n[2/4] Extracting video properties...")
    video_fps, video_width, video_height, video_num_frames = extract_video_properties(video_path)
    log(f"      Video: {video_width}x{video_height} @ {video_fps:.2f} fps ({video_num_frames} frames)")

    # Step 3: Load data
    log("\n[3/4] Loading CSV data...")
    image_tracks_df = _read_csv(image_tracks_path, "image tracks")
    sensor_data_df = _read_csv(drone_log_path, "drone log")
    log(f"      Loaded {len(image_tracks_df)} detections and {len(sensor_data_df)} sensor records")

    # Step 4: Validate the tables and run the conversion pipeline
    log("\n[4/4] Running GPS conversion pipeline...")
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
    except PixelToWorldError:
        raise
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


def compute_homographies(
    drone_log_path: str,
    video_path: str,
    ground_z: float = 0.0,
) -> np.ndarray:
    """
    Precompute ground-plane homographies for every frame of a drone video.

    Validates inputs, extracts video properties, and runs the pose estimation
    pipeline (sensor alignment, Kalman filter, RTS smoother, camera extrinsics)
    to produce one 3x3 homography per frame. Apply the result with project_bbox()
    for ad-hoc bounding box projection at inference time.

    Args:
        drone_log_path: Path to pre-filtered drone sensor data CSV file
        video_path: Path to video file (for extracting FPS and dimensions)
        ground_z: Altitude of the ground plane in meters (default 0.0 = sea level)

    Returns:
        ndarray of shape (num_frames, 3, 3) — one homography per frame

    Raises:
        ValidationError: If input files don't exist or aren't readable
        VideoError: If video file cannot be opened
        DataLoadError: If CSV file is missing required columns or contains invalid data
        PixelToWorldError: For other conversion-related errors

    Example:
        >>> from pixel_to_world_converter import compute_homographies, project_bbox
        >>> H = compute_homographies('drone.csv', 'video.mp4')
        >>> corners = project_bbox(cx, cy, w, h, H[frame_id])
    """
    validate_file_exists(drone_log_path, "Drone log file")
    validate_file_exists(video_path, "Video file")

    video_fps, video_width, video_height, video_num_frames = extract_video_properties(video_path)

    sensor_data_df = _read_csv(drone_log_path, "drone log")

    try:
        return build_frame_homographies(
            sensor_data_df=sensor_data_df,
            video_fps=video_fps,
            video_width=video_width,
            video_height=video_height,
            video_num_frames=video_num_frames,
            ground_z=ground_z,
        )
    except PixelToWorldError:
        raise
    except Exception as e:
        raise PixelToWorldError(f"Homography pipeline failed: {e}")


def _read_csv(csv_path, description):
    """Read a CSV whole; its columns are checked by the pipeline function it is passed to."""
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        raise DataLoadError(f"Failed to load {description} CSV: {e}")
