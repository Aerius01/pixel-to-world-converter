#!/usr/bin/env python3
"""
Convert pixel coordinates from image-based object tracking to world GPS coordinates.

This script serves as the main entry point for the GPS converter pipeline. It:
1. Validates input files and their schemas
2. Extracts video properties (FPS, dimensions)
3. Orchestrates the conversion pipeline
4. Outputs world trajectory data in CSV format

The conversion process transforms 2D pixel coordinates from drone video frames into
3D world coordinates (ENU system) using camera pose estimation and ray-plane intersection.
"""

import argparse
import sys
import os
import cv2
import pandas as pd

from pixel_to_gps import convert_pixel_to_world
from pixel_to_gps.schema import DRONE_LOG_SCHEMA, IMAGE_TRACKS_SCHEMA


class ValidationError(Exception):
    """Raised when input file validation fails (missing files, permissions, etc.)."""
    pass


class VideoError(Exception):
    """Raised when video file cannot be opened or read."""
    pass


class DataLoadError(Exception):
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


def main():
    """
    Main entry point for GPS conversion pipeline.

    Parses command-line arguments, validates inputs, extracts video properties,
    and orchestrates the full pixel-to-world conversion pipeline.
    """
    parser = argparse.ArgumentParser(
        description='Convert pixel coordinates from image tracking to world GPS coordinates.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  %(prog)s --image-tracks tracks.csv --drone-log drone.csv --video video.mp4 --output world_coords.csv

Notes:
  - The drone log file should be pre-filtered for the specific video timeframe
  - All required columns will be validated before processing
  - Output will be written in CSV format with GPS coordinates
        """
    )

    # Required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '--image-tracks',
        required=True,
        metavar='PATH',
        help='Path to input CSV file containing pixel coordinates of tracked objects'
    )
    required.add_argument(
        '--drone-log',
        required=True,
        metavar='PATH',
        help='Path to pre-filtered drone sensor data CSV file'
    )
    required.add_argument(
        '--output',
        required=True,
        metavar='PATH',
        help='Path to output CSV file for world GPS coordinates'
    )
    required.add_argument(
        '--video',
        required=True,
        metavar='PATH',
        help='Path to video file (properties will be extracted automatically)'
    )

    args = parser.parse_args()

    try:
        print("=" * 80)
        print("GPS CONVERTER - Pixel to World Coordinate Transformation")
        print("=" * 80)

        # Step 1: Validate input files
        print("\n[1/6] Validating input files...")
        validate_file_exists(args.image_tracks, "Image tracks file")
        validate_file_exists(args.drone_log, "Drone log file")
        validate_file_exists(args.video, "Video file")
        validate_output_path(args.output)
        print("      All input files validated successfully")

        # Step 2: Extract video properties
        print("\n[2/6] Extracting video properties...")
        video_fps, video_width, video_height, video_num_frames = extract_video_properties(args.video)
        print(f"      Video: {video_width}x{video_height} @ {video_fps:.2f} fps ({video_num_frames} frames)")

        # Step 3: Validate CSV schemas
        print("\n[3/6] Validating CSV schemas...")
        validate_csv_schema(args.image_tracks, IMAGE_TRACKS_SCHEMA, "Image tracks file")
        validate_csv_schema(args.drone_log, DRONE_LOG_SCHEMA, "Drone log file")
        print("      All required columns present")

        # Step 4: Load data
        print("\n[4/6] Loading CSV data...")
        image_tracks_df = pd.read_csv(args.image_tracks, usecols=IMAGE_TRACKS_SCHEMA.get_required_columns())
        sensor_data_df = pd.read_csv(args.drone_log, usecols=DRONE_LOG_SCHEMA.get_required_columns(), dtype=float)

        num_tracks = len(image_tracks_df)
        num_targets = image_tracks_df['id'].nunique()
        num_sensor_records = len(sensor_data_df)
        print(f"      Loaded {num_tracks} detections across {num_targets} targets")
        print(f"      Loaded {num_sensor_records} sensor records")

        # Step 5: Validate velocity data (required for Kalman filter)
        print("\n[5/6] Validating sensor data completeness...")
        velocity_columns = [DRONE_LOG_SCHEMA.vel_x_col, DRONE_LOG_SCHEMA.vel_y_col, DRONE_LOG_SCHEMA.vel_z_col]
        for col in velocity_columns:
            if sensor_data_df[col].isna().any():
                missing_count = sensor_data_df[col].isna().sum()
                raise DataLoadError(
                    f"Drone log contains {missing_count} missing values in {col}. "
                    f"All velocity measurements are required for GPS conversion."
                )
        print("      All velocity measurements present (required for Kalman filter)")

        # Step 6: Run conversion pipeline
        print("\n[6/6] Running GPS conversion pipeline...")
        print("      This may take a few minutes depending on video length...")
        trajectory_df = convert_pixel_to_world(
            image_tracks_df=image_tracks_df,
            sensor_data_df=sensor_data_df,
            video_fps=video_fps,
            video_width=video_width,
            video_height=video_height,
            video_num_frames=video_num_frames
        )

        # Save output
        trajectory_df.to_csv(args.output, index=False)
        num_trajectory_records = len(trajectory_df)
        print(f"      Conversion complete! Generated {num_trajectory_records} trajectory records")

        print(f"\n{'=' * 80}")
        print(f"SUCCESS: Output written to {args.output}")
        print(f"{'=' * 80}\n")

    except (ValidationError, VideoError, DataLoadError) as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nUNEXPECTED ERROR:", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
