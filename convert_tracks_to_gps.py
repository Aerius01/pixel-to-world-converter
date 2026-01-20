#!/usr/bin/env python3
"""
GPS Coordinate Converter for Fish Tracking

Converts pixel coordinates from image tracking to world GPS coordinates using
drone sensor data (GPS, IMU, gimbal orientation).

Usage:
    python convert_tracks_to_gps.py \
        --image-tracks path/to/image_tracks.csv \
        --drone-log path/to/drone_log.csv \
        --video path/to/video.mp4 \
        --output path/to/output.csv
"""

import argparse
import sys
import os
import cv2
import pandas as pd

from pixel_to_gps import convert_pixel_to_world
from pixel_to_gps.schema import DRONE_LOG_SCHEMA, IMAGE_TRACKS_SCHEMA


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


class VideoError(Exception):
    """Raised when video file cannot be processed."""
    pass


class DataLoadError(Exception):
    """Raised when CSV data cannot be loaded or is invalid."""
    pass


def validate_file_exists(filepath, file_description):
    """Validate that a file exists and is readable.

    Args:
        filepath (str): Path to the file to validate
        file_description (str): Human-readable description of the file

    Raises:
        ValidationError: If file doesn't exist or isn't readable
    """
    if not os.path.isfile(filepath):
        raise ValidationError(f"{file_description} not found: {filepath}")
    if not os.access(filepath, os.R_OK):
        raise ValidationError(f"{file_description} is not readable: {filepath}")


def validate_output_path(output_path):
    """Validate that output directory exists or can be created.

    Args:
        output_path (str): Path to the output file

    Raises:
        ValidationError: If output directory cannot be created
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            raise ValidationError(f"Cannot create output directory {output_dir}: {e}")


def validate_csv_schema(csv_path, schema, file_description):
    """Validate that a CSV file matches the expected schema.

    Args:
        csv_path (str): Path to the CSV file
        schema: Schema object with validate() method
        file_description (str): Human-readable description of the file

    Raises:
        DataLoadError: If schema validation fails
    """
    csv_preview = pd.read_csv(csv_path, nrows=0)  # Read only header
    missing_columns = schema.validate(csv_preview)
    if missing_columns:
        error_msg = f"{file_description} is missing required columns: {', '.join(missing_columns)}\n"
        error_msg += f"Required columns: {', '.join(schema.get_required_columns())}"
        raise DataLoadError(error_msg)


def extract_video_properties(video_path):
    """Extract video properties using OpenCV.

    Args:
        video_path (str): Path to the video file

    Returns:
        tuple: (fps, width, height, num_frames)

    Raises:
        VideoError: If video file cannot be opened
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
        # Validate input files
        print("Validating input files...")
        validate_file_exists(args.image_tracks, "Image tracks file")
        validate_file_exists(args.drone_log, "Drone log file")
        validate_file_exists(args.video, "Video file")

        # Validate output path
        validate_output_path(args.output)

        # Get video parameters
        print(f"Extracting video properties from: {args.video}")
        video_fps, video_width, video_height, video_num_frames = extract_video_properties(args.video)
        print(f"  FPS: {video_fps}")
        print(f"  Resolution: {video_width}x{video_height}")
        print(f"  Frames: {video_num_frames}")

        # Load CSV files into DataFrames
        print("\nLoading input files into memory...")

        # Validate schemas
        validate_csv_schema(args.image_tracks, IMAGE_TRACKS_SCHEMA, "Image tracks file")
        validate_csv_schema(args.drone_log, DRONE_LOG_SCHEMA, "Drone log file")

        # Load the actual data
        image_tracks_df = pd.read_csv(args.image_tracks, usecols=IMAGE_TRACKS_SCHEMA.get_required_columns())
        print(f"  Loaded {len(image_tracks_df)} image track records")

        sensor_data_df = pd.read_csv(args.drone_log, usecols=DRONE_LOG_SCHEMA.get_required_columns(), dtype=float)
        print(f"  Loaded {len(sensor_data_df)} sensor records")

        # Verify all velocity measurements are populated with non-NaN values
        velocity_columns = [DRONE_LOG_SCHEMA.vel_x_col, DRONE_LOG_SCHEMA.vel_y_col, DRONE_LOG_SCHEMA.vel_z_col]
        for col in velocity_columns:
            if sensor_data_df[col].isna().any():
                missing_count = sensor_data_df[col].isna().sum()
                raise DataLoadError(
                    f"Drone log contains {missing_count} missing values in {col}. "
                    f"All velocity measurements are required for GPS conversion."
                )

        # Run conversion
        print("\nStarting GPS coordinate conversion...")
        print(f"Input tracks: {args.image_tracks}")
        print(f"Input drone log: {args.drone_log}")
        print(f"Output: {args.output}")
        print()

        trajectory_df = convert_pixel_to_world(
            image_tracks_df=image_tracks_df,
            sensor_data_df=sensor_data_df,
            video_fps=video_fps,
            video_width=video_width,
            video_height=video_height,
            video_num_frames=video_num_frames
        )

        # Save output to CSV
        print(f"\nSaving {len(trajectory_df)} trajectory records to CSV...")
        trajectory_df.to_csv(args.output, index=False)

        print("✅ Conversion completed successfully!")
        print(f"Output written to: {args.output}")

    except (ValidationError, VideoError, DataLoadError) as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
