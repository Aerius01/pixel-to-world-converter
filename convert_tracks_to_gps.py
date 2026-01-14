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
from pathlib import Path

from fish_tracking.pixel_to_world.convert_pixel_to_world import convert_pixel_to_world

# Required sensor data columns (all cast to float)
SENSOR_COLUMNS = [
    'time(millisecond)', 'latitude', 'longitude', 'altitude(m)',
    'velocityX(mps)', 'velocityY(mps)', 'velocityZ(mps)',
    'pitch(deg)', 'roll(deg)', 'yaw(deg)', 'isTakingVideo',
    'gimbalPitchRaw', 'gimbalRollRaw', 'gimbalYawRaw'
]


def validate_file_exists(filepath, file_description):
    """Validate that a file exists and is readable."""
    if not os.path.isfile(filepath):
        print(f"Error: {file_description} not found: {filepath}", file=sys.stderr)
        sys.exit(1)
    if not os.access(filepath, os.R_OK):
        print(f"Error: {file_description} is not readable: {filepath}", file=sys.stderr)
        sys.exit(1)


def validate_output_path(output_path):
    """Validate that output directory exists or can be created."""
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error: Cannot create output directory {output_dir}: {e}", file=sys.stderr)
            sys.exit(1)


def extract_video_properties(video_path):
    """Extract video properties using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    return fps, width, height, num_frames


def progress_callback(frame_idx):
    """Simple progress callback that prints every 100 frames."""
    if frame_idx % 100 == 0:
        print(f"Processing frame {frame_idx}...", end='\r')


def main():
    parser = argparse.ArgumentParser(
        description='Convert pixel coordinates from image tracking to world GPS coordinates.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  %(prog)s --image-tracks tracks.csv --drone-log drone.csv --video video.mp4 --output world_coords.csv

  # With explicit video parameters (skips video reading)
  %(prog)s --image-tracks tracks.csv --drone-log drone.csv \\
           --fps 59.94 --width 2720 --height 1530 --frames 1137 \\
           --output world_coords.csv

  # Quiet mode (no progress output)
  %(prog)s --image-tracks tracks.csv --drone-log drone.csv --video video.mp4 \\
           --output world_coords.csv --quiet

Notes:
  - The drone log file should be pre-filtered for the specific video timeframe
  - Image tracks CSV should contain columns: id, x, y, frame, label
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

    # Video input (mutually exclusive with explicit parameters)
    video_group = parser.add_argument_group('video source')
    video_mutex = video_group.add_mutually_exclusive_group(required=True)
    video_mutex.add_argument(
        '--video',
        metavar='PATH',
        help='Path to video file (properties will be extracted automatically)'
    )
    video_mutex.add_argument(
        '--video-params',
        action='store_true',
        help='Use explicit video parameters instead of video file (requires --fps, --width, --height, --frames)'
    )

    # Explicit video parameters
    params_group = parser.add_argument_group('explicit video parameters (requires --video-params)')
    params_group.add_argument(
        '--fps',
        type=float,
        metavar='FPS',
        help='Video frames per second'
    )
    params_group.add_argument(
        '--width',
        type=int,
        metavar='PIXELS',
        help='Video frame width in pixels'
    )
    params_group.add_argument(
        '--height',
        type=int,
        metavar='PIXELS',
        help='Video frame height in pixels'
    )
    params_group.add_argument(
        '--frames',
        type=int,
        metavar='COUNT',
        help='Total number of frames in video'
    )

    # Optional arguments
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )

    args = parser.parse_args()

    # Validate input files
    print("Validating input files...")
    validate_file_exists(args.image_tracks, "Image tracks file")
    validate_file_exists(args.drone_log, "Drone log file")

    # Validate output path
    validate_output_path(args.output)

    # Get video parameters
    if args.video:
        validate_file_exists(args.video, "Video file")
        print(f"Extracting video properties from: {args.video}")
        video_fps, video_width, video_height, video_num_frames = extract_video_properties(args.video)
        print(f"  FPS: {video_fps}")
        print(f"  Resolution: {video_width}x{video_height}")
        print(f"  Frames: {video_num_frames}")
    elif args.video_params:
        # Validate that all required parameters are provided
        if not all([args.fps, args.width, args.height, args.frames]):
            parser.error("--video-params requires --fps, --width, --height, and --frames")
        video_fps = args.fps
        video_width = args.width
        video_height = args.height
        video_num_frames = args.frames
        print(f"Using explicit video parameters:")
        print(f"  FPS: {video_fps}")
        print(f"  Resolution: {video_width}x{video_height}")
        print(f"  Frames: {video_num_frames}")

    # Load CSV files into DataFrames
    print("\nLoading input files into memory...")
    try:
        image_tracks_df = pd.read_csv(args.image_tracks)
        print(f"  Loaded {len(image_tracks_df)} image track records")

        sensor_data_df = pd.read_csv(args.drone_log, usecols=SENSOR_COLUMNS, dtype=float)
        print(f"  Loaded {len(sensor_data_df)} sensor records")
    except Exception as e:
        print(f"\n❌ Failed to load input files: {e}", file=sys.stderr)
        sys.exit(1)

    # Run conversion
    print("\nStarting GPS coordinate conversion...")
    print(f"Input tracks: {args.image_tracks}")
    print(f"Input drone log: {args.drone_log}")
    print(f"Output: {args.output}")
    print()

    try:
        trajectory_df = convert_pixel_to_world(
            image_tracks_df=image_tracks_df,
            sensor_data_df=sensor_data_df,
            video_fps=video_fps,
            video_width=video_width,
            video_height=video_height,
            video_num_frames=video_num_frames,
            progress_callback=None if args.quiet else progress_callback
        )
        
        # Save output to CSV
        print(f"\nSaving {len(trajectory_df)} trajectory records to CSV...")
        trajectory_df.to_csv(args.output, index=False)
        
        print("✅ Conversion completed successfully!")
        print(f"Output written to: {args.output}")

    except Exception as e:
        print(f"\n❌ Conversion failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
