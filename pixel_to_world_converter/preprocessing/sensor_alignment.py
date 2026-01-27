"""
Sensor data preprocessing and frame-to-sensor alignment.

This module handles temporal synchronization between video frames and drone sensor
measurements. Since video frames and sensor data are recorded at different rates
and may not be perfectly synchronized, we must map each frame to its nearest sensor
timestamp for accurate GPS conversion.

Key functions:
- align_frames_to_sensors: Map each video frame to nearest sensor measurement
- get_pose_gps: Convert GPS coordinates to ENU (East-North-Up) meters
- get_velocity_enu: Convert NED velocity to ENU coordinate system
"""

import numpy as np
import pymap3d as pm


def align_frames_to_sensors(video_fps, video_num_frames, sensor_data_df):
    """
    Map each video frame to its nearest sensor measurement timestamp.

    Video frames and sensor data are recorded independently, so we need to establish
    a temporal mapping. This function computes the timestamp for each frame based on
    the video FPS, then finds the nearest sensor measurement using binary search.

    The mapping is bidirectional: we return both the frame-to-sensor index mapping
    and the raw timestamps for validation purposes (checking time drift).

    Algorithm:
        1. Compute frame timestamps: [0, 1/fps, 2/fps, ..., (N-1)/fps]
        2. Normalize sensor timestamps relative to first sensor time
        3. Use binary search (searchsorted) to find insertion points
        4. Compare left and right neighbors to find nearest sensor index

    Args:
        video_fps: Video frame rate (frames per second)
        video_num_frames: Total number of frames in video
        sensor_data_df: DataFrame with 'time(millisecond)' column

    Returns:
        tuple: (frame_times, sensor_times, frame_to_sensor_idx)
            - frame_times: ndarray of shape (video_num_frames,) with frame timestamps in seconds
            - sensor_times: ndarray of sensor timestamps in seconds (normalized to start at 0)
            - frame_to_sensor_idx: ndarray of shape (video_num_frames,) mapping frame index -> sensor index
    """
    # Compute time per frame in seconds
    frame_duration = 1.0 / video_fps
    # Generate frame timestamps starting at 0
    frame_times = np.arange(video_num_frames) * frame_duration

    # Normalize sensor timestamps to start at 0 (subtract first timestamp, convert ms to seconds)
    sensor_times = (sensor_data_df['time(millisecond)'].values - sensor_data_df['time(millisecond)'].iloc[0]) / 1000

    # Binary search to find where each frame time would be inserted in sensor_times
    # side='left' means we get the index of the first sensor >= frame_time
    insert_idx = np.searchsorted(sensor_times, frame_times, side='left')

    # Find left and right neighbor indices (clipping to valid range)
    left_idx = np.clip(insert_idx - 1, 0, len(sensor_times) - 1)
    right_idx = np.clip(insert_idx, 0, len(sensor_times) - 1)

    # Compute absolute time differences to left and right neighbors
    left_dist = np.abs(sensor_times[left_idx] - frame_times)
    right_dist = np.abs(sensor_times[right_idx] - frame_times)

    # Choose the neighbor with smaller distance (ties go to left)
    frame_to_sensor_idx = np.where(left_dist <= right_dist, left_idx, right_idx)

    return frame_times, sensor_times, frame_to_sensor_idx


def get_velocity_enu(sensor_df, idx):
    """
    Extract velocity from sensor DataFrame and convert from NED to ENU coordinates.

    Drone IMU reports velocities in NED (North-East-Down) frame, but the pipeline
    operates in ENU (East-North-Up) frame. This function performs the coordinate
    transformation by negating North and Down components.

    Coordinate Transformation:
        NED -> ENU
        North (X) -> East (negated)
        East (Y) -> North (unchanged)
        Down (Z) -> Up (negated)

    Args:
        sensor_df: DataFrame with velocity columns [velocityX(mps), velocityY(mps), velocityZ(mps)]
        idx: Row index to extract velocity from

    Returns:
        ndarray of shape (3,) with velocity [vx_east, vy_north, vz_up] in meters/second
    """
    return np.array([
        -sensor_df.iloc[idx]['velocityX(mps)'],  # North -> East (negated)
        sensor_df.iloc[idx]['velocityY(mps)'],   # East -> North
        -sensor_df.iloc[idx]['velocityZ(mps)']   # Down -> Up (negated)
    ])


def get_pose_gps(sensor_df):
    """
    Convert GPS coordinates from geodetic (lat/lon/alt) to ENU (East-North-Up) meters.

    Uses the first GPS position as the origin (reference point) and converts all
    positions to local ENU coordinates in meters. The reference altitude is set to
    0 (sea level) so all positions are relative to the ocean surface.

    The pymap3d library handles the WGS84 ellipsoid projection, accounting for
    Earth's curvature when computing local distances.

    Coordinate System Details:
        - Origin: First GPS position (ref_lat, ref_lon, altitude=0)
        - X-axis: East (meters)
        - Y-axis: North (meters)
        - Z-axis: Up (meters), where sea level = 0
        - Note: Output has negated Up to match expected convention

    Args:
        sensor_df: DataFrame with columns [latitude, longitude, altitude(m)]

    Returns:
        ndarray of shape (N, 3) where each row is [north, east, -up] in meters
        relative to the first GPS position at sea level
    """
    # Extract GPS arrays
    lats = sensor_df['latitude'].values
    lons = sensor_df['longitude'].values
    alts = sensor_df['altitude(m)'].values

    # Use first position as reference origin
    ref_lat = lats[0]
    ref_lon = lons[0]
    ref_alt = 0  # Sea level

    # Convert geodetic to ENU using WGS84 ellipsoid
    east, north, up = pm.geodetic2enu(
        lats, lons, alts,
        ref_lat, ref_lon, ref_alt,
        ell=pm.Ellipsoid.from_name("wgs84"),
        deg=True
    )

    # Stack as [north, east, -up] to match expected output format
    pose_gps = np.column_stack([north, east, -up])

    return pose_gps
