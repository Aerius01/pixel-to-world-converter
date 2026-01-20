"""Sensor data alignment and coordinate transformations."""

import numpy as np
import pymap3d as pm


def align_frames_to_sensors(video_fps, video_num_frames, sensor_data_df):
    """
    Create mapping from video frames to sensor data rows based on timestamps.

    Args:
        video_fps (float): Video frames per second
        video_num_frames (int): Total number of frames in video
        sensor_data_df (pd.DataFrame): Sensor data with 'time(millisecond)' column

    Returns:
        tuple: (frame_times, sensor_times, frame_to_sensor_idx) where:
            - frame_times: Array of timestamps for each frame (seconds)
            - sensor_times: Array of sensor measurement timestamps (seconds)
            - frame_to_sensor_idx: Array mapping each frame to nearest sensor row index
    """
    # Create time arrays for frames and sensor measurements
    frame_duration = 1.0 / video_fps  # Time between frames
    frame_times = np.arange(video_num_frames) * frame_duration
    sensor_times = (sensor_data_df['time(millisecond)'].values - sensor_data_df['time(millisecond)'].iloc[0]) / 1000

    # For each frame, find the sensor data row with the closest timestamp
    # Use binary search (O(n log m)) instead of linear search (O(n*m))
    # searchsorted finds insertion points; we need to check both neighbors for closest match
    insert_idx = np.searchsorted(sensor_times, frame_times, side='left')

    # Vectorized computation of closest neighbor indices
    # Clamp insert_idx to valid range for left neighbor
    left_idx = np.clip(insert_idx - 1, 0, len(sensor_times) - 1)
    # Clamp insert_idx to valid range for right neighbor
    right_idx = np.clip(insert_idx, 0, len(sensor_times) - 1)

    # Compute distances to left and right neighbors
    left_dist = np.abs(sensor_times[left_idx] - frame_times)
    right_dist = np.abs(sensor_times[right_idx] - frame_times)

    # Choose the closer neighbor (prefer left when equal, matching original argmin behavior)
    # argmin returns the first (lowest index) when there are ties
    frame_to_sensor_idx = np.where(left_dist <= right_dist, left_idx, right_idx)

    return frame_times, sensor_times, frame_to_sensor_idx


def get_velocity_enu(sensor_df, idx):
    """
    Get drone velocity at specified index, converted from NED to ENU coordinate system.

    The drone's IMU reports velocity in NED (North-East-Down) coordinates, which we
    convert to ENU (East-North-Up) for consistency with the world coordinate system.

    NED to ENU conversion:
    - X: North → East (negated)
    - Y: East → North (unchanged)
    - Z: Down → Up (negated)

    Args:
        sensor_df (pd.DataFrame): Sensor data DataFrame with velocity columns
        idx (int): Index of the sensor measurement to retrieve

    Returns:
        np.array: Velocity vector [x, y, z] in m/s (ENU coordinates)
    """
    return np.array([
        -sensor_df.iloc[idx]['velocityX(mps)'],   # North → East (negated)
        sensor_df.iloc[idx]['velocityY(mps)'],     # East → North
        -sensor_df.iloc[idx]['velocityZ(mps)']     # Down → Up (negated)
    ])


def get_pose_gps(sensor_df):
    """
    Convert GPS positions to local ENU coordinates relative to first position.

    Args:
        sensor_df (pd.DataFrame): Sensor data with latitude, longitude, altitude columns

    Returns:
        np.ndarray: Array of [east, north, up] positions in meters relative to first GPS position
                    Shape: (n_sensors, 3)
    """
    # Extract all GPS coordinates as numpy arrays for vectorized processing
    lats = sensor_df['latitude'].values
    lons = sensor_df['longitude'].values
    alts = sensor_df['altitude(m)'].values

    # Get reference point (first GPS position)
    ref_lat = lats[0]
    ref_lon = lons[0]
    ref_alt = 0  # Reference altitude at sea level

    # Vectorized conversion to ENU coordinates
    # pymap3d.geodetic2enu returns (east, north, up) and accepts arrays
    east, north, up = pm.geodetic2enu(
        lats, lons, alts,
        ref_lat, ref_lon, ref_alt,
        ell=pm.Ellipsoid.from_name("wgs84"),
        deg=True
    )

    # Stack into (N, 3) array with [north, east, -up] order to match original behavior
    # Note: Original code swapped east/north with [enu_coords[1], enu_coords[0], -enu_coords[2]]
    pose_gps = np.column_stack([north, east, -up])

    return pose_gps
