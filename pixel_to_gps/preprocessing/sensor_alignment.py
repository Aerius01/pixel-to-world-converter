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
    frame_to_sensor_idx = np.array([
        np.argmin(np.abs(sensor_times - frame_time))
        for frame_time in frame_times
    ])

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
        list: List of [east, north, up] positions in meters relative to first GPS position
    """
    pose_gps = []

    for i in range(len(sensor_df)):
        lat = sensor_df.iloc[i]['latitude']
        lon = sensor_df.iloc[i]['longitude']
        alt = sensor_df.iloc[i]['altitude(m)']

        # Convert to ENU coordinates relative to reference (first GPS position)
        # Reference altitude at sea level == 0
        # This call converts from geodetic (latitude, longitude, altitude) to local ENU (north, east, up) coordinates (in meters).
        enu_coords = pm.geodetic2enu(lat, lon, alt, sensor_df.iloc[0]['latitude'], sensor_df.iloc[0]['longitude'], 0,
                              ell=pm.Ellipsoid.from_name("wgs84"), deg=True)

        # Reorder to [east, north, up] and flip Z-axis
        pose_gps.append([enu_coords[1], enu_coords[0], -enu_coords[2]])

    return pose_gps
