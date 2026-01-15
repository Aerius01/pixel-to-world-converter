import numpy as np
import pandas as pd
import os
import sys
import copy
import math

from fish_tracking.common.kalman_filter import StaticKalmanFilter
from fish_tracking.common.camera_utils.camera_handler import CameraHandler
from fish_tracking.common.camera_utils.camera import Camera
from fish_tracking.common.animation_smoother import AnimationSmoother
from fish_tracking.common.globals import *
from fish_tracking.pixel_tracking.tracker import Particle
from fish_tracking.pixel_to_world.compute_statistics import *
from fish_tracking.pixel_to_world.buffer import ATDBuffer
from fish_tracking.pixel_to_world.rts_smoother import RTSSmoother

import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
import pymap3d as pm
from fish_tracking.common.video_data import VideoData
from fish_tracking.common.path_manager import PathManager
# current velocity for position, used for animation smoothing
cs_pos = np.array([0.0,0.0,0.0])

# current velocity for cam euler angles, used for animation smoothing
cs_cam_angle = np.array([0.0,0.0,0.0])

def compute_extrinsic_matrix(angles, cam_pos):
    """
    Compute the extrinsic matrix from gimbal angles and camera position.
    
    Args:
        angles (array-like): Gimbal angles in degrees where:
            - angles[0]: roll
            - angles[1]: pitch
            - angles[2]: yaw
        cam_pos (array-like): Camera position [x, y, z] in meters
    
    Returns:
        np.ndarray: 4x4 extrinsic matrix
    """
    r = Rot.from_euler('xyz', [np.deg2rad(angles[0]), np.deg2rad(angles[1]), -np.deg2rad(angles[2])])
    R = r.as_matrix()
    # center world coordinate system at initial drone pose
    trans = np.array([cam_pos[0], cam_pos[1], cam_pos[2]]).reshape(3, 1)
    # Combine R and t into a 3x4 matrix
    Rt = np.hstack((R, trans))
    # Create the 4x4 extrinsic matrix
    extrinsic = np.vstack((Rt, [0, 0, 0, 1]))

    return extrinsic

def lla_to_ecef(lat, lon, alt):
     # WGS-84 ellipsoid parameters
    a = 6378137.0  # Semi-major axis in meters
    e = 8.1819190842622e-2  # First eccentricity

    # Convert latitude and longitude to radians
    lat = np.radians(lat)
    lon = np.radians(lon)

    # Prime vertical radius of curvature
    N = a / np.sqrt(1 - e**2 * np.sin(lat)**2)

    # Calculate ECEF coordinates
    X = (N + alt) * np.cos(lat) * np.cos(lon)
    Y = (N + alt) * np.cos(lat) * np.sin(lon)
    Z = ((1 - e**2) * N + alt) * np.sin(lat)

    return np.array([X, Y, Z])


def ecef_to_ned(ecef_target, ecef_ref, ref_lat, ref_lon):
    """
    Convert ECEF coordinates to NED coordinates.
    """
    # Convert reference latitude and longitude from degrees to radians
    ref_lat = np.radians(ref_lat)
    ref_lon = np.radians(ref_lon)

    # Calculate the difference between the target and reference ECEF coordinates
    delta_ecef = ecef_target - ecef_ref

    # Define the rotation matrix from ECEF to NED
    R = np.array([
        [-np.sin(ref_lat) * np.cos(ref_lon), -np.sin(ref_lat) * np.sin(ref_lon), np.cos(ref_lat)],
        [-np.sin(ref_lon), np.cos(ref_lon), 0],
        [-np.cos(ref_lat) * np.cos(ref_lon), -np.cos(ref_lat) * np.sin(ref_lon), -np.sin(ref_lat)]
    ])

    # Calculate NED coordinates
    ned = np.dot(R, delta_ecef)

    return ned


def lla_to_ned(target_lat, target_lon, target_alt, ref_lat, ref_lon, ref_alt):
    """
    Convert target LLA coordinates to NED coordinates relative to a reference point.
    """
    # Convert both the target and reference LLA to ECEF coordinates
    ecef_target = lla_to_ecef(target_lat, target_lon, target_alt)
    ecef_ref = lla_to_ecef(ref_lat, ref_lon, ref_alt)

    # Convert ECEF coordinates to NED coordinates
    ned = ecef_to_ned(ecef_target, ecef_ref, ref_lat, ref_lon)

    return ned


def rotation_matrix_ned(yaw, pitch, roll):
    """
    Create a rotation matrix for NED coordinates from yaw, pitch, roll.
    """
    # Rotation around the Down (D) axis (Yaw)
    R_yaw = np.array([
        [np.cos(yaw), np.sin(yaw), 0],
        [-np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Rotation around the East (E) axis (Pitch)
    R_pitch = np.array([
        [np.cos(pitch), 0, -np.sin(pitch)],
        [0, 1, 0],
        [np.sin(pitch), 0, np.cos(pitch)]
    ])

    # Rotation around the North (N) axis (Roll)
    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), np.sin(roll)],
        [0, -np.sin(roll), np.cos(roll)]
    ])

    # Combined rotation matrix
    R = np.dot(R_roll, np.dot(R_pitch, R_yaw))

    return R

def compute_intrinsic_matrix(focal_length, pixel_size, image_width, image_height):
    # Calculate the focal lengths in pixels
    f_x = focal_length / pixel_size[0]
    f_y = focal_length / pixel_size[1]

    # Assume the principal point is at the center of the image
    c_x = image_width / 2
    c_y = image_height / 2

    # Construct the intrinsic matrix
    K = np.array([
        [f_x, 0, c_x],
        [0, f_y, c_y],
        [0, 0, 1]
    ])

    return K

def compute_fov(focal_length, sensor_dimension):
    # Compute the FOV in radians
    fov_rad = 2 * np.arctan(sensor_dimension / (2 * focal_length))

    # Convert the FOV to degrees
    fov_deg = np.degrees(fov_rad)

    return fov_deg

def extractSensorData(metadataFilePath):
    """
    Extract sensor data from pre-filtered drone log CSV file.

    Args:
        metadataFilePath (str): Path to CSV file containing drone sensor data.
                                Expected to be pre-filtered for the specific video.

    Returns:
        pd.DataFrame: Sensor data with columns:
            - time_msec: timestamp in milliseconds
            - latitude, longitude, altitude_m: GPS position
            - velocityX_mps, velocityY_mps, velocityZ_mps: velocity components
            - pitch_degree, roll_degree, yaw_degree: drone orientation
            - isTakingVideo: video recording flag
            - gimbalPitch, gimbalRoll, gimbalYaw: gimbal angles (raw)
    """
    # Read CSV using pandas for efficiency
    df = pd.read_csv(metadataFilePath)

    # Select and rename columns to match expected format
    sensor_data = pd.DataFrame({
        'time_msec': df['time(millisecond)'].astype(float),
        'latitude': df['latitude'].astype(float),
        'longitude': df['longitude'].astype(float),
        'altitude_m': df['altitude(m)'].astype(float),
        'velocityX_mps': df['velocityX(mps)'].astype(float),
        'velocityY_mps': df['velocityY(mps)'].astype(float),
        'velocityZ_mps': df['velocityZ(mps)'].astype(float),
        'pitch_degree': df['pitch(deg)'].astype(float),
        'roll_degree': df['roll(deg)'].astype(float),
        'yaw_degree': df['yaw(deg)'].astype(float),
        'isTakingVideo': df['isTakingVideo'].astype(float),
        'gimbalPitch': df['gimbalPitchRaw'].astype(float),
        'gimbalRoll': df['gimbalRollRaw'].astype(float),
        'gimbalYaw': df['gimbalYawRaw'].astype(float)
    })

    return sensor_data

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

def lat_lon_alt_to_cartesian(lat, lon, alt):
    # Constants for WGS84
    R_E = 6378137.0  # Earth's equatorial radius in meters
    e = 0.0818191908426  # Eccentricity

    # Convert latitude, longitude, and altitude to radians and meters
    phi = np.radians(lat)
    lam = np.radians(lon)
    h = alt

    # Calculate the prime vertical radius of curvature
    N = R_E / np.sqrt(1 - e**2 * np.sin(phi)**2)

    # Calculate Cartesian coordinates
    X = (N + h) * np.cos(phi) * np.cos(lam)
    Y = (N + h) * np.cos(phi) * np.sin(lam)
    Z = ((1 - e**2) * N + h) * np.sin(phi)

    return X, Y, Z

def ecef_to_enu(lat_ref, lon_ref, x, y, z, x_ref, y_ref, z_ref):
    # Convert reference latitude and longitude to radians
    lat_ref_rad = np.radians(lat_ref)
    lon_ref_rad = np.radians(lon_ref)

    # Compute the ENU basis vectors
    E = np.array([-np.sin(lon_ref_rad), np.cos(lon_ref_rad), 0])
    N = np.array([-np.sin(lat_ref_rad) * np.cos(lon_ref_rad),
                  -np.sin(lat_ref_rad) * np.sin(lon_ref_rad),
                  np.cos(lat_ref_rad)])
    U = np.array([np.cos(lat_ref_rad) * np.cos(lon_ref_rad),
                  np.cos(lat_ref_rad) * np.sin(lon_ref_rad),
                  np.sin(lat_ref_rad)])

    # Translation vector from reference ECEF to point ECEF
    delta_ecef = np.array([x - x_ref, y - y_ref, z - z_ref])

    # Calculate ENU coordinates by projecting the delta_ecef onto the ENU basis vectors
    east = np.dot(E, delta_ecef)
    north = np.dot(N, delta_ecef)
    up = np.dot(U, delta_ecef)

    return east, north, up

def getPose_GPS(sensor_df):
    poseGPS = []

    for i in range(len(sensor_df)):
        lat = sensor_df.iloc[i]['latitude']
        lon = sensor_df.iloc[i]['longitude']
        alt = sensor_df.iloc[i]['altitude(m)']

        # Convert to ENU coordinates relative to reference (first GPS position)
        # Reference altitude at sea level == 0
        # This call converts from geodetic (latitude, longitude, altitude) to local ENU (north, east, up) coordinates (in meters).
        enu = pm.geodetic2enu(lat, lon, alt, sensor_df.iloc[0]['latitude'], sensor_df.iloc[0]['longitude'], 0,
                              ell=pm.Ellipsoid.from_name("wgs84"), deg=True)

        # Reorder to [north, east, up] and flip Z-axis
        poseGPS.append([enu[1], enu[0], -enu[2]])

    return poseGPS

def get_camera(drone_record_data, i, time_per_frame_ms, current_camera, target_cam_pos, smoother, cam_pos, cam_q, reference_geo):
    """
    Update camera pose from drone sensor data.

    Note: This function no longer performs Kalman filtering. Camera positions are now
    sourced from the forward KF + RTS smoother pipeline (Section 2-3). This function
    only handles quaternion updates from gimbal sensors and animation smoothing.

    Legacy behavior (removed): Previously contained a real-time Kalman filter that
    matched the original C++ implementation, but was made redundant when the batch
    processing pipeline was added.
    """
    updated = drone_record_data.read_next_line(float(i) * time_per_frame_ms)

    if updated and len(drone_record_data.last_line) != 0:
        drone_record_data.update_drone_data()

        if i == 0:
            reference_geo = np.array([drone_record_data.geo[0], drone_record_data.geo[1], 0])

            # Initialize positions
            cam_q = get_quaternion_from_euler(drone_record_data.gimbal_euler)
            cam_pos = geo_to_cartesian(drone_record_data.geo, reference_geo)

            current_camera.compute_image_pose(copy.deepcopy(cam_q), copy.deepcopy(cam_pos))

    # Quaternion update from gimbal sensors
    target_cam_q = get_quaternion_from_euler(drone_record_data.gimbal_euler)

    # Perform iteration step for animation smoothing
    cam_pos, cam_q = smoother.update_cam_pose_animation(cam_pos, target_cam_pos, cam_q, target_cam_q)

    return cam_pos, cam_q, reference_geo

def get_img_tracks(statistics_data_handler, target_id):
    particles = []
    target_id = target_id #Baitball
    species_label = None

    for line in statistics_data_handler:

        if int(line["id"]) == target_id:
            x_pos = int(round(float(line["x"])))
            y_pos = int(round(float(line["y"])))
            particles.append([Particle(x_pos, y_pos, 1, int(line["frame"]))])
            # Get the species label (should be the same for all entries of this target_id)
            if species_label is None:
                species_label = line.get("label", "Unknown")

    return particles, species_label

def convert_pixel_to_world(
    image_tracks_df: pd.DataFrame,
    sensor_data_df: pd.DataFrame,
    video_fps: float,
    video_width: int,
    video_height: int,
    video_num_frames: int,
    progress_callback=None
):
    """Convert pixel coordinates from image tracks to world GPS coordinates.

    Args:
        image_tracks_df (pd.DataFrame): DataFrame containing pixel coordinates with columns:
            id, x, y, frame, label
        sensor_data_df (pd.DataFrame): DataFrame containing preprocessed drone sensor data with columns:
            time(millisecond), latitude, longitude, altitude(m), velocityX(mps), velocityY(mps),
            velocityZ(mps), pitch(deg), roll(deg), yaw(deg), isTakingVideo, gimbalPitchRaw,
            gimbalRollRaw, gimbalYawRaw
        video_fps (float): Video frames per second
        video_width (int): Video frame width in pixels
        video_height (int): Video frame height in pixels
        video_num_frames (int): Total number of frames in video
        progress_callback (callable, optional): Callback function to report progress
        
    Returns:
        pd.DataFrame: World trajectory data with GPS coordinates
    """

    # ============================================================
    # SECTION 1: Process drone sensor data
    # ============================================================
    # Create time arrays for frames and sensor measurements
    frame_dt = 1.0 / video_fps # Time between frames
    frame_times = [frame_dt * i for i in range(video_num_frames)]
    sensor_times = (sensor_data_df['time(millisecond)'].values - sensor_data_df['time(millisecond)'].iloc[0]) / 1000  # Convert to seconds
    
    # For each frame, find the sensor data row with the closest timestamp
    frame_to_sensor_idx = np.array([
        np.argmin(np.abs(sensor_times - frame_time)) 
        for frame_time in frame_times
    ])

    # ============================================================
    # SECTION 2: Forward Kalman filter for camera pose estimation
    # ============================================================
    # The Kalman filter predicts future values based on all the values that came before them.
    # That is why it's called a 'forward' pass.
    estimated_positions = []
    estimated_velocities = []
    estimated_position_variances = []
    estimated_velocity_variances = []

    # Extract GPS positions in meters relative to the first GPS position (origin)
    drone_positions_gps = getPose_GPS(sensor_data_df)

    # TODO: Investigate optimal noise attribution for camera pose estimation.
    # ORIGINAL C++ PARAMETERS (main.cpp:127, from master_thesis_project-main-HU):
    #   - pos_variance = POS_VAR * 10 = 6.0      # GPS horizontal position noise
    #   - alt_variance = Z_POS_VAR * 10 = 2.0    # GPS altitude noise
    #   - vel_variance = VEL_VAR * 10 = 2.0      # IMU velocity noise
    #   - out_q = OUTER_Q_FACTOR * 10 = 100.0    # Process noise outer factor
    #   - in_q = INNER_Q_FACTOR = 0.5            # Process noise inner factor (time-adaptive)
    #   - P_initial = INIT_STATE_VAR * I = 10 * I(6)  # Initial covariance
    #   - R: Per-state variances (pos_variance for x,y; alt_variance for z; vel_variance for vx,vy,vz)
    #   - Q: Time-adaptive Q(dt) using in_q and out_q scaling factors

    # The noise matrices were changed to their current static forms during the port from C++ to python.
    # The static noise matrices date back to Pia's ported implementation
    # The redundant Kalman filter and dates back to the C++ implementation
    
    # Initialize Kalman filter with constant noise matrices (matching original implementation)
    for k in range(1, len(frame_times)):
        # Get the sensor data row index for this frame using pre-computed mapping
        current_time = frame_times[k - 1]
        mapped_idx = frame_to_sensor_idx[k - 1]

        # On first iteration, initialize; otherwise predict
        if k == 1:
            # Initialize with first measurement (matching original updateNumber==1 behavior)
            first_measurement = np.concatenate((drone_positions_gps[mapped_idx], get_velocity_enu(sensor_data_df, mapped_idx)))
            kf = StaticKalmanFilter(
                initial_state=first_measurement,
                R=100 * np.eye(6),      # Constant measurement noise
                Q=np.eye(6),            # Constant process noise
                P_initial=np.eye(6)     # Initial covariance
            )
        else:
            # Predict step
            kf.predict(frame_dt)

            # Use measurement if sensor time is close to frame time
            if np.abs(sensor_times[mapped_idx] - current_time) < frame_dt:
                # Get velocity directly from sensor data at the matched index
                current_velocity = get_velocity_enu(sensor_data_df, mapped_idx)

                # Combine position and velocity measurements
                measurement = np.concatenate((drone_positions_gps[mapped_idx], current_velocity))
                kf.update(measurement)
        
        # Extract state and covariance after initialization/update
        filtered_state = kf.state
        filtered_variance = kf.P

        # Store filtered estimates
        estimated_positions.append(filtered_state[0:3])
        estimated_velocities.append(filtered_state[3:6])
        estimated_position_variances.append(np.array([filtered_variance[0, 0], filtered_variance[1, 1], filtered_variance[2, 2]]))
        estimated_velocity_variances.append(np.array([filtered_variance[3, 3], filtered_variance[4, 4], filtered_variance[5, 5]]))

    # ============================================================
    # SECTION 3: RTS backward smoother for improved estimates
    # ============================================================
    # The RTS smoother operates on a data set already smoothed by the Kalman filter
    # and refines the estimates by making a backward pass through it. Passively, 
    # it leverages the full dataset since it's operating on calculated values that 
    # have already considered all the values that came before them (and therefore 
    # incorporates 'future knowledge')
    smoothed_positions = []

    # Initialize RTS smoother
    smoother = RTSSmoother(frame_dt, dof=KF_DOF)

    # Each iteration of the loop updates the internal state of the smoother before
    # being passed into the next iteration, resulting in a sequential dependency chain
    for k in range(len(frame_times) - 1, 0, -1):
        state_kf = np.concatenate((estimated_positions[k - 1], estimated_velocities[k - 1]))
        variance_kf = np.eye(6) * np.concatenate((estimated_position_variances[k - 1], estimated_velocity_variances[k - 1]))
        smoothed_state, _ = smoother.update(state_kf, variance_kf)

        smoothed_positions.append(smoothed_state[0:3])

    # Reconstruct full sequence: prepend frame 0 from forward KF and reverse to forward chronological order
    full_smoothed_positions = [estimated_positions[0]] + smoothed_positions[::-1]

    # Verify: full_smoothed_positions now has length video_num_frames
    assert len(full_smoothed_positions) == len(frame_times), \
        f"Expected {len(frame_times)} positions, got {len(full_smoothed_positions)}"

    # ============================================================
    # SECTION 4: Compute camera extrinsic matrices
    # ============================================================
    # Pre-allocate numpy array for extrinsic matrices (each is 4x4)
    camera_extrinsics = np.zeros((video_num_frames, 4, 4))

    # It's more efficient to extract the 1D numpy arrays once and then index into them
    # rather than calling the dataframe accessor functions repeatedly in the loop.
    gimbal_yaw_values = sensor_data_df['gimbalYawRaw'].values
    gimbal_pitch_values = sensor_data_df['gimbalPitchRaw'].values
    gimbal_roll_values = sensor_data_df['gimbalRollRaw'].values

    for i in range(video_num_frames):
        mapped_idx = frame_to_sensor_idx[i]
        
        # Get gimbal angles as a vector [roll, pitch, yaw]
        camera_angles = np.array([
            gimbal_roll_values[mapped_idx],
            gimbal_pitch_values[mapped_idx],
            gimbal_yaw_values[mapped_idx]
        ]) / CONVERSION_FACTOR

        # Get camera position from smoothed estimates
        camera_position = full_smoothed_positions[i]

        # Compute extrinsic matrix (rotation + translation)
        camera_extrinsics[i] = compute_extrinsic_matrix(camera_angles, camera_position)

    # ============================================================
    # SECTION 5: Compute camera poses with animation smoothing
    # ============================================================
    time_per_frame_ms = 1000.0 / video_fps

    # Compute image plane dimensions from focal length and field of view
    # These represent the physical dimensions of the image plane in the pinhole camera model.
    # The image plane is where the 3D world projects onto an inverted 2D surface at distance FOCAL_LENGTH
    # from the camera center, behind the aperture (within the camera). We use half-angle trigonometry
    # because the half-angle produces a right triangle: tan(θ/2) = (width/2) / focal_length
    image_plane_width = 2 * FOCAL_LENGTH * math.tan(OPENING_ANGLE_X / 2.0)
    image_plane_height = image_plane_width * (video_height / video_width)
    opening_angle_y = OPENING_ANGLE_X * image_plane_height / image_plane_width

    # Initialize camera object for pose tracking
    current_camera = Camera()
    current_camera.init(FOCAL_LENGTH, image_plane_width, image_plane_height, video_width, video_height,
                        OPENING_ANGLE_X, opening_angle_y)

    # Initialize camera handler for real-time pose updates with DataFrame
    camera_handler = CameraHandler()
    camera_handler.init_reader(sensor_data_df)

    # Animation smoothing state
    pose_smoother = AnimationSmoother()
    camera_position = np.array([0.0, 0.0, 0.0])
    camera_quaternion = np.array([0.0, 0.0, 0.0, 0.0])
    target_camera_position = np.array([0.0, 0.0, 0.0])
    reference_geo_coords = np.array([0, 0, 0])

    # Storage for all camera poses
    all_camera_objects = []

    for frame_idx in range(video_num_frames):
        if progress_callback:
            progress_callback(frame_idx)

        # Update camera pose from drone data with smoothing
        _, camera_quaternion, reference_geo_coords = get_camera(
            camera_handler, frame_idx, time_per_frame_ms, current_camera,
            target_camera_position, pose_smoother, camera_position, camera_quaternion, reference_geo_coords
        )

        # Get smoothed position and apply coordinate transformation
        camera_position = copy.deepcopy(full_smoothed_positions[frame_idx])
        camera_position[0], camera_position[1] = camera_position[1], -camera_position[0]  # Swap and negate

        # Update camera object with current pose
        current_camera.compute_image_pose(camera_quaternion, camera_position)
        all_camera_objects.append(copy.deepcopy(current_camera))

    # ============================================================
    # SECTION 6: Convert pixel tracks to world coordinates
    # ============================================================
    # Compute camera intrinsic matrix for pixel-to-world projection
    pixel_size = (SENSOR_WIDTH/video_width, SENSOR_WIDTH/video_width)
    camera_intrinsic = compute_intrinsic_matrix(FOCAL_LENGTH, pixel_size, video_width, video_height)

    # Buffers for temporal averaging
    position_buffer = ATDBuffer(POSITION_BUFFER_SIZE)
    velocity_buffer = ATDBuffer(VELOCITY_BUFFER_SIZE)
    last_world_position = np.array([0, 0, 0])

    # Collect trajectory DataFrames for each target
    trajectory_dfs = []

    # Use first GPS position as reference point (origin)
    # Reference altitude at sea level == 0
    gps_reference = [sensor_data_df.iloc[0]['latitude'], sensor_data_df.iloc[0]['longitude'], 0]

    # Process each tracked target
    for target_id in image_tracks_df['id'].unique():
        # Filter DataFrame for current target and convert to dict iterator
        target_rows = image_tracks_df[image_tracks_df['id'] == target_id]
        # Convert to list of dicts to match get_img_tracks interface
        target_dicts = target_rows.to_dict('records')
        pixel_tracks, species_label = get_img_tracks(target_dicts, target_id)

        # Convert to world coordinates and get trajectory DataFrame
        target_trajectory_df = compute_statistics(
            pixel_tracks, camera_intrinsic, camera_extrinsics, time_per_frame_ms,
            position_buffer, velocity_buffer, last_world_position,
            target_id, species_label,
            video_width, video_height, gps_reference
        )
        trajectory_dfs.append(target_trajectory_df)

    # Concatenate all target trajectories into single DataFrame
    if trajectory_dfs:
        trajectory_df = pd.concat(trajectory_dfs, ignore_index=True)
    else:
        # Return empty DataFrame with correct columns if no trajectories
        trajectory_df = pd.DataFrame(columns=[
            'avg_pos_x', 'avg_pos_y', 'avg_vel_x', 'avg_vel_y', 'avg_vel',
            'avg_pixel_pos_x', 'avg_pixel_pos_y', 'vel_pixel_pos_x', 'vel_pixel_pos_y',
            'angle', 'frame', 'target_id', 'species_label',
            'ref_latitude', 'ref_longitude', 'ref_altitude'
        ])
    
    return trajectory_df
