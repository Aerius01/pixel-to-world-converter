import numpy as np
import pandas as pd

from fish_tracking.common.kalman_filter import StaticKalmanFilter
from fish_tracking.common.globals import *
from fish_tracking.pixel_to_world.rts_smoother import RTSSmoother
from fish_tracking.pixel_to_world.trajectory_processor import TrajectoryProcessor

from scipy.spatial.transform import Rotation as Rot
import pymap3d as pm


def compute_extrinsic_matrix(angles, camera_position):
    """
    Compute the extrinsic matrix from gimbal angles and camera position.
    
    Args:
        angles (array-like): Gimbal angles in degrees where:
            - angles[0]: roll
            - angles[1]: pitch
            - angles[2]: yaw
        camera_position (array-like): Camera position [x, y, z] in meters
    
    Returns:
        np.ndarray: 4x4 extrinsic matrix
    """
    rotation = Rot.from_euler('xyz', [np.deg2rad(angles[0]), np.deg2rad(angles[1]), -np.deg2rad(angles[2])])
    rotation_matrix = rotation.as_matrix()
    # center world coordinate system at initial drone pose
    translation = np.array(camera_position).reshape(3, 1)
    # Combine rotation_matrix and translation into a 3x4 matrix
    rotation_translation = np.hstack((rotation_matrix, translation))
    # Create the 4x4 extrinsic matrix
    extrinsic = np.vstack((rotation_translation, [0, 0, 0, 1]))

    return extrinsic


def compute_intrinsic_matrix(focal_length, pixel_size, image_width, image_height):
    """
    Compute the camera intrinsic matrix that maps 3D camera coordinates to 2D pixel coordinates.

    The intrinsic matrix encodes the camera's internal optical properties (lens and sensor),
    independent of where the camera is positioned or pointing in the world.

    Args:
        focal_length (float): Physical focal length of the lens in meters
        pixel_size (tuple): Physical size of one pixel (width, height) in meters
        image_width (int): Image width in pixels
        image_height (int): Image height in pixels

    Returns:
        np.ndarray: 3x3 intrinsic matrix K with structure:
            [[f_x,   0, c_x],
             [  0, f_y, c_y],
             [  0,   0,   1]]

            where:
            - f_x, f_y: focal lengths in pixel units (controls zoom/magnification)
            - c_x, c_y: principal point (where optical axis hits image plane)
            - zeros: no skew between pixel axes (pixels are rectangular)
            - 1: homogeneous coordinate for projection math
    """
    # Convert focal length from physical units (meters) to pixel units
    # This tells us how many pixels correspond to the camera's focal length
    f_x = focal_length / pixel_size[0]  # Horizontal focal length in pixels
    f_y = focal_length / pixel_size[1]  # Vertical focal length in pixels

    # The principal point is where the camera's optical axis intersects the image plane.
    # For most cameras, this is very close to the image center.
    c_x = image_width / 2   # Horizontal center
    c_y = image_height / 2  # Vertical center

    # Construct the 3x3 intrinsic matrix K
    # This matrix is used in the pinhole camera model to project 3D points to 2D pixels
    K = np.array([
        [f_x, 0, c_x],  # Row 1: x-axis scaling and offset
        [0, f_y, c_y],  # Row 2: y-axis scaling and offset
        [0, 0, 1]       # Row 3: homogeneous coordinates
    ])

    return K


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

def prepare_pixel_tracks(target_df: pd.DataFrame):
    """Prepare pixel track data for world coordinate conversion.

    Args:
        target_df: DataFrame already filtered to a single target_id with columns: id, x, y, frame, label

    Returns:
        tracks_df: DataFrame with columns x, y, frame (one row per detection)
    """
    # Return DataFrame with only the columns needed for tracking
    # Round x, y to integers to match original Particle class behavior
    tracks_df = target_df[['x', 'y', 'frame']].copy()
    tracks_df['x'] = tracks_df['x'].round().astype(int)
    tracks_df['y'] = tracks_df['y'].round().astype(int)
    tracks_df['frame'] = tracks_df['frame'].astype(int)

    return tracks_df


def extract_species_label(target_rows: pd.DataFrame) -> str:
    """
    Extract species label from target rows using majority voting.
    
    Args:
        target_rows: DataFrame rows for a single target
        
    Returns:
        str: Species label (most common label, or "Unknown" if unavailable)
    """
    if 'label' not in target_rows.columns or len(target_rows) == 0:
        return "Unknown"
    
    mode_result = target_rows['label'].mode()
    return mode_result.iloc[0] if len(mode_result) > 0 else "Unknown"

def convert_pixel_to_world(
    image_tracks_df: pd.DataFrame,
    sensor_data_df: pd.DataFrame,
    video_fps: float,
    video_width: int,
    video_height: int,
    video_num_frames: int
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
        
    Returns:
        pd.DataFrame: World trajectory data with GPS coordinates
    """

    # This function assumes that:
    # - The sensor data is pre-filtered for the specific video
    # - Each row of the sensor data contains all velocity measurements

    # ============================================================
    # SECTION 1: Process drone sensor data
    # ============================================================
    # Create time arrays for frames and sensor measurements
    frame_duration = 1.0 / video_fps # Time between frames
    frame_times = np.arange(video_num_frames) * frame_duration
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
    # Pre-allocate numpy arrays for Kalman filter estimates (one per frame)
    kalman_estimated_positions = np.zeros((video_num_frames, 3))
    kalman_estimated_velocities = np.zeros((video_num_frames, 3))
    kalman_estimated_position_variances = np.zeros((video_num_frames, 3))
    kalman_estimated_velocity_variances = np.zeros((video_num_frames, 3))

    # Extract GPS positions in meters relative to the first GPS position (origin)
    drone_positions_gps = get_pose_gps(sensor_data_df)

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
    for k in range(video_num_frames):
        # Get the sensor data row index for this frame using pre-computed mapping
        current_time = frame_times[k]
        mapped_sensor_idx = frame_to_sensor_idx[k]

        # On first iteration, initialize; otherwise predict
        if k == 0:
            # Initialize with first measurement (matching original updateNumber==1 behavior)
            first_measurement = np.concatenate((drone_positions_gps[mapped_sensor_idx], get_velocity_enu(sensor_data_df, mapped_sensor_idx)))
            kalman_filter = StaticKalmanFilter(
                initial_state=first_measurement,
                R=100 * np.eye(6),      # Constant measurement noise
                Q=np.eye(6),            # Constant process noise
                P_initial=np.eye(6)     # Initial covariance
            )
        else:
            # Predict step
            kalman_filter.predict(frame_duration)

            # Use measurement if sensor time is close to frame time
            if np.abs(sensor_times[mapped_sensor_idx] - current_time) < frame_duration:
                # Get velocity directly from sensor data at the matched index
                current_velocity = get_velocity_enu(sensor_data_df, mapped_sensor_idx)

                # Combine position and velocity measurements
                measurement = np.concatenate((drone_positions_gps[mapped_sensor_idx], current_velocity))
                kalman_filter.update(measurement)
        
        # Extract state and covariance after initialization/update
        filtered_state = kalman_filter.state
        filtered_variance = kalman_filter.P

        # Store filtered estimates
        kalman_estimated_positions[k] = filtered_state[0:3]
        kalman_estimated_velocities[k] = filtered_state[3:6]
        kalman_estimated_position_variances[k] = np.diag(filtered_variance)[:3]
        kalman_estimated_velocity_variances[k] = np.diag(filtered_variance)[3:]

    # ============================================================
    # SECTION 3: RTS backward smoother for improved estimates
    # ============================================================
    # The RTS smoother operates on a data set already smoothed by the Kalman filter
    # and refines the estimates by making a backward pass through it. Passively, 
    # it leverages the full dataset since it's operating on calculated values that 
    # have already considered all the values that came before them (and therefore 
    # incorporates 'future knowledge')
    rts_smoothed_positions = np.zeros((video_num_frames, 3))

    # Initialize RTS smoother
    smoother = RTSSmoother(frame_duration, dof=KF_DOF)

    # Each iteration of the loop updates the internal state of the smoother before
    # being passed into the next iteration, resulting in a sequential dependency chain
    for k in range(video_num_frames - 1, -1, -1):
        kalman_state = np.concatenate((kalman_estimated_positions[k], kalman_estimated_velocities[k]))
        kalman_variance = np.eye(6) * np.concatenate((kalman_estimated_position_variances[k], kalman_estimated_velocity_variances[k]))
        rts_smoothed_state, _ = smoother.update(kalman_state, kalman_variance)

        rts_smoothed_positions[k] = rts_smoothed_state[0:3]

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
        mapped_sensor_idx = frame_to_sensor_idx[i]

        # Get gimbal angles as a vector [roll, pitch, yaw] in degrees
        # Apply GIMBAL_PITCH_OFFSET to correct the gimbal's reference frame.
        # Raw gimbal pitch (~-90°) + offset (-90°) ≈ 0° in the body frame,
        camera_angles = np.array([
            gimbal_roll_values[mapped_sensor_idx] / CONVERSION_FACTOR,  # roll
            gimbal_pitch_values[mapped_sensor_idx] / CONVERSION_FACTOR - np.rad2deg(GIMBAL_PITCH_OFFSET),  # pitch
            gimbal_yaw_values[mapped_sensor_idx] / CONVERSION_FACTOR    # yaw
        ])

        # Get camera position from RTS smoothed estimates
        camera_position = rts_smoothed_positions[i]

        # Compute extrinsic matrix (rotation + translation)
        camera_extrinsics[i] = compute_extrinsic_matrix(camera_angles, camera_position)

    # ============================================================
    # SECTION 5: Convert pixel tracks to world coordinates
    # ============================================================
    # Use first GPS position as reference point (origin)
    # Reference altitude at sea level == 0
    gps_reference = [sensor_data_df.iloc[0]['latitude'], sensor_data_df.iloc[0]['longitude'], 0]

    # Compute camera intrinsic matrix for pixel-to-world projection
    pixel_size = (SENSOR_WIDTH/video_width, SENSOR_WIDTH/video_width)
    camera_intrinsic = compute_intrinsic_matrix(FOCAL_LENGTH, pixel_size, video_width, video_height)

    # Initialize trajectory processor with camera parameters
    processor = TrajectoryProcessor(
        camera_extrinsics=camera_extrinsics,
        camera_intrinsic=camera_intrinsic,
        video_fps=video_fps,
        gps_reference=gps_reference
    )

    # Collect trajectory records from all targets
    all_trajectory_records = []

    # Process each tracked target (trajectory id)
    for target_id in image_tracks_df['id'].unique():
        # Filter DataFrame for current target
        target_rows = image_tracks_df[image_tracks_df['id'] == target_id]

        # Extract species label using majority voting
        species_label = extract_species_label(target_rows)

        # Prepare pixel tracks (round coordinates to integers)
        pixel_tracks = prepare_pixel_tracks(target_rows)

        # Process target through trajectory processor
        target_records = processor.process_target(target_id, pixel_tracks, species_label)
        all_trajectory_records.extend(target_records)

    # Convert list of records to DataFrame
    trajectory_df = pd.DataFrame(all_trajectory_records)

    return trajectory_df
