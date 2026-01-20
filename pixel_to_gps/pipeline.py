"""Main GPS conversion pipeline orchestrating all processing stages."""

import numpy as np
import pandas as pd

from .config import (
    KF_DOF, FOCAL_LENGTH, SENSOR_WIDTH,
    GIMBAL_PITCH_OFFSET, CONVERSION_FACTOR
)
from .preprocessing import align_frames_to_sensors, get_velocity_enu, get_pose_gps
from .pose_estimation import (
    StaticKalmanFilter, RTSSmoother,
    compute_extrinsic_matrix, compute_intrinsic_matrix
)
from .projection import (
    TrajectoryProcessor,
    prepare_pixel_tracks,
    extract_species_label
)


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
            velocityZ(mps), gimbalPitchRaw, gimbalRollRaw, gimbalYawRaw
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
    frame_times, sensor_times, frame_to_sensor_idx = align_frames_to_sensors(
        video_fps, video_num_frames, sensor_data_df
    )

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
    frame_duration = 1.0 / video_fps
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
