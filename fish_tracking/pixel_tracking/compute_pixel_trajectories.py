import copy
import os
import sys
from fish_tracking.common.camera_utils.camera_handler import CameraHandler
from fish_tracking.common.kalman_filter import KalmanFilter
from fish_tracking.common.camera_utils.camera import Camera
from fish_tracking.pixel_tracking.particle_filter import pf_reset, pf_prediction, pf_correction, pf_resampling, \
    pf_init, write_final_particle_csv,write_pixel_statistics
from fish_tracking.common.math_utils import geo_to_cartesian,get_quaternion_from_euler
from fish_tracking.common.animation_smoother import AnimationSmoother
import cv2
from fish_tracking.pixel_tracking.tracker import Tracker
from fish_tracking.common.globals import *
from fish_tracking.common.video_data import VideoData
from fish_tracking.common.path_manager import PathManager
from tqdm import tqdm


def compute_weighted_average(output_map):
    rows, cols, _ = output_map.shape

    indices = np.indices((rows, cols))
    values = output_map[:, :, 0]

    x_indices = indices[1]
    y_indices = indices[0]

    weighted_x = np.sum(x_indices * values)
    weighted_y = np.sum(y_indices * values)
    weight_sum = np.sum(values)

    if weight_sum == 0:
        return np.array([cols // 2, rows // 2])

    return np.array([weighted_x // weight_sum, weighted_y // weight_sum])

def compute_particles(path_manager: PathManager, video_data: VideoData):

    particle_count = 1000
    multi_class = False
    time_per_frame_ms = 1000.0 / video_data.fps

    smooth = True

    # Store all cameras
    cameras = []

    # create multiple position data to output them all in csv files
    target_cam_pos = np.array([0.0,0.0,0.0])

    # Get a list of all files in the folder
    file_list = os.listdir(path_manager.soft_masks_dir)
    # Get the number of PNG files
    n_frames = len(file_list)

    # Initialize an empty list to store the frames
    first_output_maps = []

    # Read the first three frames and append them to output_maps
    for i in range(3):
        file_name = f"image_{i+1:05d}.png"
        file_path = os.path.join(path_manager.soft_masks_dir, file_name)

        # Check if the file exists
        if os.path.exists(file_path):
            # Read the image
            frame = cv2.imread(file_path)
            # resize image
            frame = cv2.resize(frame, (video_data.width, video_data.height))
            first_output_maps.append(frame)

    pf_filters, num_targets = pf_init(video_data.width, video_data.height, particle_count, first_output_maps, multi_class, 0)
    trackers= []
    for id, filter in enumerate(pf_filters):
        trackers.append(Tracker(filter, id,0, n_frames,"Marlin"))

    # read drone data from csv files
    print("Initializing drone pose")
    with_mov_sync = True

    drone_record_data = CameraHandler()

    drone_record_data.init_reader(path_manager.drone_file, video_data.video_name, with_mov_sync)
    updated = drone_record_data.read_next_line(0)

    if updated and len(drone_record_data.last_line) != 0:
        drone_record_data.update_drone_data()


    image_width = 2 * FOCAL_LENGTH * math.tan(OPENING_ANGLE_X / 2.0)
    image_height = image_width * (video_data.height / video_data.width)

    opening_angle_y = OPENING_ANGLE_X * image_height / image_width
    reference_geo = np.array([drone_record_data.geo[0], drone_record_data.geo[1], 0])

    # initialize pose in camera frame (or world?)
    cam_q = get_quaternion_from_euler(drone_record_data.gimbal_euler)
    cam_pos = geo_to_cartesian(drone_record_data.geo, reference_geo)

    current_camera = Camera()
    current_camera.init(FOCAL_LENGTH, image_width, image_height, video_data.width, video_data.height,
                        OPENING_ANGLE_X, opening_angle_y)

    current_camera.compute_image_pose(cam_q, cam_pos)
    last_camera = copy.deepcopy(current_camera)

    # initializing kalman filter
    kf = KalmanFilter(KF_DOF, INIT_STATE_VAR)
    kf.init(pos_variance=POS_VAR,
            alt_variance=Z_POS_VAR,
            vel_variance=VEL_VAR,
            out_q=OUTER_Q_FACTOR,
            in_q=INNER_Q_FACTOR,
            initial_state=np.array([cam_pos[0],cam_pos[1],drone_record_data.geo[2],0.0,0.0,0.0])
            )
    smoother = AnimationSmoother()

    for i in tqdm(range(n_frames), desc="Processing frames"):
        file_name = f"image_{i + 1:05d}.png"
        file_path = os.path.join(path_manager.soft_masks_dir, file_name)
        frame = cv2.imread(file_path)
        frame = cv2.resize(frame, (video_data.width, video_data.height))

        # read drone data from csv files
        if i != 0:
            updated = drone_record_data.read_next_line(float(i) * time_per_frame_ms)

            if updated and len(drone_record_data.last_line) != 0:
                drone_record_data.update_drone_data()


        # Get camera rotation in quaternion
        target_cam_q = get_quaternion_from_euler(drone_record_data.gimbal_euler)

        time_diff = (drone_record_data.time_ms - drone_record_data.prev_time_ms) / 1000

        if updated and time_diff>0:
            # kalman pos update
            new_values = np.zeros(KF_DOF)
            new_values[0:2] = geo_to_cartesian(drone_record_data.geo, reference_geo)[0:2]
            new_values[2] = drone_record_data.geo[2]
            new_values[3:6] = drone_record_data.vel[0:3]

            kf.predict(time_diff)
            kf.update(new_values)

            target_cam_pos = kf.state[0:3]


        # prepare data for world space image space conversion
        if smooth:
            cam_pos, cam_q= smoother.update_cam_pose_animation(cam_pos, target_cam_pos, cam_q,
                                                                                target_cam_q)
        else:
            cam_q=target_cam_q
            cam_pos=target_cam_pos

        current_camera.compute_image_pose(cam_q, cam_pos)

        for k in range(len(trackers)):
            if current_camera.img_normal[2] > MINIMUM_SIN_ANGLE:
                if trackers[k].undet_frame_count > RESET_THRESHOLD:
                    trackers[k].particles = pf_reset(trackers[k].particles, video_data.width, video_data.height, particle_count, 0)
                    trackers[k].undet_frame_count = 0
                    trackers[k].det_frame_count = 0
                trackers[k].particles = pf_prediction(trackers[k].particles, current_camera, last_camera, video_data.width, video_data.height)
                trackers[k].particles, found = pf_correction(trackers[k].particles, frame)
                if not found:
                    trackers[k].undet_frame_count += 1
                    trackers[k].det_frame_count = 0
                else:
                    trackers[k].undet_frame_count = 0
                    trackers[k].det_frame_count += 1
                trackers[k].particles = pf_resampling(trackers[k].particles, video_data.width, video_data.height)
            trackers[k].confidence.append(MAX_CONFIDENCE if trackers[k].det_frame_count > MAX_CONFIDENCE else trackers[k].det_frame_count)
        last_camera.compute_image_pose(cam_q, cam_pos)

        # Store everything
        cameras.append(copy.deepcopy(current_camera))
        for tracker in trackers:
            tracker.memory.append(copy.deepcopy(tracker.particles))

    drone_record_data.close_file()

    for tracker in trackers:
        write_final_particle_csv(tracker.memory, path_manager.pf_coordinates_file)

    write_pixel_statistics(trackers, path_manager.statistics_file, video_data.video_name)








