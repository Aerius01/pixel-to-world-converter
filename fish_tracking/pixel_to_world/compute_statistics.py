from fish_tracking.common.globals import *
#from utils.common import *
from fish_tracking.common.math_utils import *
import math
import copy
import pandas as pd

def twoD_to_threeD(p, camera_extrinsics, K):

    p = np.array([p[0], p[1], 1])

    # Compute the inverse of the intrinsic matrix
    K_inv = np.linalg.inv(K)

    # Convert the image point to camera coordinates
    Z_scale_new = -camera_extrinsics[2, 3]
    p_camera = np.dot(K_inv, p) * Z_scale_new  # [fX/Z, fY/Z, 1] -> [fX, fY, Z]

    # convert to NED
    p_camera_ned = [-p_camera[1], p_camera[0], p_camera[2]]

    # Invert the rotation matrix
    R = camera_extrinsics[:3, :3]
    R_inv = np.linalg.inv(R)

    # Calculate the translation vector in world coordinates
    t = camera_extrinsics[:3, -1]
    t_inv = np.dot(R_inv, t)

    # Rotate then Translate
    p_world = R_inv.dot(p_camera_ned) + t

    # Compute the world coordinates
    #p_world = np.dot(R_inv, p_camera_ned) + t_inv

    return p_world

class atd_statistics_t:
    def __init__(self,avg_position,avg_velocity,avg_pixel_pos,vel_pixel_pos,avg_angle=0.0,confidence=0,num_components=0,component_centers=[]):
        self.avg_position = avg_position
        self.avg_velocity = avg_velocity
        self.avg_pixel_pos = avg_pixel_pos
        self.vel_pixel_pos = vel_pixel_pos
        self.avg_angle = avg_angle
        self.confidence = confidence
        self.num_components = num_components
        self.component_centers = component_centers

def update_statistics(current_position, time_per_frame_ms, avg_position, avg_velocity, last_position):

    current_velocity = (current_position - last_position) * 1000 / time_per_frame_ms

    if np.linalg.norm(current_velocity) > MAX_VELOCITY:
        current_velocity = (MAX_VELOCITY / np.linalg.norm(current_velocity)) * current_velocity

    avg_position.update(current_position)
    avg_velocity.update(current_velocity)

    avg_angle = math.atan2(avg_velocity.value[1], avg_velocity.value[0]) * RAD2DEG
    #avg_pixel_pos, vel_pixel_pos = world_to_image(avg_position, avg_velocity, cam)

    last_position = current_position

    return last_position, [0,0], [0,0], avg_angle, avg_position, avg_velocity


def compute_statistics(merged, K, camera_exstrinsics, time_per_frame_ms,
                       avg_position, avg_velocity, last_position, target_id, species_label, W, H, ref_geo_lla):
    """
    Compute world coordinate statistics for tracked objects.
    
    Returns:
        pd.DataFrame: DataFrame containing trajectory statistics with columns:
            avg_pos_x, avg_pos_y, avg_vel_x, avg_vel_y, avg_vel, avg_pixel_pos_x,
            avg_pixel_pos_y, vel_pixel_pos_x, vel_pixel_pos_y, angle, frame,
            target_id, species_label, ref_latitude, ref_longitude, ref_altitude
    """
    rows = []

    for i in range(len(merged)-1): # not always one needs to subtract off -1 only when last extrinsic reached

        #position = [W / 2, H / 2]
        p_world = twoD_to_threeD(merged[i][0].position, camera_exstrinsics[merged[i][0].frame], K)
        #p_world = twoD_to_threeD(position, camera_exstrinsics[i], K)

        last_position, avg_pixel_pos, vel_pixel_pos, avg_angle, avg_position, avg_velocity = update_statistics(
                p_world, time_per_frame_ms,
                avg_position, avg_velocity, last_position)

        stat = atd_statistics_t(avg_position.get_value(), avg_velocity.get_value(), avg_pixel_pos, vel_pixel_pos,
                                avg_angle)

        row = {
            'avg_pos_x': stat.avg_position[0],
            'avg_pos_y': stat.avg_position[1],
            'avg_vel_x': stat.avg_velocity[0],
            'avg_vel_y': stat.avg_velocity[1],
            'avg_vel': np.linalg.norm(stat.avg_velocity),
            'avg_pixel_pos_x': merged[i][0].position[0],
            'avg_pixel_pos_y': merged[i][0].position[1],
            'vel_pixel_pos_x': stat.vel_pixel_pos[0],
            'vel_pixel_pos_y': stat.vel_pixel_pos[1],
            'angle': stat.avg_angle,
            'frame': merged[i][0].get_frame(),
            'target_id': target_id,
            'species_label': species_label,
            'ref_latitude': ref_geo_lla[0],
            'ref_longitude': ref_geo_lla[1],
            'ref_altitude': ref_geo_lla[2]
        }
        rows.append(row)

    if len(rows) > 0 and not (len(rows) - 1) % DISPLAY_STEPS:
        print("writing stat number " + str(len(rows) - 1))

    return pd.DataFrame(rows)



