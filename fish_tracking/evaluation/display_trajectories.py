import sys
import cv2
import csv
import random
from fish_tracking.common.globals import *
import matplotlib.pyplot as plt
import numpy as np
import pymap3d as pm
from fish_tracking.common.video_data import VideoData
from fish_tracking.common.path_manager import PathManager
import gc
import subprocess

GT_MARKER = np.array([
    [52.4487070, 13.6483305],
    [52.4485310, 13.6484467],
    [52.4483757, 13.6482998],
    [52.4485358, 13.6481780],
    [52.4485650, 13.6479768],
    [52.4484324, 13.6477040],
    [52.4482760, 13.6475148],
    [52.4486172, 13.6482383]
])

def read_particles(csv_file, line_number):
    """Read particle positions from a CSV file at a specific line."""
    with open(csv_file, 'r') as file:
        for current_line, line in enumerate(file):
            if current_line == line_number:
                tmp_vector = line.strip().split(',')
                return [(int(tmp_vector[i]), int(tmp_vector[i + 1])) for i in range(0, len(tmp_vector), 2)]
    return []

def get_positions_for_frame(sorted_csv_dict, frame_number):
    """Extract position data for a specific frame from sorted CSV data."""
    positions = []

    for row in sorted_csv_dict:
        if int(row['frame']) == frame_number:
            positions.append({
                'avg_pixel_pos': [float(row['avg_pixel_pos_x']), float(row['avg_pixel_pos_y'])],
                'avg_world_pos': [float(row['avg_pos_x']), float(row['avg_pos_y'])],
                'target_id': row['target_id'],
                'ref_geo_lla': [float(row['ref_latitude']), float(row['ref_longitude']), float(row['ref_altitude'])]
            })

    return positions

def get_pose_data_for_frame(statistics_file, frame_number):
    """Read pose data for a specific frame from statistics file."""
    x_pose, y_pose, ids = [], [], []
    
    with open(statistics_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if int(row['frame']) == frame_number:
                x_pose.append(row['x'])
                y_pose.append(row['y'])
                ids.append(row['id'])
    
    return np.column_stack((np.array(x_pose), np.array(y_pose), np.array(ids)))

def find_position_bounds(rows):
    """Find min/max position values from trajectory data."""
    min_x, max_x = sys.float_info.max, -sys.float_info.max
    min_y, max_y = sys.float_info.max, -sys.float_info.max

    for r in rows:
        x_val = float(r['avg_pos_x'])
        y_val = float(r['avg_pos_y'])
        min_x = min(min_x, x_val)
        max_x = max(max_x, x_val)
        min_y = min(min_y, y_val)
        max_y = max(max_y, y_val)

    return min_x, max_x, min_y, max_y

def setup_trajectory_plot(width, height, sizing_ratio, lla_min, lla_max):
    """Set up the matplotlib plot for trajectory visualization."""
    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    plt.figure(figsize=(width * px, height * px))

    plt.grid()
    plt.ylim(lla_min[0], lla_max[0])
    plt.xlim(lla_min[1], lla_max[1])
    plt.ylabel('latitude', fontsize=18/sizing_ratio)
    plt.xlabel('longitude', fontsize=18/sizing_ratio)
    plt.xticks(fontsize=12/sizing_ratio)
    plt.yticks(fontsize=12/sizing_ratio)
    
    ax = plt.gca()
    ax.xaxis.get_offset_text().set_size(12/sizing_ratio)
    ax.yaxis.get_offset_text().set_size(12/sizing_ratio)

def draw_particles_and_centers(frame, particles, pose_data, color_map, sizing_ratio, plot_particles=True):
    """Draw particles and center points on the frame."""
    if plot_particles:
        particle_color = (8, 100, 245)
        for p in particles:
            frame = cv2.circle(frame, p, int(3/sizing_ratio), particle_color, -1)
    
    for t in range(len(pose_data)):
        # Plot object center point
        x, y = int(round(float(pose_data[t, 0]))), int(round(float(pose_data[t, 1])))
        target_id = str(pose_data[t, 2])
        
        frame = cv2.circle(frame, (x, y), int(10/sizing_ratio), color_map[target_id], -1)
        frame = cv2.circle(frame, (x, y), int(12/sizing_ratio), (0, 0, 0), 3)
    
    return frame

def display_trajectories(path_manager: PathManager, video_data: VideoData, plot_particles=True, progress_callback=None):
    """Display and save trajectory visualization from tracking data.
    
    Args:
        path_manager (PathManager): Path manager object
        video_data (VideoData): Video data object
        plot_particles (bool): Whether to plot particles
        progress_callback (callable, optional): Callback function to report progress
    """
    # Setup dimensions and scaling
    NEW_Y_RES = int(video_data.height)
    sizing_ratio = 1024/video_data.width
    new_width = int((video_data.width / video_data.height) * NEW_Y_RES)
    
    # Initialize video capture
    cap_footage = cv2.VideoCapture(path_manager.video_file)
    
    W = int(video_data.width)
    B = int(video_data.height * 2)
    
    # Create temporary directory for frames
    import tempfile
    import os
    temp_dir = tempfile.mkdtemp(prefix='trajectory_frames_')
    
    # Initialize trajectory frame
    trajectory_frame = np.ones((NEW_Y_RES, new_width, 3), np.uint8) * 255
    
    # Read trajectory data
    with open(path_manager.world_trajectory_path, 'r') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    
    # Find position bounds
    min_x, max_x, min_y, max_y = find_position_bounds(rows)
    
    # Sort data and prepare color mapping
    particle_stats_3D = sorted(rows, key=lambda row: row['frame'])
    target_ids = {row['target_id'] for row in particle_stats_3D}
    color_map = {target_id: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) 
                for target_id in target_ids}
    
    # Initialize trajectory storage
    trajectory_data = {target_id: {'trajectory_latitude': [], 'trajectory_longitude': []} 
                      for target_id in target_ids}
    
    # Process each frame
    for frame_idx in range(video_data.num_frames):
        if progress_callback:
            progress_callback(frame_idx)
            
        # Read pose data for current frame
        pose_2D = get_pose_data_for_frame(path_manager.statistics_file, frame_idx)
        ret, frame_footage = cap_footage.read()
        
        if not ret:
            break
            
        # Read estimated trajectories
        positions = get_positions_for_frame(particle_stats_3D, frame_idx)
        
        # If no trajectory detections available, just show raw frame
        if not positions:
            concatenated_img = np.concatenate((cv2.resize(frame_footage, (new_width, NEW_Y_RES)), 
                                              trajectory_frame), axis=0)
            concatenated_img = cv2.resize(concatenated_img, (W, B))
            # Save frame to temporary directory
            frame_path = os.path.join(temp_dir, f'frame_{frame_idx:06d}.png')
            cv2.imwrite(frame_path, concatenated_img)
            continue
        
        # Draw particles and center points
        # particles = read_particles(path_manager.pf_coordinates_file, frame_idx) if plot_particles else []
        frame_footage = draw_particles_and_centers(frame_footage, [], pose_2D, color_map, sizing_ratio, plot_particles)
        
        # Prepare trajectory plot
        ref_geo_lla = positions[0]["ref_geo_lla"]
        lla_min = pm.enu2geodetic(min_y, min_x, 0, ref_geo_lla[0], ref_geo_lla[1],
                                 ref_geo_lla[2], ell=pm.Ellipsoid.from_name("wgs84"), deg=True)
        lla_max = pm.enu2geodetic(max_y, max_x, 0, ref_geo_lla[0], ref_geo_lla[1],
                                 ref_geo_lla[2], ell=pm.Ellipsoid.from_name("wgs84"), deg=True)
        
        setup_trajectory_plot(video_data.width, video_data.height, sizing_ratio, lla_min, lla_max)
        
        # Process each position
        for pos in positions:
            avg_world_pos = pos["avg_world_pos"]
            target_id = pos["target_id"]
            ref_geo_lla = pos["ref_geo_lla"]
            
            # Convert to LLA coordinates
            ned = np.array([avg_world_pos[0], avg_world_pos[1], 0])
            individual_pose_lla = pm.enu2geodetic(ned[1], ned[0], -ned[2], 
                                                ref_geo_lla[0], ref_geo_lla[1], ref_geo_lla[2], 
                                                ell=pm.Ellipsoid.from_name("wgs84"), deg=True)
            
            # Store trajectory data
            trajectory_data[target_id]['trajectory_latitude'].append(individual_pose_lla[0])
            trajectory_data[target_id]['trajectory_longitude'].append(individual_pose_lla[1])
        
        # Plot trajectories
        for target_id in target_ids:
            color_tuple = color_map[str(target_id)]
            color_normalized = tuple(x / 256 for x in color_tuple)
            
            # # Plot ground truth marker
            # plt.scatter(GT_MARKER[int(target_id), 1], GT_MARKER[int(target_id), 0], 
            #            s=int(140/sizing_ratio), edgecolor="black", linewidth=1, 
            #            c=np.array([color_normalized]), marker='*')
            
            # Plot trajectory
            plt.scatter(np.array(trajectory_data[target_id]['trajectory_longitude']), 
                       np.array(trajectory_data[target_id]['trajectory_latitude']), 
                       s=int(14/sizing_ratio), c=np.array([color_normalized]), marker='o')
        
        # Save plot to image and convert to frame
        plt.savefig(path_manager.temp_png_file, format='png', bbox_inches='tight', pad_inches=0.02)
        plt.close('all')  # Close all figures to free memory
        plot_image = plt.imread(path_manager.temp_png_file)
        
        # Process plot image
        if plot_image.shape[-1] == 4:  # Convert RGBA to RGB
            plot_image = plot_image[..., :3]
        
        # Calculate padding for the plot image
        padding = (new_width - plot_image.shape[1], NEW_Y_RES - plot_image.shape[0])
        padding = (padding[0] // 2, padding[1] // 2, (padding[0] + 1) // 2, (padding[1] + 1) // 2)
        
        # Create trajectory frame with the plot
        trajectory_frame = np.ones((NEW_Y_RES, new_width, 3), np.uint8) * 255
        trajectory_frame[padding[1]:padding[1]+plot_image.shape[0], 
                        padding[0]:padding[0]+plot_image.shape[1]] = plot_image * 255
        
        # Resize and concatenate frames
        resized_footage = cv2.resize(frame_footage, (new_width, NEW_Y_RES)) if frame_footage.shape[0] != NEW_Y_RES else frame_footage
        concatenated_img = np.concatenate((resized_footage, trajectory_frame), axis=0)
        concatenated_img = cv2.resize(concatenated_img, (W, B))
        
        # Save frame to temporary directory
        frame_path = os.path.join(temp_dir, f'frame_{frame_idx:06d}.png')
        cv2.imwrite(frame_path, concatenated_img)
        
        # Clean up memory
        del plot_image
        del concatenated_img
        gc.collect()
    
    # Clean up video capture
    cap_footage.release()
    cv2.destroyAllWindows()
    plt.close('all')
    gc.collect()
    
    # Combine frames into video using ffmpeg
    ffmpeg_cmd = [
        'ffmpeg', '-y',  # Overwrite output file if it exists
        '-framerate', str(video_data.fps),
        '-i', os.path.join(temp_dir, 'frame_%06d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '23',  # High quality, reasonable file size
        path_manager.world_track_file
    ]
    
    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e}")
    finally:
        # Clean up temporary directory
        import shutil
        shutil.rmtree(temp_dir)