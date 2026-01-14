import os

class PathManager:
    """
    A class to manage all file paths used in the fish tracking pipeline.
    """
    def __init__(self, video_file, drone_file, base_output_dir):
        """
        Initialize the PathManager with input files and output directories.
        
        Args:
            video_file (str): Path to the input video file
            drone_file (str): Path to the drone flight log file
            base_output_dir (str, optional): Base directory for all outputs. 
                                            Defaults to 'tracking-output' in current directory.
        """
        self.video_file = video_file
        self.drone_file = drone_file
        self.base_output_dir = base_output_dir

        self.video_name = os.path.splitext(os.path.basename(video_file))[0]
        self.predictions_dir = os.path.join(self.base_output_dir, '01-predictions')
        self.gps_coordinates_dir = os.path.join(self.base_output_dir, '02-gps-coordinates')
        self.preferred_format_dir = os.path.join(self.base_output_dir, '03-preferred-format')

        # Create output directories
        os.makedirs(self.base_output_dir, exist_ok=True)
        os.makedirs(self.predictions_dir, exist_ok=True)
        os.makedirs(self.gps_coordinates_dir, exist_ok=True)
        os.makedirs(self.preferred_format_dir, exist_ok=True)
        
        # Define output file paths
        self.predictions_file = os.path.join(self.predictions_dir, f'{self.video_name}-predictions.csv')
        self.output_pkl = os.path.join(self.base_output_dir, 'pre_' + self.video_name + '.pkl')
        self.statistics_file = os.path.join(self.base_output_dir, 'image_tracks.csv') 
        self.world_trajectory_path = os.path.join(self.gps_coordinates_dir, 'world-trajectory.csv')
        self.world_track_file = os.path.join(self.gps_coordinates_dir, 'world-tracks.avi')
        self.temp_png_file = os.path.join(self.gps_coordinates_dir, 'scatter-plot.png')
    