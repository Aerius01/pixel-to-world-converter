import pandas as pd
from fish_tracking.common.globals import *

class CameraHandler:
    def __init__(self):
        self.geo = np.array([0.0,0.0,0.0])
        self.time_ms = 0.0
        self.prev_time_ms = 0.0
        self.vel = np.array([0.0,0.0,0.0])
        self.drone_euler = np.array([0.0,0.0,0.0])
        self.gimbal_euler = np.array([0.0,0.0,0.0])
        self.curr_line = None
        self.last_line = None
        self.next_time_ms = 0.0
        self.offset_time_ms = 0.0
        self.updated = False
        # DataFrame-based attributes
        self.drone_data_df = None
        self.current_row_idx = 0

    def update_drone_data(self):
        self.geo[0] = float(self.last_line[LAT_COL_N])
        self.geo[1] = float(self.last_line[LON_COL_N])
        self.geo[2] = float(
            self.last_line[ALT_COL_N]) * HEIGHT_CORRECTION - CAMERA_HEIGHT_OFFSET + STARTING_OFFSET
        self.prev_time_ms = self.time_ms
        self.time_ms = float(self.last_line[TIME_MS_COL_N])

        # Convert NED to ENU
        self.vel[0] = float(self.last_line[VEL_Y_COL_N])
        self.vel[1] = float(self.last_line[VEL_X_COL_N])
        self.vel[2] = -float(self.last_line[VEL_Z_COL_N])

        # Convert NED to ENU
        self.drone_euler[0] = float(self.last_line[DRONE_ROLL_COL_N]) * DEG2RAD
        self.drone_euler[1] = float(self.last_line[DRONE_PITCH_COL_N]) * DEG2RAD
        self.drone_euler[2] = -float(self.last_line[DRONE_YAW_COL_N]) * DEG2RAD

        # Convert NED to ENU
        self.gimbal_euler[0] = (float(self.last_line[GIMBAL_ROLL_COL_N]) / CONVERSION_FACTOR) * DEG2RAD
        self.gimbal_euler[1] = (float(self.last_line[GIMBAL_PITCH_COL_N]) / CONVERSION_FACTOR) * DEG2RAD - GIMBAL_PITCH_OFFSET
        self.gimbal_euler[2] = -(float(self.last_line[GIMBAL_YAW_COL_N]) / CONVERSION_FACTOR) * DEG2RAD

    def init_reader(self, drone_data_df):
        """
        Initialize the drone log reader with a DataFrame.

        Args:
            drone_data_df (pd.DataFrame): DataFrame containing drone sensor data.
                                          Expected to be pre-filtered for the specific video.
        """
        if not isinstance(drone_data_df, pd.DataFrame):
            raise TypeError(f"drone_data_df must be a pandas DataFrame, got {type(drone_data_df)}")
        
        if drone_data_df.empty:
            raise ValueError("Drone log DataFrame is empty")
        
        self.drone_data_df = drone_data_df.reset_index(drop=True)
        self.current_row_idx = 0
        
        # Get first row as Series (supports column indexing like a dict)
        self.curr_line = self.drone_data_df.iloc[0]
        self.offset_time_ms = float(self.curr_line[TIME_MS_COL_N])
        self.next_time_ms = self.offset_time_ms

    def read_next_line(self, current_time_ms):
        """
        Read the next row from the drone log DataFrame if the current video timestamp
        exceeds the next sensor timestamp. If end of DataFrame is reached, gracefully
        return False and reuse the last available sensor data.

        Args:
            current_time_ms: Current video timestamp in milliseconds

        Returns:
            bool: True if a new row was read, False otherwise
        """
        updated = False
        self.last_line = self.curr_line
        counter = self.next_time_ms - self.offset_time_ms - SENSOR_DATA_DELAY_MS

        while current_time_ms > counter:
            self.current_row_idx += 1
            
            if self.current_row_idx >= len(self.drone_data_df):
                # Reached end of DataFrame
                # Keep using the last available sensor data
                return updated
            
            self.last_line = self.curr_line
            self.curr_line = self.drone_data_df.iloc[self.current_row_idx]
            self.next_time_ms = float(self.curr_line[TIME_MS_COL_N])
            counter = self.next_time_ms - self.offset_time_ms - SENSOR_DATA_DELAY_MS
            updated = True

        return updated



