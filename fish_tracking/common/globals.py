import math
import numpy as np

# ============================================================
# Kalman Filter Parameters
# ============================================================
KF_DOF = 6  # state size [x, y, z, vx, vy, vz]

# ============================================================
# Camera Hardware Specifications
# ============================================================
FOCAL_LENGTH = 0.0088  # distance from camera origin (aperture) to inverted image plane in meters (8.8mm)
SENSOR_WIDTH = 0.0132  # 13.2mm - sensor width in meters

# ============================================================
# Gimbal and Drone Parameters
# ============================================================
GIMBAL_PITCH_OFFSET = -math.pi / 2.0  # Gimbal pitch offset to correct reference frame
CONVERSION_FACTOR = 10.0  # Converting raw gimbal angles to real angles

# ============================================================
# Trajectory Processing Parameters
# ============================================================
VELOCITY_BUFFER_SIZE = 60  # Number of frames for velocity averaging
POSITION_BUFFER_SIZE = 1   # Number of frames for position averaging
MAX_VELOCITY = 10.0        # Maximum velocity threshold in m/s (filter out larger velocities)

# ============================================================
# Mathematical Constants
# ============================================================
RAD2DEG = 180.0 / math.pi
DEG2RAD = math.pi / 180.0
EARTH_RADIUS_M = 6371000.0

# ============================================================
# Camera Geometry (for legacy camera.py module)
# ============================================================
DEFAULT_IMG_NORMAL = np.array([0.0, 0.0, 1.0])
DEFAULT_IMG_UP = np.array([0.0, 1.0, 0.0])
OCEAN_POINT = np.array([0.0, 0.0, 0.0])
OCEAN_NORMAL = np.array([0.0, 0.0, 1.0])
DBL_MAX = float("inf")

# ============================================================
# Math Utilities (for legacy math_utils.py functions)
# ============================================================
STD_DEV = 1.0  # Standard deviation for Gaussian noise function
