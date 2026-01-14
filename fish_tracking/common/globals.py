import math
import numpy as np

KF_DOF = 6  # state size
INIT_STATE_VAR = 10.0  # initial variance, how certain are we about our initial position
POS_VAR = 0.6  # default variance in gps reading, i.e. how much we trust gps
Z_POS_VAR = 0.2  # default variance in height or altitude
VEL_VAR = 0.2  # default variance in velocity reading, i.e. how much we trust imu
INNER_Q_FACTOR = 0.5  # default variance for process covariance matrix
OUTER_Q_FACTOR = 10.0  # default variance for process covariance matrix

# Particle stat
MAX_CONFIDENCE = 200  # maximum confidence value
DISPLAY_STEPS = 100

# Paths
PF_COORDS_FB_FILE = "/pf_coords_fb.csv"
PF_COORDS_FB_GI_FILE = "/pf_coords_fb_gi.csv"
DL_STATS_FILE = "/dl_stats.csv"
PF_STATS_FILE = "/pf_stats.csv"
PF_STATS_FB_FILE = "/pf_stats_fb.csv"
PF_STATS_FB_GI_FILE = "/pf_stats_fb_gi.csv"
ATE_DL_FILE = "/ate_dl.csv"
ATE_PF_FILE = "/ate_pf.csv"
ATE_PF_FB_FILE = "/ate_pf_fb.csv"
ATE_PF_FB_GI_FILE = "/ate_pf_fb_gi.csv"  # atd_io.h
NUM_ENTRIES = 15
ATD_OK = 0

ALGORITHM_LEVEL_DEEPLABV3 = 0
ALGORITHM_LEVEL_PARTICLE_FILTER = 1
ALGORITHM_LEVEL_FORWARD_BACKWARD = 2
ALGORITHM_LEVEL_GAP_SMOOTHING = 3

# Camera hardware specifications
FOCAL_LENGTH = 0.0088 #0.02  # distance from camera origin (aperture) to inverted image plane in meters (8.8mm)
OPENING_ANGLE_X = 1.25  # horizontal field of view in radians (~71.6°)

# The sensor is the physical surface upon which the inverted image plane is projected, within the camera.
SENSOR_WIDTH = 0.0132   # 13.2mm
SENSOR_HEIGHT = 0.0088  # 8.8mm

# Tracking algorithm parameters
MINIMUM_SIN_ANGLE = 0.1736  # minimum downward angle for stable tracking (sin(10°) ≈ 0.1736)
                            # camera must point at least 10° down from the horizontal plane to avoid geometric instability

GIMBAL_PITCH_OFFSET = - math.pi / 2.0  #
CAMERA_HEIGHT_OFFSET = 0.4  # // height offset between gps module and camera in meters
STARTING_OFFSET = 0.0
HEIGHT_CORRECTION = 0.943  # // height correction for gps module as
CONFIDENCE_COL = 9

CHANGE_SPEED = 0.001
POS_ACC = 0.01
P_GAIN = 200
ANGLE_ACC = 0.001

LAT_COL_N = "latitude"
LON_COL_N = "longitude"
ALT_COL_N = "altitude(m)"
TIME_MS_COL_N = "time(millisecond)"
DATE_COL_N = "datetime(utc)"
VEL_X_COL_N = "velocityX(mps)"
VEL_Y_COL_N = "velocityY(mps)"
VEL_Z_COL_N = "velocityZ(mps)"
DRONE_PITCH_COL_N = "pitch(deg)"
DRONE_ROLL_COL_N = "roll(deg)"
DRONE_YAW_COL_N = "yaw(deg)"
GIMBAL_PITCH_COL_N = "gimbalPitchRaw"
GIMBAL_ROLL_COL_N = "gimbalRollRaw"
GIMBAL_YAW_COL_N = "gimbalYawRaw"
MOV_NAME_COL_N = "MOV_Name"
CONVERSION_FACTOR = 10.0  # converting raw angles to real angles

SENSOR_DATA_DELAY_MS = 300  # delay of e.g. gps data, negative delay means the sensor data is faster than video footage

DEG2RAD = (math.pi / 180.0)

LAT_COL = 0
LON_COL = 1
ALT_COL = 2
TIME_MS_COL = 3
DATE_COL = 4
VEL_X_COL = 5
VEL_Y_COL = 6
VEL_Z_COL = 7
DRONE_PITCH_COL = 8
DRONE_ROLL_COL = 9
DRONE_YAW_COL = 10
GIMBAL_PITCH_COL = 11
GIMBAL_ROLL_COL = 12
GIMBAL_YAW_COL = 13
MOV_NAME_COL = 14

DEFAULT_IMG_NORMAL = np.array([0.0, 0.0, 1.0])
DEFAULT_IMG_UP = np.array([0.0, 1.0, 0.0])
OCEAN_POINT = np.array([0.0, 0.0, 0.0])  # were  np.array([0, 0, 0]) in one of the files
OCEAN_NORMAL = np.array([0.0, 0.0, 1.0])  # were  np.array([0, 0, 1]) in one of the files

EARTH_RADIUS_M = 6371000.0
DBL_MAX = float("inf")

# FOUND_THRESHOLD = 128 # threshold for determining if baitball is found
# lower threshold
FOUND_THRESHOLD = 50
AVG_X_POS_COL = 0
AVG_Y_POS_COL = 1
GAUSS_MAX = 0.3989423  # sigma = 1 gives a peak value of ~0.4
MAX_NOISE = 100  # max pixel movement permitted between frames
LOW_WEIGHT = 1.0  # standard weight of a particle
HIGH_WEIGHT = 5.0  # higher weight when particle is in detected bb position
RESET_THRESHOLD = 120  # 200 is better for marlins and 120 for bb # reset particle filter after number of consecutive
# frames without detecting baitball
VELOCITY_BUFFER_SIZE = 60  # averaging velocity
POSITION_BUFFER_SIZE = 1  # averaging position
MAX_VELOCITY = 10.0  # filter out larger velocities
INTERPOLATION_RADIUS = 10  # create a circle around the detected baitball with this radius when interpolating
PARTICLE_WEIGHT_MULTIPLIER = 0.01  # multiply particle weight by this value
MIN_COMPONENT_SIZE = 1000  # minimum size of a connected component in the image in pixels
MAX_COMPONENT_SIZE = 200000  # maximum size of a connected component in the image in pixels
MINIMUM_WEIGHT = 0.1  # minimum weight of a particle
STD_DEV = 1.0  # standard deviation of gaussian noise
PARTICLE_RADIUS_DEFAULT = 5  # default radius of a particle
CONFIDENCE_THRESHOLD = 10

RAD2DEG = (180.0 / math.pi)
