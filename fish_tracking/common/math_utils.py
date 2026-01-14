from  fish_tracking.common.globals import *
import random

def gauss_function(x):
    pre_mult = 1.0 / (STD_DEV * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x / STD_DEV) ** 2)
    return pre_mult * math.exp(exponent)

def create_rand_decimal(max_value):
    RAND_ACCURACY = 10000.0
    if max_value == 0:
        return 0.0
    tmp_rand = float(random.randint(0, int(RAND_ACCURACY * max_value)))
    return tmp_rand / RAND_ACCURACY

def get_quaternion_from_euler(euler: np.array):
    cy = math.cos(euler[2] * 0.5)
    sy = math.sin(euler[2] * 0.5)
    cp = math.cos(euler[0] * 0.5)
    sp = math.sin(euler[0] * 0.5)
    cr = math.cos(euler[1] * 0.5)
    sr = math.sin(euler[1] * 0.5)

    q = np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy]
    )

    return q

def get_euler_from_quaternion(q):
    r = np.array([0.0, 0.0, 0.0])

    # roll (x-axis rotation)
    sinr_cosp = 2 * (q[0] * q[1] + q[2] * q[3])
    cosr_cosp = 1 - 2 * (q[1] * q[1] + q[2] * q[2])
    r[1] = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (q[0] * q[2] - q[3] * q[1])
    if abs(sinp) >= 1:
        r[0] = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        r[0] = math.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (q[0] * q[3] + q[1] * q[2])
    cosy_cosp = 1 - 2 * (q[2] * q[2] + q[3] * q[3])
    r[2] = math.atan2(siny_cosp, cosy_cosp)

    return r

def create_matrix_from_quaternion(q):
    r = np.zeros((3,3))

    r[0][0] = 2.0 * (q[0] * q[0] + q[1] * q[1]) - 1.0
    r[0][1] = 2.0 * (q[1] * q[2] - q[0] * q[3])
    r[0][2] = 2.0 * (q[1] * q[3] + q[0] * q[2])

    r[1][0] = 2.0 * (q[1] * q[2] + q[0] * q[3])
    r[1][1] = 2.0 * (q[0] * q[0] + q[2] * q[2]) - 1.0
    r[1][2] = 2.0 * (q[2] * q[3] - q[0] * q[1])

    r[2][0] = 2.0 * (q[1] * q[3] - q[0] * q[2])
    r[2][1] = 2.0 * (q[2] * q[3] + q[0] * q[1])
    r[2][2] = 2.0 * (q[0] * q[0] + q[3] * q[3]) - 1.0

    return r


def normalize_vector(in_vector):
    if in_vector[0]== 0.0 and in_vector[1] == 0.0 and in_vector[2] == 0.0:
        return in_vector

    root = math.sqrt(in_vector[0] * in_vector[0] + in_vector[1] * in_vector[1] + in_vector[2] * in_vector[2])
    result = np.array([0.0, 0.0, 0.0])
    result[0] = in_vector[0] / root
    result[1] = in_vector[1] / root
    result[2] = in_vector[2] / root
    return result


'''
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
    '''
reference_geo = np.array([0.0,0.0,0.0])

def line_plane_intersection(plane_point, plane_normal, line_point, line_direction):
    r = np.array([DBL_MAX, DBL_MAX, DBL_MAX])
    d = np.dot(plane_normal, normalize_vector(line_direction))
    if math.isclose(d, 0):
        return r

    if abs(d) > 0.0000001:
        w = line_point-  plane_point
        scalar = -np.dot(plane_normal, w) / d
        u = scalar* normalize_vector(line_direction)
        r = line_point+ u

    return r



def geo_to_cartesian(geo, reference_geo):

    r = np.array([0.0,0.0,0.0])
    same_lon = np.array([geo[0],reference_geo[1],0])

    zero_alt = np.array([geo[0],geo[1],0])

    r[1] = geo_distance(reference_geo, same_lon)
    total_distance = geo_distance(reference_geo, zero_alt)
    r[0] = math.sqrt(math.pow(total_distance, 2.0) - math.pow(r[1], 2.0))
    r[2] = geo[2]

    if (geo[0]  > reference_geo[0] and r[1] < 0) or (geo[0] < reference_geo[0] and r[1] > 0):
        r[1] = -r[1]
    if (geo[1] > reference_geo[1] and r[0] < 0) or (geo[1] < reference_geo[1] and r[0] > 0):
        r[0] = -r[0]

    return r

def geo_distance(a, b):
    lat_distance = (b[0] - a[0]) * DEG2RAD
    lon_distance = (b[1] - a[1]) * DEG2RAD
    c = math.pow(math.sin(lat_distance / 2.0), 2.0) + \
        math.cos(a[0] * DEG2RAD) * math.cos(b[0] * DEG2RAD) * \
        math.pow(math.sin(lon_distance / 2.0), 2.0)
    d = 2 * math.atan2(math.sqrt(c), math.sqrt(1 - c))
    distance = EARTH_RADIUS_M * d
    height = b[2] - a[2]

    distance = math.pow(distance, 2.0) + math.pow(height, 2.0)

    return math.sqrt(distance)
