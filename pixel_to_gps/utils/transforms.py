"""Coordinate transformation utilities for geodetic conversions."""

import math
import numpy as np
from ..config import DEG2RAD, EARTH_RADIUS_M


def geo_to_cartesian(geo, reference_geo):
    """Convert geodetic coordinates to Cartesian coordinates relative to a reference point.

    Args:
        geo: Geodetic coordinates [latitude, longitude, altitude]
        reference_geo: Reference geodetic coordinates [latitude, longitude, altitude]

    Returns:
        np.array: Cartesian coordinates [x, y, z] in meters
    """
    r = np.array([0.0, 0.0, 0.0])
    same_lon = np.array([geo[0], reference_geo[1], 0])
    zero_alt = np.array([geo[0], geo[1], 0])

    r[1] = geo_distance(reference_geo, same_lon)
    total_distance = geo_distance(reference_geo, zero_alt)
    r[0] = math.sqrt(math.pow(total_distance, 2.0) - math.pow(r[1], 2.0))
    r[2] = geo[2]

    if (geo[0] > reference_geo[0] and r[1] < 0) or (geo[0] < reference_geo[0] and r[1] > 0):
        r[1] = -r[1]
    if (geo[1] > reference_geo[1] and r[0] < 0) or (geo[1] < reference_geo[1] and r[0] > 0):
        r[0] = -r[0]

    return r


def geo_distance(a, b):
    """Calculate the distance between two geodetic coordinates using the Haversine formula.

    Args:
        a: First geodetic coordinate [latitude, longitude, altitude]
        b: Second geodetic coordinate [latitude, longitude, altitude]

    Returns:
        float: Distance in meters
    """
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
