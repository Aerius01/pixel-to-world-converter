# Pixel to World Converter
*Note: written by Claude Code*

Convert pixel coordinates from drone video tracking to world GPS coordinates using sensor fusion and camera projection.

## Overview

This package implements a 5-stage pipeline for converting 2D pixel coordinates from drone video tracking into 3D world GPS coordinates. It fuses GPS, IMU, and gimbal sensor data with camera geometry to accurately project tracked objects from the image plane to geographic coordinates.

**Key Features:**
- Sensor fusion using Kalman filtering and RTS smoothing for robust camera pose estimation
- Handles noisy GPS and IMU measurements
- Ray-plane intersection for pixel-to-world projection
- ENU (East-North-Up) coordinate system with configurable reference point
- Support for gimbal-mounted cameras with arbitrary orientations

## Installation

### Direct Install from GitHub

```bash
pip install git+https://github.com/Aerius01/pixel-to-world-converter.git
```

### Install from Source (for development)

```bash
git clone https://github.com/Aerius01/pixel-to-world-converter.git
cd pixel-to-world-converter
pip install -e .
```

> **Note:** This package is not currently published to PyPI.

## Quick Start

```python
import pandas as pd
from pixel_to_world_converter import convert

# Load your tracking data and sensor logs
tracks_df = pd.read_csv('tracks.csv')  # columns: id, x, y, frame, label
video_path = 'drone_video.mp4'
sensor_log_path = 'DJIFlightRecord.csv'

# Convert pixel tracks to GPS coordinates
world_trajectory = convert(
    image_tracks_csv=tracks_df,
    video_path=video_path,
    sensor_log_path=sensor_log_path
)

# Output contains GPS coordinates and velocities
print(world_trajectory[['avg_pos_x', 'avg_pos_y', 'frame', 'species_label']])
```

## Pipeline Architecture

The conversion process consists of 5 sequential stages:

### 1. Sensor Alignment
Maps video frame timestamps to the nearest sensor data timestamps, accounting for different sampling rates between the video and sensor logs.

### 2. Forward Kalman Filter
Estimates camera pose (position + velocity) from noisy GPS and IMU measurements using a constant velocity motion model. The filter predicts future states and corrects them with measurements.

**State Vector:** `[x, y, z, vx, vy, vz]` (position in meters, velocity in m/s)

### 3. RTS Smoother (Rauch-Tung-Striebel)
Performs a backward pass through the Kalman-filtered estimates to refine camera poses using "future knowledge" from the full trajectory. This reduces noise and improves accuracy.

### 4. Camera Extrinsics Computation
Computes 4x4 transformation matrices for each frame based on:
- Smoothed camera position from RTS
- Gimbal angles (roll, pitch, yaw) from sensor logs
- Drone orientation (converted from NED to ENU coordinate system)

### 5. Pixel-to-World Projection
Projects 2D pixel coordinates to 3D world coordinates using:
- Ray-plane intersection (assuming targets are at sea level)
- Camera intrinsic matrix (focal length, sensor size)
- Camera extrinsic matrices from stage 4

## Input Data Requirements

### Image Tracks CSV
Must contain columns: `id`, `x`, `y`, `frame`, `label`

```csv
id,x,y,frame,label
1,320.5,240.3,0,marlin
1,322.1,241.8,1,marlin
2,450.2,180.6,0,shark
```

### Sensor Log CSV
Must contain columns as defined in `DRONE_LOG_SCHEMA`:
- GPS: `latitude`, `longitude`, `altitude(m)`
- IMU: `velocityX(mps)`, `velocityY(mps)`, `velocityZ(mps)` (NED frame)
- Gimbal: `gimbalRollRaw`, `gimbalPitchRaw`, `gimbalYawRaw`
- Timestamp: `datetime(utc)`

DJI flight logs typically contain all required fields.

## Configuration

Adjust pipeline parameters in `pixel_to_world_converter/config.py`:

```python
# Kalman Filter Parameters
KF_MEASUREMENT_NOISE_SCALE = 100.0  # GPS/IMU measurement noise
KF_PROCESS_NOISE_SCALE = 1.0        # Motion model uncertainty
KF_INITIAL_COVARIANCE_SCALE = 1.0   # Initial state uncertainty

# Camera Parameters
FOCAL_LENGTH = 0.0088               # meters
SENSOR_WIDTH = 0.01276              # meters (DJI Mavic 3)
GIMBAL_PITCH_OFFSET = 0.03665191    # radians (90° - actual)
```

## Output Format

The output DataFrame contains smoothed world trajectories with the following columns:

| Column | Description |
|--------|-------------|
| `avg_pos_x` | World position in meters (East, ENU) |
| `avg_pos_y` | World position in meters (North, ENU) |
| `avg_vel_x` | Velocity in m/s (East direction) |
| `avg_vel_y` | Velocity in m/s (North direction) |
| `avg_vel` | Speed in m/s (magnitude) |
| `angle` | Heading in degrees (0=East, 90=North) |
| `frame` | Video frame number |
| `target_id` | Tracked object ID |
| `species_label` | Object classification label |
| `ref_latitude` | Reference GPS latitude (origin) |
| `ref_longitude` | Reference GPS longitude (origin) |
| `ref_altitude` | Reference altitude (sea level = 0) |

## Coordinate Systems

- **Input (Sensor):** NED (North-East-Down) from drone IMU
- **Processing:** ENU (East-North-Up) with origin at first GPS position
- **Output:** ENU coordinates in meters relative to reference GPS point

## API Reference

### Main Functions

#### `convert(image_tracks_csv, video_path, sensor_log_path)`
High-level function that handles file loading and validation.

**Parameters:**
- `image_tracks_csv`: Path to CSV or DataFrame with pixel tracks
- `video_path`: Path to video file (for extracting fps, dimensions)
- `sensor_log_path`: Path to sensor log CSV

**Returns:** DataFrame with world trajectory data

#### `convert_pixel_to_world(image_tracks_df, sensor_data_df, video_fps, video_width, video_height, video_num_frames)`
Low-level pipeline function for direct data processing.

**Parameters:**
- `image_tracks_df`: DataFrame with pixel coordinates
- `sensor_data_df`: DataFrame with sensor measurements
- `video_fps`: Frame rate (float)
- `video_width`, `video_height`: Video dimensions (int)
- `video_num_frames`: Total frame count (int)

**Returns:** DataFrame with world trajectory data

## Project Structure

```
pixel_to_world_converter/
├── __init__.py              # Package interface
├── converter.py             # High-level API with validation
├── pipeline.py              # Main 5-stage pipeline
├── config.py                # Configuration constants
├── schema.py                # Data validation schemas
├── preprocessing/
│   ├── sensor_alignment.py  # Frame-to-sensor timestamp mapping
├── pose_estimation/
│   ├── kalman.py            # Kalman filter implementations
│   ├── smoother.py          # RTS smoother
│   └── extrinsics.py        # Camera transformation matrices
└── projection/
    ├── pixel_to_world.py    # Ray-plane intersection
    └── trajectory.py        # Trajectory smoothing and processing
```

## TODO / Future Improvements

- **Replace custom Kalman/RTS with FilterPy**: The current implementation uses custom Kalman filter and RTS smoother classes. Consider migrating to [FilterPy](https://github.com/rlabbe/filterpy), a mature open-source library that provides well-tested implementations of Kalman filtering and RTS smoothing, reducing maintenance burden and potentially improving numerical stability.

- **Investigate optimal noise parameters**: The current static noise matrices date back to an earlier Python port. The original C++ implementation used time-adaptive process noise. Compare performance between static and adaptive noise models.

- **Add support for non-planar surfaces**: Current projection assumes targets are at sea level (z=0). Extend to support arbitrary ground planes or terrain elevation models.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

This implementation is based on research and code from Duc Pham's original C++ implementation. Special thanks to Pia for the initial Python port.