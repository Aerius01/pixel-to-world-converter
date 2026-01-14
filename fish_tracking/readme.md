# Fish Tracking Module

This guide explains how to use the fish tracking module to track fish swarms in drone videos. The module processes videos through several stages to produce both 2D pixel trajectories and 3D world coordinates of tracked fish.

## Prerequisites

Before running the tracking system, ensure you have:

1. A trained model file (`.pt`), output from the fish detection training module
2. A drone video file (supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`, `.flv`)
3. A drone flight log file (`.csv`) containing GPS and IMU data
4. The conda environment installed from the `environment.yml` file

## Drone Flight Log Requirements

The drone flight log must be a CSV file with specific columns that provide essential telemetry data for the tracking system. This file is critical for converting 2D pixel coordinates to 3D world coordinates.

### Required Columns

| Column Name | Description | Units/Format | Notes |
|-------------|-------------|--------------|-------|
| **latitude** | GPS coordinates of the drone | Decimal degrees | Geographic coordinate |
| **longitude** | GPS coordinates of the drone | Decimal degrees | Geographic coordinate |
| **altitude(m)** | Height above sea level | Meters | Altitude measurement |
| **ultrasonicHeight(m)** | Height measured by ultrasonic sensor | Meters | Alternative height measurement |
| **speed(mps)** | Total speed | Meters per second | Overall drone speed |
| **distance(m)** | Distance from home point | Meters | How far from takeoff location |
| **time(millisecond)** | Timestamp | Milliseconds | Time since start of recording |
| **isTakingVideo** | Recording status | Boolean (0=no, 1=yes) | Indicates if video is being recorded |
| **pitch(deg)** | Drone pitch angle | Degrees | Rotation around lateral axis |
| **roll(deg)** | Drone roll angle | Degrees | Rotation around longitudinal axis |
| **yaw(deg)** | Drone yaw angle | Degrees | Rotation around vertical axis (heading) |
| **velocityX(mps)** | Velocity in X direction | Meters per second | In NED coordinate system |
| **velocityY(mps)** | Velocity in Y direction | Meters per second | In NED coordinate system |
| **velocityZ(mps)** | Velocity in Z direction | Meters per second | In NED coordinate system |
| **remainPowerPercent** | Remaining battery percentage | Percentage | Battery level |
| **flightmode** | Current flight mode | String | e.g., GPS, OPTI |
| **isflying** | Flight status | Boolean (0=no, 1=yes) | Indicates if drone is in flight |
| **MOV_Name** | Video filename | String | **Critical for video-log synchronization** |

### Example Data

Below is a sample of key columns from a DJI flight log:

| latitude | longitude | altitude(m) | ultrasonicHeight(m) | speed(mps) | distance(m) | time(millisecond) | isTakingVideo | pitch(deg) | roll(deg) | yaw(deg) | velocityX(mps) | velocityY(mps) | velocityZ(mps) | remainPowerPercent | flightmode | isflying | MOV_Name |
|----------|-----------|-------------|---------------------|------------|-------------|-------------------|---------------|------------|-----------|----------|----------------|----------------|----------------|---------------------|------------|----------|----------|
| 0 | 0 | 0.1 | 2.2 | 1.4 | 0.9 | 5 | 0 | 0 | 1.3 | -168.7 | 0 | 0.1 | -1.4 | 98 | OPTI | 1 | |
| 0 | 0 | 0.2 | 2.4 | 1.91 | 0.9 | 96 | 0 | 0.6 | 1.6 | -168.7 | -0.1 | 0.1 | -1.9 | 98 | OPTI | 1 | |
| 0 | 0 | 0.5 | 2.6 | 2.5 | 0.9 | 199 | 0 | 1.3 | 1.5 | -168.5 | -0.1 | 0 | -2.5 | 98 | OPTI | 1 | |
| 23.941659 | -111.909966 | 2.8 | 1.7 | 0.3 | 496.07 | 1218605 | 1 | -3 | -3.9 | -141.9 | 0 | -0.3 | 0 | 29 | GPS | 1 | DJI_0004 |
| 23.941659 | -111.909966 | 2.8 | 1.7 | 0.4 | 496.07 | 1218701 | 1 | -3.7 | -4.1 | -142.4 | 0 | -0.4 | 0 | 29 | GPS | 1 | DJI_0004 |
| 23.941659 | -111.909967 | 2.7 | 1.6 | 0.4 | 496.08 | 1218809 | 1 | -3 | -3.7 | -142.9 | 0 | -0.4 | 0 | 29 | GPS | 1 | DJI_0004 |

## Running the Tracking Pipeline

The main entry point for the tracking system is `run.py`, which is located one level up from the `fish_tracking` directory. It orchestrates the entire tracking process from video frames to final trajectory visualization.

### Basic Usage

    python3 run.py [options]

### Command Line Arguments

- `-v, --video_file`: Path to the drone video file (default: first video file in current directory)
- `-d, --drone_file`: Path to the drone flight log CSV file (default: `drone_flight_log.csv` in current directory)
- `-m, --model`: Path to the trained neural network weights (default: `model.pt` in current directory)
- `-o, --output_directory`: Directory for saving all outputs (default: `tracking-output` in current directory)

### Example

At it's simplest, the code can be executed with the following command if all files are in the present working directory and are appropriately named:

```bash
python3 /path/to/fish_tracking/run.py
```

For more flexibility, you can specify locations for the required files and instead execute the script from any directory:

```bash
python3 /path/to/fish_tracking/run.py -v /path/to/video.mp4 -d /path/to/drone_flight_log.csv -m /path/to/weights_BEST_1.pt
```

If you want to, you can also specify an output directory. It will be created automatically in the present working directory if it doesn't already exist:

```bash
python3 /path/to/fish_tracking/run.py -v /path/to/video.mp4 -d /path/to/drone_flight_log.csv -m /path/to/weights_BEST_1.pt -o /path/to/output_directory
```

Note that all path can be specified as relative paths.

## Pipeline Stages

The tracking framework processes videos through the following stages:

### 1. Frame Extraction

First, the system extracts individual frames from the input video:

```
Extracting video frames...
```

Frames are saved as PNG files in the `extracted-frames` subdirectory of your output folder.

### 2. Soft Segmentation Mask Generation

Next, the trained neural network generates probability maps for each frame:

```
Generating soft segmentation masks on extracted frames...
```

These masks highlight regions likely to contain fish swarms. The masks are saved in the `soft-masks` subdirectory.

### 3. Pixel Trajectory Computation

The system then applies a particle filter to track fish swarms across frames:

```
Computing pixel trajectories...
```

This stage tracks the fish in 2D pixel coordinates, accounting for their movement between frames.

### 4. World Coordinate Conversion

The 2D pixel trajectories are converted to 3D world coordinates using drone telemetry data:

```
Converting pixel trajectories to world coordinates...
```

This step combines video data with drone GPS and IMU (inertial measurement unit) information to place the fish in real-world geographic coordinates.

### 5. Trajectory Visualization

Finally, the system generates visualizations of the tracking results:

```
Displaying trajectories...
```

A video file showing both the original footage and the tracked trajectories is saved in your output directory.

## Output Files

After running the pipeline, you'll find these key outputs in your specified output directory:

- `world-trajectory.csv`: CSV file containing the 3D world coordinates of tracked fish
- `world-tracks.avi`: Video visualization of the tracking results
- `pf_coords_0.csv`: Particle filter coordinates for each frame
- `pf_pixel_stats.csv`: Statistical data about the tracked objects in pixel space

## Troubleshooting

- If you see an error about mismatched frame counts, delete any existing frames in the output directory.
- Ensure your drone flight log file is properly formatted with GPS and IMU data.
- If tracking quality is poor, you may need to retrain your model with more data or adjust parameters.

