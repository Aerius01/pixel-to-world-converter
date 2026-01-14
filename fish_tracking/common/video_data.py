import os
import cv2

class VideoData:
    """Class to hold video data and related parameters"""
    def __init__(self, video_file, video_name=None):
        self.video_file = video_file
        self.video_name = video_name if video_name else os.path.splitext(os.path.basename(video_file))[0]

        # Extract video properties
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_file}")
            
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.release()