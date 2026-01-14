import cv2
import os
from tqdm import tqdm
import glob
from fish_tracking.common.path_manager import PathManager

def extract_frames(path_manager: PathManager):
    # Open the video
    cap = cv2.VideoCapture(path_manager.video_file)
    
    # Get total frame count for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
      
    # Check if frames are already extracted
    existing_frames = len(glob.glob(os.path.join(path_manager.extracted_frames_dir, "image_*.png")))
    
    if existing_frames == total_frames:
        print(f"Skipping extraction: supplied video has {total_frames} frames, and {existing_frames} frames already exist in output directory.")
        return
    elif existing_frames > 0 and existing_frames != total_frames:
        error_msg = f"Error: Found {existing_frames} frames in output directory, but video has {total_frames} frames. Please delete existing frames or specify a different output directory using the -o flag."
        raise ValueError(error_msg)
    
    # Create progress bar
    with tqdm(total=total_frames) as pbar:
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(os.path.join(path_manager.extracted_frames_dir, f"image_{count+1:05d}.png"), frame)
                count += 1
                pbar.update(1)
            else:
                break
                
    cap.release()
    cv2.destroyAllWindows()