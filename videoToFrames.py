import cv2
import os
import gdown

def download_video_from_google_drive(url, output_path):
    gdown.download(url, output_path, quiet=False)

def extract_frames_at_intervals(video_path, output_dir, interval_seconds):
    vidcap = cv2.VideoCapture(video_path)

    if not vidcap.isOpened():
        print("Error opening video file:", video_path)
        return

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval_frames = int(interval_seconds * fps)

    print(f"Total frames in video: {total_frames}")
    print(f"Frames per second (FPS): {fps}")
    print(f"Interval frames to skip: {interval_frames}")

    success, image = vidcap.read()
    count = 0
    frame_number = 0
    while success:
        if count % interval_frames == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_number:04d}.jpg")
            cv2.imwrite(frame_path, image)
            print(f'Saved frame {frame_number} at {count / fps} seconds')

            frame_number += 1
        
        success, image = vidcap.read()
        count += 1
    
    vidcap.release()
    print(f"Extracted {frame_number} frames at intervals of {interval_seconds} seconds.")

def vid():
    drive_url = "https://drive.google.com/uc?id=1feSJy4w4Tn_wNJCPEP8yqxX-8IdEHrDH"
    local_video_path = "00002.MTS" 
    download_video_from_google_drive(drive_url, local_video_path)
    
    output_dir = "School Video Frames/2" 
    os.makedirs(output_dir, exist_ok=True)

    extract_frames_at_intervals(local_video_path, output_dir, 60) 

    os.remove(local_video_path)
    print(f"Deleted the temporary video file.")

vid()
