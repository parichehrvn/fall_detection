import cv2
import numpy as np

def read_video(video_source):
    """
    Opens a video file and returns a VideoCapture object.
    Args:
        video_source (str): Path to the video file or camera Index.
    Returns:
        cv2.VideoCapture: VideoCapture object for the video.
    """

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        if video_source.isdigit():
            raise ValueError(f"Unable to access the camera at index {video_source}.")
        elif not Path(video_source).is_file():
            raise ValueError(f"The video file '{video_source}' does not exist.")
        else:
            raise ValueError(f"Unable to open video file: {video_source}")

    return cap

def extract_all_frames(video_path):
    """
    Extracts all frames from a video.
    Args:
        video_path (str): Path to the video file.
    Returns:
        list: List of all frames in the video.
    """
    cap = read_video(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames
