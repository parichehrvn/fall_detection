import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from utils.video_utils import extract_all_frames
from utils.frame_utils import extract_key_frames
from pose_estimation import load_pose_model, pose_inference


def extract_and_save_key_frames(video_path: Path, save_path: Path, num_keyframes=15):
    """
    Extract and save key frames from a video.
    Args:
        video_path (Path): Path to the video file.
        save_path (Path): Path to save key frames.
        num_keyframes (int): Number of key frames to extract from each video.
    """

    frames = extract_all_frames(video_path)
    key_frames = extract_key_frames(frames, num_keyframes)

    # Save key frames
    for i, frame in enumerate(key_frames):
        frame_path = save_path / f"frame_{i}.jpg"
        cv2.imwrite(frame_path, frame)


def extract_keypoints_with_pose_model(frame_paths: list(), model_path: Path, output_csv_path: Path):
    """
    Processes labeled frames with a pose model and saves keypoints to a CSV file.

    Args:
        frame_paths (list): List of frames Paths.
        model_path (Path): Path to the fine-tuned pose estimation model.
        output_csv_path (Path): Path to save the resulting keypoints as a CSV file.
        """
    model = load_pose_model(model_path)
    with open(output_csv_path / 'keypoints.csv', 'a+') as f:
        for frame_path in frame_paths:
            xyn = pose_inference(frame_path, model)
            if xyn.shape[0] == 0:
                xyn = np.zeros((17, 2), dtype=float)
            row = [f"{float(x):.4f},{float(y):.4f}" for x,y in xyn]
            f.writelines(','.join(row) + '\n')
