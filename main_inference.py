import torch
import cv2
import numpy as np
from pathlib import Path
from collections import deque
from utils.video_utils import read_video
from pose_estimation import load_pose_model, pose_inference
from fall_detection import load_tcn_model, tcn_inference


def extract_keypoints(pose_model, frames, keypoints, first):
    # Placeholder: Apply pose estimation on each frame and return keypoints.
    # keypoints = []
    if first:
        for frame in frames:
            # Apply pose model on each frame to extract keypoints
            xyn = pose_inference(frame, pose_model)
            if xyn.shape[0] == 0:
                xyn = np.zeros((17, 2))
            keypoints.append(xyn)
        first = False
    else:
        xyn = pose_inference(frames[-1], pose_model)
        if xyn.shape[0] == 0:
            xyn = np.zeros((17, 2))
        keypoints.append(xyn)
    return np.array(keypoints), first


def main_inference(video_source, pose_model_path, tcn_model_path, save_dir=None, show_video=False, seq_len=15):
    """
    Prepares a sliding window sequence of frames for inference function.

    :param video_source: video_source (str): Path to the video file or camera Index.
    :param pose_model_path: Path to the fine-tuned pose estimation model.
    :param tcn_model_path: Path to the trained TCN model.
    :param save_dir: Path to the directory for saving results, if provided.
    :param show_video: Whether to display the video stream (default: False).
    :param seq_len: Number of frames in the sequence (default 15).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pose_model = load_pose_model(pose_model_path)
    tcn_model = load_tcn_model(tcn_model_path)

    cap = read_video(video_source)
    if save_dir:
        Path.mkdir(save_dir, parents=True, exist_ok=True)
        result_file_name = Path(video_source).stem
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_dir/f'fall_detection_{result_file_name}.mp4', fourcc, fps, (frame_width, frame_height))

    sliding_window = deque(maxlen=seq_len)
    keypoints_window = deque(maxlen=seq_len)
    first = True
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        sliding_window.append(frame)

        if len(sliding_window) == seq_len:

            keypoints, first = extract_keypoints(pose_model, list(sliding_window), keypoints_window, first)
            # Convert keypoints to a PyTorch tensor
            keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32).to(device)

            # Reshape to match TCN input shape: (batch_size=1, num_keypoints=34, seq_len=15)
            keypoints_tensor = keypoints_tensor.reshape(15, -1).permute(1, 0)

            # Perform inference with TCN
            predicted_label = tcn_inference(keypoints_tensor, tcn_model, device)

            if predicted_label == 1:
                cv2.putText(frame, 'Fall Detected!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if show_video:
            cv2.imshow('Fall Detection Project', frame)

        if save_dir:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if save_dir:
        out.release()
    cv2.destroyAllWindows()
