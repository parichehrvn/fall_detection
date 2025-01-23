import cv2
import numpy as np
import torch

def extract_key_frames(frames, num_frames=10):
    """
    Uniformly samples key frames from a list of frames.
    Args:
        frames (list): List of frames.
        num_frames (int): Number of key frames to extract.
    Returns:
        list: List of key frames.
    """
    frame_indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
    return [frames[i] for i in frame_indices]