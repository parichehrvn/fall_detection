from ultralytics import YOLO

import numpy as np


def load_pose_model(model_path):
    model = YOLO(model_path)
    return model


def pose_inference(image: np.ndarray, model):
    """
    Performs pose estimation inference on a single image.

    :param image: Input image as a NumPy array.
    :param model: Loaded YOLO model for pose estimation.
    :return: Normalized keypoints (xyn).
    """
    results = model(image)[0]
    keypoints = results.keypoints
    xyn = keypoints.xyn[0]
    return xyn
