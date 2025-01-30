# Fall Detection Model

## Overview
A computer vision-based fall detection system designed to identify falls in videos using pose estimation and Temporal Convolutional Networks (TCNs). The system processes 15-frame sliding windows, extracts normalized joint locations from 17 keypoints using pose estimation, and applies TCNs for accurate and efficient fall detection.

---
## Table of Contents
- [Introduction](#introduction)
- [Features](#Features)
- [Installation](#Installation)
- [Usage](#Usage)
- [References](#References)
- [Contributing](#contributing)
- [Acknowledgments](#Acknowledgments)

---

## Introduction
Falls are a major health concern, especially for elderly individuals or patients in hospitals. This project aims to provide an efficient fall detection model using pose estimation and Temporal Convolutional Networks (TCNs). The system processes 15-frame sliding windows, extracts normalized joint locations from 17 keypoints using pose estimation, and applies TCNs for fall classification.

---

## Features
- **Pose Estimation**: Extracts 17 keypoints from human poses in video frames.
- **Temporal Convolutional Networks (TCNs)**: Processes temporal data over 15-frame sliding windows for fall detection.
- **Efficient and Accurate**: Combines lightweight pose estimation with robust TCN models.

---

## Installation

### Cloning the Repository
```bash
git clone https://github.com/parichehrvn/fall-detection.git
cd fall-detection
```

### Prerequisites
- Python 3.9.20 or later
- Required libraries:
  ```bash
  pip install -r requirements.txt
  ```

---

## Usage
This project includes two main components: **pose estimation** and **fall detection** using Temporal Convolutional Networks (TCNs). As such, two types of datasets are required to train and evaluate each part.

## **Pose Estimation**
For the pose estimation component, we fine-tuned the yolov11-pose model from the Ultralytics library using the Fall_Simulation_Data [1] dataset. The goal was to adapt the pose model to accurately identify joint locations in fall scenarios.

### Data Preparation:

* We selected a subset of fall videos from the dataset.
* Keyframes were extracted and annotated in the YOLO pose format using CVAT.
* The processed dataset is located in the **cfg/dataset/pose directory**.

### Fine-Tuning the Pose Model:

If you wish to fine-tune the yolov11-pose model with your own dataset, you can use the Ultralytics YOLO library to train on your annotated data.
Place the fine-tuned model in the **cfg/models/pose** directory with the name **best_yolov11_pose.pt**.

## **Fall Detection using TCN**
For the fall detection component, we also used the Fall_Simulation_Data [1] dataset. This time, we selected videos categorized as Activities of Daily Living (ADL) and Fall scenarios to train the TCN model.

### Split Dataset:

The dataset has already been split into train, validation, and test sets for your convenience. You can find it in the **cfg/dataset/tcn/split_data** directory.
The dataset is organized into two main categories: **FALL** and **ADL**.

#### Using Your Own Data:

If you want to train the TCN model with your own data, ensure that the dataset follows the same structure (i.e., videos categorized into FALL and ADL).
Use the provided script to split your dataset into training, validation, and test sets

```bash
python tcn_cli.py split --input /path/to/input/directory --output /path/to/save/output/directory --train_ratio 0.7 --val_ratio 0.2 --test_ratio 0.1
```

### Extracting Key Frames and Keypoints
The next step involves extracting key frames from videos in the split dataset and generating keypoints for each frame using the provided pose estimation model.
Use the provided script to extract 15 key frames from each video and generate keypoints for each frame. Upon running this script, a separate folder is created for each video, containing:

* **15 key frames** extracted from the video.
* A **keypoints.csv** file, which stores the joint locations for each frame.

```bash
python tcn_cli.py extract --input /path/to/input/directory --output /path/to/save/output/directory --model /path/to/the/pose estimation/model
```

### Training TCN Model
The provided script enables you to train the Temporal Convolutional Network (TCN) model for fall detection. It uses a extracted dataset containing key frames and keypoints stored in "train" and "val" subdirectories.

#### Command-Line Arguments
The following options are available when training the TCN model:

* --dataset (Path, required):
Path to the extracted dataset directory. The directory must contain train and val subdirectories with the processed keypoints.

* --epochs (int, required):
The number of epochs for training the model.

* --batch_size (int, optional, default=2):
Number of samples per batch to load during training.

* --lr (float, optional, default=0.001):
Learning rate for the optimizer.

* --patience (int, optional, default=10):
Early stopping patience. If no improvement is observed in the validation loss after the specified number of epochs, training will stop.

* --save_path (Path, required):
Directory to save the best-trained model and training-related plots, such as the loss and accuracy curves.

```bash
python tcn_cli.py train --dataset /path/to/dataset --epochs 50 --batch_size 4 --lr 0.001 --patience 10 --save_path /path/to/save/model
```

### Evaluating the TCN Model
The provided script allows you to evaluate the performance of the trained Temporal Convolutional Network (TCN) model on a specified dataset. It computes various evaluation metrics, including accuracy, precision, recall, F1 score, and the confusion matrix.

#### Command-Line Arguments

The following options are available for the evaluation script:

* --model (Path, required):
Path to the trained TCN model to be evaluated.

* --data (Path, required):
Path to the extracted dataset directory for evaluation.

* --batch_size (int, optional, default=2):
Number of samples per batch to load during evaluation.

* --split (str, optional, default='val'):
Specifies the dataset split to evaluate. Options are:

   * "test": Evaluate on the test dataset.
   * "val": Evaluate on the validation dataset.

```bash
python tcn_cli.py evaluate --model /path/to/trained/tcn/model.pt --data /path/to/extracted/dataset/directory --batch_size 4 --split 'val'
```

## Final Inference: Fall Detection in Videos
This script performs the final fall detection inference on a video file or a live camera feed. It uses a sliding window approach to extract sequences of frames, applies pose estimation to extract keypoints, and passes the keypoints to the trained TCN model to detect falls in the video.

### Command-Line Arguments
The following options are available for the inference script:

* --source (str, required):
Path to the input video file or camera index for live video stream.

    * Example for video file: /path/to/video.mp4
    * Example for camera: 0 for the default webcam.
* --pose_model (Path, required):
Path to the fine-tuned pose estimation model (e.g., best_yolov11_pose.pt).

* --tcn_model (Path, required):
Path to the trained TCN model for fall detection.

* --save_dir (Path, optional, default=None):
Directory to save the output video with fall detection results. If not provided, the results are not saved.

* --show_video (bool, optional, default=False):
Whether to display the video stream with fall detection results during inference.

```bash
python fall_cli.py inference --source /path/to/video.mp4 --pose_model /path/to/best_yolov11_pose.pt --tcn_model /path/to/best_tcn_model.pt --save_dir /path/to/save/results --show_video False
```
---

## Results
### Example Output

![Fall Detection Example](results/fall_detection_fall95.mp4)
![Fall Detection Example](results/fall_detection_fall175.mp4)

---

## References
[1] Baldewijns, G., Debard, G., Mertes, G., Vanrumste, B., Croonenborghs, T. (2016). Bridging the gap between real-life data and simulated data by providing realistic fall dataset for evaluating camera-based fall detection algorithms. Healthcare Technology Letters.  
[2] I. Charfi, J. Mitéran, J. Dubois, M. Atri, R. Tourki, "Optimised spatio-temporal descriptors for real-time fall detection: comparison of SVM and Adaboost based classification, Journal of Electronic Imaging (JEI), Vol.22. Issue.4, pp.17, October 2013.

---

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue.

---

## Acknowledgments
Special thanks to the open-source libraries and datasets that made this project possible.
