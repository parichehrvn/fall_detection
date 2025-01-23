from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
import torch
import os


class FallDetectionDataset(Dataset):
    def __init__(self, root_dir, split='train', seq_len=15, num_keypoints=34):
        """
        Args:
            root_dir (str): Root directory containing `train`, `val`, and `test` subdirectories.
            split (str): One of 'train', 'val', or 'test'.
            seq_len (int): Number of frames in the sequence (default 15).
            num_keypoints (int): Number of keypoints (default 34).
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.seq_len = seq_len
        self.num_keypoints = num_keypoints
        self.sequence_paths, self.labels = self._load_data_paths()

    def _load_data_paths(self):
        sequence_paths = []
        labels = []

        # Iterate through both ADL and FALL directories
        for label, class_name in enumerate(['ADL', 'FALL']):
            class_dir = self.root_dir / self.split / class_name
            for folder in class_dir.iterdir():
                if folder.is_dir():
                    keypoint_path = folder / 'keypoints.csv'
                    sequence_paths.append(keypoint_path)
                    labels.append(label)

        return sequence_paths, labels

    def __len__(self):
        return len(self.sequence_paths)

    def __getitem__(self, idx):
        keypoint_path = self.sequence_paths[idx]
        label = self.labels[idx]

        # Load and process the keypoints
        keypoints = pd.read_csv(keypoint_path, header=None).to_numpy()  # Ignore the first column (frame)
        keypoints = torch.tensor(keypoints, dtype=torch.float32)  # Shape: (seq_len, num_keypoints*2)

        # If necessary, you can pad or truncate the sequence to ensure it's exactly `seq_len`
        # if keypoints.shape[0] < self.seq_len:
        #     padding = torch.zeros(self.seq_len - keypoints.shape[0], self.num_keypoints * 2)
        #     keypoints = torch.cat((keypoints, padding), dim=0)  # Pad if sequence is too short
        # elif keypoints.shape[0] > self.seq_len:
        #     keypoints = keypoints[:self.seq_len]  # Truncate if sequence is too long

        return {'keypoints': keypoints, 'label': label}
