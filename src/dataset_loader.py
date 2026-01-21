import numpy as np
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset

class ViolenceDataset(Dataset):
    def __init__(self, csv_file, num_frames=16, transform=None):
        """
        Args:
            csv_file (str): Path to the csv file (train.csv, val.csv, or test.csv)
            num_frames (int): Number of frames to sample from each video
            transform: Optional transforms to apply on frames
        """
        self.data = pd.read_csv(csv_file)
        self.num_frames = num_frames
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path = self.data.iloc[idx, 0]
        label = int(self.data.iloc[idx, 1])

        frames = self._load_video(video_path, self.num_frames)

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        processed_frames = []
        for f in frames:
            if isinstance(f, torch.Tensor):
                # already a tensor (C,H,W)
                processed_frames.append(f)
            else:
                # numpy array (H,W,C) → torch (C,H,W)
                processed_frames.append(torch.from_numpy(f).permute(2, 0, 1) / 255.0)

        frames = torch.stack(processed_frames)  # (T, C, H, W)

        return frames.float(), torch.tensor(label).long()

    def _load_video(self, path, num_frames):
        """Load video and sample num_frames evenly spaced frames"""
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            raise ValueError(f"Video {path} has 0 frames!")

        # pick evenly spaced indices
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in frame_indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (112, 112))  # resize for CNN
                frames.append(frame)
        cap.release()

        # if video shorter than required frames → pad by repeating last frame
        while len(frames) < num_frames and len(frames) > 0:
            frames.append(frames[-1])

        return frames
