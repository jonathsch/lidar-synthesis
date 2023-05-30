import os
from typing import Optional
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import open3d as o3d

from lidar_synthesis.utils.pc_utils import random_sampling

class CARLALidarDataset(Dataset):
    def __init__(self, root: os.PathLike,
        num_points: int = 8192,
        add_gaussian_noise: bool = False,
        use_histogram_sampling: bool = False,
        num_histogram_samples: Optional[int] = 200,
    ) -> None:

        super(CARLALidarDataset, self).__init__()
        self.root = Path(root)
        self.add_noise = add_gaussian_noise
        self.num_points = num_points
        self.use_histogram_sampling = use_histogram_sampling
        self.num_histogram_samples = num_histogram_samples

        sequences = [
            p
            for p in self.root.joinpath("town1").iterdir()
            if p.is_dir() and p.name.startswith("sequence_")
        ]
        self.index = []
        for seq in sequences:
            with open(seq / "labels.json", mode="r") as f:
                labels = json.load(f)

            for label in labels:
                lidar_file_name = f"{Path(label['filename']).stem}.ply"
                label["filename"] = str(Path(seq, "lidar", lidar_file_name).resolve())
            self.index.append(labels)

        self.index = [item for sublist in self.index for item in sublist]
        self.index_df = pd.DataFrame(self.index)

        if self.use_histogram_sampling:
            self.index_df = self.index_df.groupby(
                self.index_df["steering"].apply(lambda x: round(x, 1))
            ).apply(
                lambda x: x.sample(
                    self.num_histogram_samples, replace=True, ignore_index=True
                )
            )

        self.index_df.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.index_df)

    def __getitem__(self, idx):
        lidar_file = Path(self.root, self.index_df["filename"][idx])
        lidar_pc = o3d.t.io.read_point_cloud(str(lidar_file)).point["positions"].numpy()
        lidar_pc = lidar_pc[lidar_pc[:, 0] > 0.0]

        lidar_pc = random_sampling(lidar_pc, self.num_points).T.astype(np.float32)
        if self.add_noise:
            noise = np.random.normal(scale=0.01, size=len(lidar_pc.T) * 3).reshape(
                lidar_pc.shape
            )
            lidar_pc += noise

        return {
            "lidar": lidar_pc,
            "steering": np.float32(self.index_df["steering"][idx]),
        }