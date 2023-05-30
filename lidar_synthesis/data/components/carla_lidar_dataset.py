from typing import Optional
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import open3d as o3d

from lidar_synthesis.utils.pc_utils import random_sampling


class AugmentedPseudoLidarSet(Dataset):
    def __init__(
        self,
        root,
        add_gaussian_noise: bool = False,
        num_points: int = 8192,
        offset_limit: float = 1.5,
        use_histogram_sampling: bool = False,
        num_histogram_samples: Optional[int] = 200,
    ) -> None:
        super(AugmentedPseudoLidarSet, self).__init__()
        self.root = Path(root)
        self.add_noise = add_gaussian_noise
        self.num_points = num_points
        self.use_histogram_sampling = use_histogram_sampling
        self.num_histogram_samples = num_histogram_samples

        offset_limit = int(offset_limit * 10)

        self.index = []
        sequences = [
            p
            for p in self.root.joinpath("town1").iterdir()
            if p.is_dir() and p.name.startswith("sequence") and int(p.name[9]) <= 3
        ]

        for seq in sorted(sequences):
            with open(seq / "raycast" / "augmented-labels.json", mode="r") as f:
                labels = json.load(f)

            labels = [label for label in labels if int(label["file"][-6:-4]) <= offset_limit]
            for idx, label in enumerate(labels):
                label["file"] = str(Path(seq, "raycast", label["file"]).resolve())

            self.index.append(labels)

        self.index = [item for sublist in self.index for item in sublist]
        self.index_df = pd.DataFrame(self.index)

        # self.index_df = self.index_df[self.index_df["file"].str.endswith("-4.ply")]

        if self.use_histogram_sampling:
            self.index_df = self.index_df.groupby(
                self.index_df["dy"].apply(lambda x: round(x, 1))
            ).apply(lambda x: x.sample(self.num_histogram_samples, replace=True, ignore_index=True))

        self.index_df.reset_index(drop=True, inplace=True)
        print(f"Loaded {len(self.index_df)} samples")

    def __len__(self):
        return len(self.index_df)

    def __getitem__(self, idx):
        pc = o3d.t.io.read_point_cloud(str(self.index_df["file"][idx])).point["positions"].numpy()

        pc = random_sampling(pc, self.num_points).T.astype(np.float32)
        if self.add_noise:
            noise = np.random.normal(scale=0.005, size=len(pc.T) * 3).reshape(pc.shape)
            pc += noise

        dy = np.float32(self.index_df["dy"][idx])
        waypoints = np.array(self.index_df["waypoints"][idx]).astype(np.float32)
        waypoints = waypoints[:, 1:]
        # waypoints[1] *= -1

        return {
            "steering": dy,
            "waypoints": waypoints,
            "lidar": pc,
        }
