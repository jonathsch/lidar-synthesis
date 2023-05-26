import os
from pathlib import Path
import json
from PIL import Image

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor


class CARLAImageDataset(Dataset):
    def __init__(self, root_dir: os.PathLike) -> None:
        super(CARLAImageDataset, self).__init__()
        self.root_dir = Path(root_dir)

        # Transforms
        self.to_tensor = ToTensor()

        # Build index
        index = []
        town_dirs = [p for p in self.root_dir.iterdir() if p.is_dir()]
        for town_dir in town_dirs:
            sequences = [p for p in town_dir.iterdir() if p.is_dir()]
            for sequence in sequences:
                with open(sequence / "labels.json", mode="r") as fp:
                    labels = json.load(fp)

                for frame in labels:
                    left_file = sequence / "rgb_left" / frame["filename"]
                    center_file = sequence / "rgb_center" / frame["filename"]
                    right_file = sequence / "rgb_right" / frame["filename"]
                    steering = frame["steering"]

                    index.append((left_file, steering + 0.1))
                    index.append((center_file, steering))
                    index.append((right_file, steering - 0.1))

        self.index = index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int):
        img = Image.open(self.index[index][0]).convert("RGB")
        # except Exception as e:
        #     print(f"{type(e)}: {str(e)}, file: {self.index[index][0]}")
        
        img = self.to_tensor(img).float()
        steering = torch.tensor(self.index[index][1]).float()

        return {"image": img, "steering": steering}
