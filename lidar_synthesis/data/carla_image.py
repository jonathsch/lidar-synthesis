from typing import Tuple, Optional, Dict, Any

import torch
from torch.utils.data import DataLoader, random_split
import lightning.pytorch as pl

from data.components.carla_image_dataset import CARLAImageDataset


class CARLAImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        val_split: float = 0.2,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super(CARLAImageDataModule, self).__init__()

        self.save_hyperparameters(logger=False)

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        dataset = CARLAImageDataset(self.hparams.data_dir)
        val_split = int(len(dataset) * self.hparams.val_split)
        train_split = len(dataset) - val_split

        self.train_set, self.val_set = random_split(dataset, [train_split, val_split])

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
