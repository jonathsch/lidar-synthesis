from typing import Optional
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
import lightning.pytorch as pl

from data.components.carla_lidar_dataset import AugmentedPseudoLidarSet
from data.components.carla_lidar_dataset import BaselineLidarDataset


class LidarDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        batch_size: int = 8,
        val_train_ratio: float = 0.2,
        num_workers: int = 4,
        add_gaussian_noise: bool = False,
        num_points: int = 8192,
        use_histogram_sampling: bool = False,
        num_histogram_samples: Optional[int] = 200,
    ) -> None:
        super(LidarDataModule, self).__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        root_dir = Path(self.hparams.root_dir).resolve()
        dataset = BaselineLidarDataset(
            root_dir,
            num_points=self.hparams.num_points,
            add_gaussian_noise=self.hparams.add_gaussian_noise,
            use_histogram_sampling=self.hparams.use_histogram_sampling,
            num_histogram_samples=self.hparams.num_histogram_samples,
        )

        val_size = int(self.hparams.val_train_ratio * len(dataset))
        train_size = len(dataset) - val_size
        self.train_set, self.val_set = random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
        )


class AugmentedPseudoLidarDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        batch_size: int = 8,
        val_train_ratio: float = 0.2,
        num_workers: int = 4,
        add_gaussian_noise: bool = False,
        num_points: int = 8192,
        max_offset: float = 1.5,
        use_histogram_sampling: bool = False,
        num_histogram_samples: Optional[int] = 200,
    ):
        super(AugmentedPseudoLidarDataModule, self).__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        # Dataset
        root_dir = Path(self.hparams.root_dir).resolve()
        dataset = AugmentedPseudoLidarSet(
            root_dir,
            add_gaussian_noise=self.hparams.add_gaussian_noise,
            num_points=self.hparams.num_points,
            offset_limit=self.hparams.max_offset,
            use_histogram_sampling=self.hparams.use_histogram_sampling,
            num_histogram_samples=self.hparams.num_histogram_samples,
        )

        # Train / Val split
        val_size = int(self.hparams.val_train_ratio * len(dataset))
        train_size = len(dataset) - val_size
        self.train_set, self.val_set = random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
        )