import os

import numpy as np
import torch
from torch import nn, optim
import lightning.pytorch as pl

from lidar_synthesis.models.components.pointnet2 import PointNet2Features
from lidar_synthesis.models.components.pointnet import PointNetfeat


class LitLidar2Waypoints(pl.LightningModule):
    def __init__(self, learning_rate: float = 1e-3):
        super(LitLidar2Waypoints, self).__init__()

        self.save_hyperparameters()

        # Network modules
        self.pointnet = PointNetfeat(feature_transform=True)
        self.features2waypoints = nn.Sequential(
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 8),
        )

        self.loss_fn = nn.L1Loss()

    def forward(self, x: torch.tensor):
        x, _, _ = self.pointnet(x)
        x = self.features2waypoints(x)
        return x

    def _general_step(self, batch, batch_idx):
        lidar_pcd = batch["lidar"]
        waypoints = batch["waypoints"]

        pred_waypoints = self.forward(lidar_pcd)
        loss = self.loss_fn(pred_waypoints, torch.flatten(waypoints, start_dim=1))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._general_step(batch, batch_idx)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._general_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return [optimizer], [{"scheduler": scheduler, "monitor": "val/loss", "interval": "epoch"}]
