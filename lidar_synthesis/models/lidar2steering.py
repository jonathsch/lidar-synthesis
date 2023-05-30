import os

import numpy as np
import torch
from torch import nn, optim
import lightning.pytorch as pl

from lidar_synthesis.models.components.pointnet2 import PointNet2Features


class LitLidar2Steering(pl.LightningModule):
    def __init__(self, learning_rate: float = 1e-3, visualize_pcd: bool = False):
        super(LitLidar2Steering, self).__init__()

        self.save_hyperparameters()

        # Network modules
        self.pointnet = PointNet2Features()
        self.features2steering = nn.Sequential(
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.tanh = nn.Tanh()
        self.loss_fn = nn.L1Loss()

    def forward(self, x: torch.tensor):
        x, _, _ = self.pointnet(x)
        x = self.steering_regression(x)
        return self.tanh(x)

    def _general_step(self, batch, batch_idx):
        lidar_pcd = batch["lidar"]
        steering_gt = batch["steering"]

        steering_pred = self.forward(lidar_pcd)
        loss = self.loss_fn(steering_pred, steering_gt)
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
