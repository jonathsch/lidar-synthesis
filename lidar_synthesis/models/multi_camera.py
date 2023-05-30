import numpy as np
import torch
from torch import nn, optim
import lightning.pytorch as pl


class LitImageToSteering(pl.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        mlp: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        criterion: nn.Module,
    ):
        super(LitImageToSteering, self).__init__()

        self.save_hyperparameters(logger=False)

        # Components
        self.backbone = backbone
        self.mlp = mlp
        self.tanh = nn.Tanh()

        # Loss
        self.criterion = criterion

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.backbone(x)
        x = x.flatten(1)
        x = self.mlp(x)
        x = self.tanh(x)
        return x

    def general_step(self, batch, batch_idx):
        x, y = batch["image"], batch["steering"]
        y_hat = self.forward(x)
        return self.criterion(y_hat, y.view(-1, 1))

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                },
            }
        else:
            return {"optimizer": optimizer}
