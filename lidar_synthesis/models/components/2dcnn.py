from typing import List, Optional

import torch
from torch import nn


class CNNBackbone(nn.Module):
    """CNN Backbone for 2D feature extraction.

    Reference: https://arxiv.org/abs/1604.07316
    """

    def __init__(self):
        super(CNNBackbone, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2),  # 3x256x512 -> 24x128x256
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # 24x128x256 -> 24x64x128
            nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=2), # 24x64x128 -> 36x32x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # 36x32x64 -> 36x16x32
            nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=2), # 36x16x32 -> 48x8x16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # 48x8x16 -> 48x4x8
            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1), # 48x4x8 -> 64x4x8
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # 64x4x8 -> 64x4x8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # 64x4x8 -> 64x2x4
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.model(x)
        return x
