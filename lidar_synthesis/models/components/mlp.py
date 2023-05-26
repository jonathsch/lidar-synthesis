import torch
from torch import nn


class MLP(nn.Module):
    """Flexible MLP template."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        out_dim: int,
        activation: nn.Module = nn.ReLU(),
        normalization: nn.Module = None,
    ):
        super().__init__()

        self.model = nn.Sequential(nn.Linear(in_dim, hidden_dim))
        if normalization is not None:
            self.model.append(normalization(hidden_dim))
        self.model.append(activation)

        for _ in range(num_layers - 1):
            self.model.append(nn.Linear(hidden_dim, hidden_dim))
            if normalization is not None:
                self.model.append(normalization(hidden_dim))
            self.model.append(activation)

        self.model.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.model(x)
