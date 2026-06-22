from __future__ import annotations

import copy
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class ModalityEncoder(nn.Module):
    """Single-modality feature encoder: Linear → BN → ReLU → Dropout."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultimodalNet(nn.Module):
    """Late-fusion multimodal classifier.

    MRI  → ModalityEncoder → ┐
                              ├ concat → FC(64→32) → BN → ReLU → Dropout → FC(32→C)
    Gene → ModalityEncoder → ┘
    """

    def __init__(
        self,
        mri_dim: int,
        gene_dim: int,
        num_classes: int,
        hidden: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.mri_encoder  = ModalityEncoder(mri_dim,  hidden, dropout)
        self.gene_encoder = ModalityEncoder(gene_dim, hidden, dropout)

        fused_dim = hidden * 2
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_embeddings(self, x_mri: torch.Tensor, x_gene: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.mri_encoder(x_mri), self.gene_encoder(x_gene)], dim=1)

    def forward(self, x_mri: torch.Tensor, x_gene: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.get_embeddings(x_mri, x_gene))


class MultimodalDataset(Dataset):
    """PyTorch Dataset for paired MRI / gene data."""

    def __init__(self, X_mri: np.ndarray, X_gene: np.ndarray, y: np.ndarray):
        self.X_mri  = torch.FloatTensor(X_mri)
        self.X_gene = torch.FloatTensor(X_gene)
        self.y      = torch.LongTensor(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X_mri[idx], self.X_gene[idx], self.y[idx]
