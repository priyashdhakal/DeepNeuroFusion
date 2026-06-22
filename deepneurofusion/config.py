from __future__ import annotations

import numpy as np
import torch

SEED = 42

PALETTE = {
    "mri":    "#2980B9",
    "gene":   "#C0392B",
    "ours":   "#E74C3C",
    "accent": "#27AE60",
    "ad":     "#E74C3C",
    "mci":    "#F39C12",
    "cn":     "#2ECC71",
    "ctl":    "#2ECC71",
}

TASK_NAMES = ["AD_vs_CN", "AD_vs_MCI", "MCI_vs_CN"]

POS_CLASS = {
    "AD_vs_CN":  "AD",
    "AD_vs_MCI": "AD",
    "MCI_vs_CN": "MCI",
}

TASK_PARAMS: dict = {
    "AD_vs_CN": {
        "hidden": 32, "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-2,
        "batch_size": 16, "n_genes": 181, "svm_C": 1.0, "dl_weight": 0.25,
        "gene_method": "wilcoxon",
    },
    "AD_vs_MCI": {
        "hidden": 32, "dropout": 0.35, "lr": 5e-4, "weight_decay": 1e-2,
        "batch_size": 16, "n_genes": 211, "svm_C": 1.0, "dl_weight": 0.2,
        "gene_method": "wilcoxon",
    },
    "MCI_vs_CN": {
        "hidden": 32, "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-2,
        "batch_size": 24, "n_genes": 300, "svm_C": 1.0, "dl_weight": 0.15,
        "gene_method": "variance",
    },
}


def _select_device(gpu_id: int = 0) -> torch.device:
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        idx = gpu_id if gpu_id < n else 0
        return torch.device(f"cuda:{idx}")
    return torch.device("cpu")


DEVICE = _select_device(0)


def set_device(gpu_id: int = 0) -> torch.device:
    global DEVICE
    DEVICE = _select_device(gpu_id)
    return DEVICE


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
