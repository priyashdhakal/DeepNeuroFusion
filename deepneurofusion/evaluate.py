from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_NAMES = ["Acc", "Sens", "Spec", "Prec", "F1", "Gm", "AUC", "MCC", "BA"]


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    pos_label: int = 1,
) -> Dict[str, float]:
    """Compute the full suite of classification metrics.

    Returns a dict with keys: Acc, Sens, Spec, Prec, F1, Gm, AUC, MCC, BA.
    """
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        if pos_label == 0:
            tp, tn = tn, tp
            fp, fn = fn, fp

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1   = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
        gm   = np.sqrt(sens * spec)
        mcc  = matthews_corrcoef(y_true, y_pred)
        ba   = balanced_accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true == pos_label, y_proba[:, pos_label])
        except Exception:
            auc = 0.0
    else:
        sens = recall_score(y_true, y_pred, average="macro")
        spec = 0.0
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        f1   = f1_score(y_true, y_pred, average="macro")
        gm   = 0.0
        mcc  = matthews_corrcoef(y_true, y_pred)
        ba   = balanced_accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        except Exception:
            auc = 0.0

    return {
        "Acc": acc, "Sens": sens, "Spec": spec, "Prec": prec,
        "F1": f1, "Gm": gm, "AUC": auc, "MCC": mcc, "BA": ba,
    }
