from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind


def gene_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    n_select: int = 200,
    method: str = "wilcoxon",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select top-*n_select* genes from training data only.

    Parameters
    ----------
    X_train, X_val : ndarray  — scaled gene matrices
    y_train : ndarray          — encoded class labels (training set)
    n_select : int             — number of genes to retain
    method : {'wilcoxon', 'ttest', 'variance'}

    Returns
    -------
    X_train_sel, X_val_sel : ndarray  — reduced gene matrices
    selected_idx : ndarray            — column indices of selected genes
    """
    if method == "variance":
        variances = np.var(X_train, axis=0)
        top_idx = np.argsort(variances)[-n_select:]
        return X_train[:, top_idx], X_val[:, top_idx], top_idx

    n_features = X_train.shape[1]
    p_values = np.ones(n_features)
    classes = np.unique(y_train)
    mask0 = y_train == classes[0]
    mask1 = y_train == classes[1]

    test_fn = mannwhitneyu if method == "wilcoxon" else ttest_ind
    kwargs = {"alternative": "two-sided"} if method == "wilcoxon" else {}

    for i in range(n_features):
        try:
            _, p_values[i] = test_fn(X_train[mask0, i], X_train[mask1, i], **kwargs)
        except Exception:
            p_values[i] = 1.0

    top_idx = np.argsort(p_values)[:n_select]
    return X_train[:, top_idx], X_val[:, top_idx], top_idx
