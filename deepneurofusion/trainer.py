from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .config import DEVICE, SEED, TASK_PARAMS, POS_CLASS
from .evaluate import compute_all_metrics
from .gene_select import gene_selection
from .models import MultimodalDataset, MultimodalNet


class Trainer:
    """Encapsulates model creation, training loop, and inference."""

    def __init__(
        self,
        mri_dim: int,
        gene_dim: int,
        num_classes: int,
        class_weights: Optional[Dict[int, float]] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        dropout: float = 0.3,
        hidden: int = 32,
    ):
        self.model = MultimodalNet(mri_dim, gene_dim, num_classes, hidden, dropout).to(DEVICE)

        weights = None
        if class_weights is not None:
            weights = torch.FloatTensor(
                [class_weights[i] for i in range(num_classes)]
            ).to(DEVICE)

        self.ce_loss = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6
        )
        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []
        }

    def _train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x_mri, x_gene, y in loader:
            x_mri, x_gene, y = x_mri.to(DEVICE), x_gene.to(DEVICE), y.to(DEVICE)
            self.optimizer.zero_grad()
            out = self.model(x_mri, x_gene)
            loss = self.ce_loss(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item() * x_mri.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        return total_loss / total, correct / total

    @torch.no_grad()
    def _validate_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        ce = nn.CrossEntropyLoss()
        for x_mri, x_gene, y in loader:
            x_mri, x_gene, y = x_mri.to(DEVICE), x_gene.to(DEVICE), y.to(DEVICE)
            out = self.model(x_mri, x_gene)
            total_loss += ce(out, y).item() * x_mri.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        return total_loss / total, correct / total

    def fit(
        self,
        X_mri_tr: np.ndarray,
        X_gene_tr: np.ndarray,
        y_tr: np.ndarray,
        X_mri_va: np.ndarray,
        X_gene_va: np.ndarray,
        y_va: np.ndarray,
        epochs: int = 100,
        batch_size: int = 16,
        patience: int = 25,
    ) -> None:
        train_loader = DataLoader(
            MultimodalDataset(X_mri_tr, X_gene_tr, y_tr),
            batch_size=batch_size, shuffle=True, drop_last=False,
        )
        val_loader = DataLoader(
            MultimodalDataset(X_mri_va, X_gene_va, y_va),
            batch_size=batch_size, shuffle=False,
        )

        best_val_loss = float("inf")
        patience_ctr = 0
        best_weights = copy.deepcopy(self.model.state_dict())

        for _ in range(epochs):
            tr_loss, tr_acc = self._train_epoch(train_loader)
            va_loss, va_acc = self._validate_epoch(val_loader)
            self.scheduler.step(va_loss)

            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(va_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_acc"].append(va_acc)

            if va_loss < best_val_loss - 1e-4:
                best_val_loss = va_loss
                patience_ctr = 0
                best_weights = copy.deepcopy(self.model.state_dict())
            else:
                patience_ctr += 1
            if patience_ctr >= patience:
                break

        self.model.load_state_dict(best_weights)

    @torch.no_grad()
    def predict_proba(self, X_mri: np.ndarray, X_gene: np.ndarray) -> np.ndarray:
        self.model.eval()
        out = self.model(
            torch.FloatTensor(X_mri).to(DEVICE),
            torch.FloatTensor(X_gene).to(DEVICE),
        )
        return F.softmax(out, dim=1).cpu().numpy()

    @torch.no_grad()
    def get_embeddings_np(self, X_mri: np.ndarray, X_gene: np.ndarray) -> np.ndarray:
        self.model.eval()
        emb = self.model.get_embeddings(
            torch.FloatTensor(X_mri).to(DEVICE),
            torch.FloatTensor(X_gene).to(DEVICE),
        )
        return emb.cpu().numpy()


def run_task(
    task_name: str,
    X_mri_raw: np.ndarray,
    X_gene_raw: np.ndarray,
    y_all: np.ndarray,
    mask: np.ndarray,
    mri_feat_names: List[str],
    gene_feat_names: List[str],
    n_repeats: int = 10,
    n_folds: int = 5,
) -> dict:
    """Execute the full repeated stratified k-fold evaluation for one binary task.

    Parameters
    ----------
    task_name : 'AD_vs_CN' | 'AD_vs_MCI' | 'MCI_vs_CN'
    X_mri_raw, X_gene_raw : full (unmasked) feature matrices
    y_all : full diagnosis labels
    mask : boolean mask selecting rows for this binary task
    n_repeats, n_folds : evaluation protocol (default 10 × 5-fold)

    Returns
    -------
    dict with result_summary, per-fold metrics, confusion matrix, ROC data,
    and last-fold artefacts for SHAP.
    """
    X_mri  = X_mri_raw[mask]
    X_gene = X_gene_raw[mask]
    y      = y_all[mask]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)

    print(f"\n{'='*70}")
    print(f"TASK: {task_name}")
    print(f"{'='*70}")
    print(f"  Samples: {len(y_enc)}, Classes: {le.classes_}, Dist: {np.bincount(y_enc)}")

    pos_cls   = POS_CLASS.get(task_name, le.classes_[1])
    pos_label = list(le.classes_).index(pos_cls) if pos_cls in le.classes_ else 1
    print(f"  Positive class: {le.classes_[pos_label]} (index {pos_label})")

    cw_array     = compute_class_weight("balanced", classes=np.unique(y_enc), y=y_enc)
    class_weights = dict(enumerate(cw_array))

    params      = TASK_PARAMS.get(task_name, TASK_PARAMS["AD_vs_CN"])
    k_gene      = params["n_genes"]
    gene_method = params["gene_method"]
    print(f"  Params: {params}")

    all_metrics: List[dict] = []
    all_preds, all_true, all_probs = [], [], []
    roc_data_folds: List[dict] = []
    last: dict = {}

    for iteration in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED + iteration)

        for _, (train_idx, val_idx) in enumerate(skf.split(X_mri, y_enc)):
            X_mri_tr, X_mri_va   = X_mri[train_idx],  X_mri[val_idx]
            X_gene_tr, X_gene_va = X_gene[train_idx], X_gene[val_idx]
            y_tr, y_va           = y_enc[train_idx],  y_enc[val_idx]

            # Scale
            sc_mri  = StandardScaler()
            X_mri_tr_s  = sc_mri.fit_transform(X_mri_tr)
            X_mri_va_s  = sc_mri.transform(X_mri_va)
            sc_gene = StandardScaler()
            X_gene_tr_s = sc_gene.fit_transform(X_gene_tr)
            X_gene_va_s = sc_gene.transform(X_gene_va)

            # Gene selection (from training data only)
            X_g_tr, X_g_va, g_idx = gene_selection(
                X_gene_tr_s, y_tr, X_gene_va_s,
                n_select=k_gene, method=gene_method,
            )

            # Branch C — DL model
            trainer = Trainer(
                X_mri_tr_s.shape[1], X_g_tr.shape[1], n_classes,
                class_weights=class_weights,
                lr=params["lr"], weight_decay=params["weight_decay"],
                dropout=params["dropout"], hidden=params["hidden"],
            )
            trainer.fit(
                X_mri_tr_s, X_g_tr, y_tr,
                X_mri_va_s, X_g_va, y_va,
                epochs=100, batch_size=params["batch_size"], patience=25,
            )

            # Branch A — Linear SVM on concatenated features
            X_tr_cat = np.hstack([X_mri_tr_s, X_g_tr])
            X_va_cat = np.hstack([X_mri_va_s, X_g_va])
            svm_lin  = SVC(
                kernel="linear", probability=True, C=params["svm_C"],
                class_weight="balanced", random_state=SEED,
            )
            svm_lin.fit(X_tr_cat, y_tr)
            svm_proba = svm_lin.predict_proba(X_va_cat)

            # Branch B — RBF SVM on DL embeddings
            emb_tr = trainer.get_embeddings_np(X_mri_tr_s, X_g_tr)
            emb_va = trainer.get_embeddings_np(X_mri_va_s, X_g_va)
            svm_emb = SVC(
                kernel="rbf", probability=True, C=1.0, gamma="scale",
                class_weight="balanced", random_state=SEED,
            )
            svm_emb.fit(emb_tr, y_tr)
            svm_emb_proba = svm_emb.predict_proba(emb_va)

            # Weighted fusion
            dl_proba = trainer.predict_proba(X_mri_va_s, X_g_va)
            dw       = params["dl_weight"]
            w_rest   = 1.0 - dw
            y_proba  = (
                w_rest * 0.6 * svm_proba
                + w_rest * 0.4 * svm_emb_proba
                + dw * dl_proba
            )
            y_pred = np.argmax(y_proba, axis=1)

            metrics = compute_all_metrics(y_va, y_pred, y_proba, pos_label=pos_label)
            all_metrics.append(metrics)

            try:
                fpr_fold, tpr_fold, _ = roc_curve(
                    y_va == pos_label, y_proba[:, pos_label]
                )
                roc_data_folds.append({"fpr": fpr_fold, "tpr": tpr_fold})
            except Exception:
                pass

            all_preds.extend(y_pred)
            all_true.extend(y_va)
            all_probs.append(y_proba)

            last.update({
                "trainer": trainer, "svm": svm_lin,
                "sc_mri": sc_mri, "sc_gene": sc_gene,
                "gene_idx": g_idx,
                "X_mri_tr": X_mri_tr_s, "X_gene_tr": X_g_tr, "y_tr": y_tr,
            })

        if (iteration + 1) % 2 == 0:
            recent = all_metrics[-n_folds:]
            print(
                f"  Iter {iteration+1}/{n_repeats}: "
                f"BA={np.mean([m['BA'] for m in recent]):.4f}, "
                f"AUC={np.mean([m['AUC'] for m in recent]):.4f}"
            )

    # Aggregate
    from sklearn.metrics import confusion_matrix as _cm
    metric_names = ["Acc", "Sens", "Spec", "Prec", "F1", "Gm", "AUC", "MCC", "BA"]
    result_summary: dict = {}
    print(f"\n{'='*70}")
    print(f"RESULTS — {task_name} ({n_repeats}×{n_folds} = {n_repeats*n_folds} evaluations)")
    print(f"{'='*70}")
    for m in metric_names:
        vals = [r[m] for r in all_metrics]
        result_summary[m]             = float(np.mean(vals))
        result_summary[f"{m}_std"]    = float(np.std(vals))
        print(f"  {m:<8} {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    cm = _cm(all_true, all_preds)
    print(f"\n  Confusion Matrix (All {n_repeats*n_folds} Folds — Aggregated):\n{cm}")

    return {
        "task_name": task_name,
        "result_summary": result_summary,
        "all_metrics": all_metrics,
        "confusion_matrix": cm,
        "label_encoder": le,
        "all_true": all_true,
        "all_preds": all_preds,
        "all_probs": all_probs,
        "pos_label": pos_label,
        "roc_data_folds": roc_data_folds,
        **{f"last_{k}": v for k, v in last.items()},
    }
