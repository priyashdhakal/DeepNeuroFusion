from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler


from .config import PALETTE, SEED

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: shap not installed — pip install shap")

# ── Publication style ─────────────────────────────────────────────────────────
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


# ── EDA ───────────────────────────────────────────────────────────────────────

def plot_data_overview(
    X_mri: np.ndarray,
    X_gene: np.ndarray,
    y: np.ndarray,
    mri_names: List[str],
    gene_names: List[str],
    out_dir: str,
) -> None:
    """Generate comprehensive EDA figures and save to *out_dir*."""
    os.makedirs(out_dir, exist_ok=True)

    # Class distribution
    fig, ax = plt.subplots(figsize=(5, 4))
    counts  = pd.Series(y).value_counts()
    colours = [PALETTE.get(c.lower(), "#888") for c in counts.index]
    bars    = ax.bar(counts.index, counts.values, color=colours, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1, str(val),
                ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylabel("Number of Subjects")
    ax.set_title("Class Distribution — ANMerge Cohort")
    ax.set_ylim(0, counts.max() * 1.15)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "class_distribution.png"))
    plt.close(fig)

    # MRI correlation heat-map (top-30 by variance)
    top30_idx   = np.argsort(np.var(X_mri, axis=0))[-30:]
    top30_names = [mri_names[i] if i < len(mri_names) else f"MRI_{i}" for i in top30_idx]
    df_corr     = pd.DataFrame(X_mri[:, top30_idx], columns=top30_names).corr()
    fig, ax = plt.subplots(figsize=(10, 9))
    mask = np.triu(np.ones_like(df_corr, dtype=bool))
    sns.heatmap(df_corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.75, "label": "Pearson r"})
    ax.set_title("MRI Feature Correlation (Top-30 by Variance)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "mri_correlation_heatmap.png"))
    plt.close(fig)

    # Gene-expression variance distribution
    gene_var = np.var(X_gene, axis=0)
    fig, ax  = plt.subplots(figsize=(7, 4))
    ax.hist(gene_var, bins=80, color=PALETTE["gene"], alpha=0.75, edgecolor="white")
    ax.axvline(np.median(gene_var), color="black", ls="--", lw=1.2,
               label=f"Median = {np.median(gene_var):.4f}")
    ax.set_xlabel("Variance across subjects")
    ax.set_ylabel("Number of genes")
    ax.set_title("Gene-Expression Variance Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "gene_variance_distribution.png"))
    plt.close(fig)

    # PCA of combined MRI + Gene features
    scaler    = StandardScaler()
    X_combined = np.hstack([scaler.fit_transform(X_mri), scaler.fit_transform(X_gene)])
    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X_combined)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for cls in np.unique(y):
        m = y == cls
        ax.scatter(X_pca[m, 0], X_pca[m, 1], label=cls, alpha=0.65, s=40,
                   edgecolors="white", linewidth=0.4, color=PALETTE.get(cls.lower(), "#888"))
    ax.set_xlabel(f"PC-1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC-2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA — Combined MRI + Gene Features")
    ax.legend(title="Diagnosis", frameon=True, fancybox=True, shadow=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pca_combined.png"))
    plt.close(fig)

    # PCA per modality (MRI-only vs Gene-only side by side)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax_i, (X_mod, title_mod) in enumerate([
        (scaler.fit_transform(X_mri),  "MRI Only"),
        (scaler.fit_transform(X_gene), "Gene Only"),
    ]):
        pca_mod   = PCA(n_components=2, random_state=SEED)
        X_mod_pca = pca_mod.fit_transform(X_mod)
        for cls in np.unique(y):
            m = y == cls
            axes[ax_i].scatter(X_mod_pca[m, 0], X_mod_pca[m, 1], label=cls,
                               alpha=0.6, s=35, edgecolors="white", linewidth=0.4,
                               color=PALETTE.get(cls.lower(), "#888"))
        axes[ax_i].set_xlabel(f"PC-1 ({pca_mod.explained_variance_ratio_[0]*100:.1f}%)")
        axes[ax_i].set_ylabel(f"PC-2 ({pca_mod.explained_variance_ratio_[1]*100:.1f}%)")
        axes[ax_i].set_title(f"PCA — {title_mod}")
        axes[ax_i].legend(title="Diagnosis", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pca_modality_comparison.png"))
    plt.close(fig)

    # Top-10 MRI features by diagnosis (box-plots)
    var_rank = np.argsort(np.var(X_mri, axis=0))[-10:][::-1]
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.ravel()
    for i, idx in enumerate(var_rank):
        feat_name   = mri_names[idx] if idx < len(mri_names) else f"MRI_{idx}"
        df_box      = pd.DataFrame({"value": X_mri[:, idx], "Diagnosis": y})
        palette_map = {cls: PALETTE.get(cls.lower(), "#888") for cls in np.unique(y)}
        sns.boxplot(data=df_box, x="Diagnosis", y="value", ax=axes[i],
                    palette=palette_map, width=0.6, fliersize=2)
        axes[i].set_title(feat_name, fontsize=9)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
    fig.suptitle("Top-10 MRI Features by Diagnosis", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out_dir, "mri_feature_boxplots.png"))
    plt.close(fig)

    print(f"  EDA figures saved to {out_dir}/")


# ── Result figures ────────────────────────────────────────────────────────────

def plot_performance(all_results: dict, out_dir: str) -> None:
    """Bar chart of DeepNeuroFusion results across all three tasks."""
    metrics_to_plot = ["Acc", "Sens", "Spec", "AUC", "BA", "MCC", "F1"]
    tasks = [t for t in ["AD_vs_CN", "AD_vs_MCI", "MCI_vs_CN"] if t in all_results]

    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 5))
    if len(tasks) == 1:
        axes = [axes]

    colours = [PALETTE["mri"], PALETTE["ours"], PALETTE["accent"]]

    for ax, task, colour in zip(axes, tasks, colours):
        ours = all_results[task]["result_summary"]
        x    = np.arange(len(metrics_to_plot))
        vals = [ours.get(m, 0) for m in metrics_to_plot]
        errs = [ours.get(f"{m}_std", 0) for m in metrics_to_plot]

        bars = ax.bar(x, vals, yerr=errs, color=colour, alpha=0.85,
                      edgecolor="white", capsize=4)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(metrics_to_plot, rotation=45)
        ax.set_ylabel("Score")
        ax.set_title(task.replace("_", " "))
        ax.set_ylim(0, 1.12)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("DeepNeuroFusion — Classification Performance",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "performance.png"))
    plt.close(fig)


def plot_confusion_matrices(all_results: dict, out_dir: str) -> None:
    """Annotated confusion matrices for each task."""
    tasks = [t for t in ["AD_vs_CN", "AD_vs_MCI", "MCI_vs_CN"] if t in all_results]

    fig, axes = plt.subplots(1, len(tasks), figsize=(5.5 * len(tasks), 4.5))
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        res    = all_results[task]
        cm     = res["confusion_matrix"]
        le     = res["label_encoder"]
        labels = [c.replace("CTL", "CN") for c in le.classes_]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=labels, yticklabels=labels,
                    linewidths=0.8, linecolor="white",
                    annot_kws={"fontsize": 14, "fontweight": "bold"})
        ax.set_title(task.replace("_", " ").replace("CN", "CN"))
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

    fig.suptitle("Confusion Matrices (Aggregated — All Folds)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrices.png"))
    plt.close(fig)


def plot_roc_curves(all_results: dict, out_dir: str) -> None:
    """ROC curves with mean ± 1 std band across all folds."""
    tasks = [t for t in ["AD_vs_CN", "AD_vs_MCI", "MCI_vs_CN"] if t in all_results]

    fig, axes = plt.subplots(1, len(tasks), figsize=(5.5 * len(tasks), 5))
    if len(tasks) == 1:
        axes = [axes]

    colours = [PALETTE["mri"], PALETTE["ours"], PALETTE["accent"]]

    for ax, task, colour in zip(axes, tasks, colours):
        res       = all_results[task]
        roc_folds = res.get("roc_data_folds", [])
        auc_val   = res["result_summary"]["AUC"]
        auc_std   = res["result_summary"].get("AUC_std", 0)

        if roc_folds:
            mean_fpr = np.linspace(0, 1, 200)
            tprs     = [np.interp(mean_fpr, rd["fpr"], rd["tpr"]) for rd in roc_folds]
            mean_tpr = np.mean(tprs, axis=0)
            std_tpr  = np.std(tprs, axis=0)
            mean_tpr[0], mean_tpr[-1] = 0.0, 1.0

            ax.plot(mean_fpr, mean_tpr, color=colour, lw=2,
                    label=f"Mean AUC = {auc_val:.3f} ± {auc_std:.3f}")
            ax.fill_between(mean_fpr,
                            np.clip(mean_tpr - std_tpr, 0, 1),
                            np.clip(mean_tpr + std_tpr, 0, 1),
                            color=colour, alpha=0.15, label="± 1 std")
        else:
            y_true  = np.array(res["all_true"])
            y_probs = np.vstack(res["all_probs"])
            pos     = res["pos_label"]
            fpr, tpr, _ = roc_curve(y_true == pos, y_probs[:, pos])
            ax.plot(fpr, tpr, color=colour, lw=2, label=f"AUC = {auc_val:.3f}")

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC — {task.replace('_', ' ')}")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.25)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    fig.suptitle("Receiver Operating Characteristic Curves",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "roc_curves.png"))
    plt.close(fig)


def plot_metric_distributions(all_results: dict, out_dir: str) -> None:
    """Violin + strip plots of metric distributions across all folds."""
    tasks   = [t for t in ["AD_vs_CN", "AD_vs_MCI", "MCI_vs_CN"] if t in all_results]
    metrics = ["BA", "AUC", "F1", "MCC"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 5))

    for ax, metric in zip(axes, metrics):
        rows = []
        for task in tasks:
            for m in all_results[task]["all_metrics"]:
                rows.append({"Task": task.replace("_", " "), metric: m[metric]})
        df = pd.DataFrame(rows)
        task_palette = {t.replace("_", " "): c for t, c in
                        zip(tasks, [PALETTE["mri"], PALETTE["ours"], PALETTE["accent"]])}
        sns.violinplot(data=df, x="Task", y=metric, ax=ax,
                       palette=task_palette, inner=None, alpha=0.35, cut=0)
        sns.stripplot(data=df, x="Task", y=metric, ax=ax,
                      palette=task_palette, size=2, alpha=0.5, jitter=True)
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Metric Distributions (50 Evaluations)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "metric_distributions.png"))
    plt.close(fig)


def plot_training_history(trainer, task_name: str, out_dir: str) -> None:
    """Loss and accuracy curves from the last training fold."""
    h = trainer.history
    if not h["train_loss"]:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(h["train_loss"], label="Train", color=PALETTE["mri"], lw=1.5)
    ax1.plot(h["val_loss"],   label="Val",   color=PALETTE["ours"], lw=1.5)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title(f"{task_name} — Loss"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(h["train_acc"], label="Train", color=PALETTE["mri"], lw=1.5)
    ax2.plot(h["val_acc"],   label="Val",   color=PALETTE["ours"], lw=1.5)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{task_name} — Accuracy"); ax2.legend(); ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"training_history_{task_name}.png"))
    plt.close(fig)


# ── SHAP ─────────────────────────────────────────────────────────────────────

def run_shap_analysis(
    result: dict,
    mri_feat_names: List[str],
    gene_feat_names: List[str],
    out_dir: str,
) -> Optional[dict]:
    """SHAP analysis on the linear SVM from the last fold.

    Produces per-task:
      - shap_<task>_bar.png        — top-20 overall + top-15 per modality
      - shap_<task>_beeswarm.png   — SHAP summary beeswarm
      - shap_<task>_modality.png   — modality contribution pie chart
    """
    if not HAS_SHAP:
        print("  SHAP not available — skipping.")
        return None

    task_name = result["task_name"]
    svm       = result["last_svm"]
    gene_idx  = result["last_gene_idx"]
    X_mri_tr  = result["last_X_mri_tr"]
    X_gene_tr = result["last_X_gene_tr"]

    mri_names = mri_feat_names if mri_feat_names else [f"MRI_{i}" for i in range(X_mri_tr.shape[1])]
    sel_gene_names = (
        [gene_feat_names[i] for i in gene_idx]
        if gene_feat_names is not None and gene_idx is not None
        else [f"Gene_{i}" for i in range(X_gene_tr.shape[1])]
    )
    all_names = list(mri_names) + list(sel_gene_names)
    n_mri = len(mri_names)

    X_train_cat = np.hstack([X_mri_tr, X_gene_tr])

    print(f"\n  Running SHAP for {task_name}…")
    print(f"    Features: {len(all_names)} (MRI: {n_mri}, Gene: {len(sel_gene_names)})")

    explainer   = shap.LinearExplainer(svm, X_train_cat, feature_names=all_names)
    shap_values = explainer.shap_values(X_train_cat)

    shap_vals    = shap_values[1] if isinstance(shap_values, list) else shap_values
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)

    mri_shap  = mean_abs_shap[:n_mri]
    gene_shap = mean_abs_shap[n_mri:]

    total_mri  = np.sum(mri_shap)
    total_gene = np.sum(gene_shap)
    total      = total_mri + total_gene
    pct_mri    = total_mri  / total * 100
    pct_gene   = total_gene / total * 100
    print(f"    MRI contribution  : {pct_mri:.1f}%")
    print(f"    Gene contribution : {pct_gene:.1f}%")

    os.makedirs(out_dir, exist_ok=True)

    # Figure 1: Bar charts
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    top20   = np.argsort(mean_abs_shap)[-20:]
    colors_20 = [PALETTE["mri"] if i < n_mri else PALETTE["gene"] for i in top20]
    axes[0].barh(range(20), mean_abs_shap[top20], color=colors_20, edgecolor="white", linewidth=0.5)
    axes[0].set_yticks(range(20))
    axes[0].set_yticklabels([all_names[i] for i in top20], fontsize=8)
    axes[0].set_xlabel("Mean |SHAP value|")
    axes[0].set_title("(a)  Top-20 Features\n(Blue = MRI, Red = Gene)")

    top15_m = np.argsort(mri_shap)[-15:]
    axes[1].barh(range(15), mri_shap[top15_m], color=PALETTE["mri"], edgecolor="white", linewidth=0.5)
    axes[1].set_yticks(range(15))
    axes[1].set_yticklabels([mri_names[i] for i in top15_m], fontsize=8)
    axes[1].set_xlabel("Mean |SHAP value|")
    axes[1].set_title("(b)  Top-15 MRI Features")

    top15_g = np.argsort(gene_shap)[-15:]
    axes[2].barh(range(15), gene_shap[top15_g], color=PALETTE["gene"], edgecolor="white", linewidth=0.5)
    axes[2].set_yticks(range(15))
    axes[2].set_yticklabels([sel_gene_names[i] for i in top15_g], fontsize=8)
    axes[2].set_xlabel("Mean |SHAP value|")
    axes[2].set_title("(c)  Top-15 Gene Features")

    for ax in axes:
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle(f"SHAP Feature Importance — {task_name}", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"shap_{task_name}_bar.png"))
    plt.close(fig)

    # Figure 2: Beeswarm
    fig, ax = plt.subplots(figsize=(10, 9))
    shap.summary_plot(shap_vals, X_train_cat, feature_names=all_names,
                      max_display=25, show=False)
    plt.title(f"SHAP Beeswarm — {task_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"shap_{task_name}_beeswarm.png"))
    plt.close("all")

    # Figure 3: Modality pie
    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax.pie(
        [pct_mri, pct_gene], labels=["MRI", "Gene"],
        colors=[PALETTE["mri"], PALETTE["gene"]],
        autopct="%1.1f%%", startangle=90, pctdistance=0.55,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for t in autotexts:
        t.set_fontweight("bold"); t.set_fontsize(12)
    ax.set_title(f"Modality Contribution — {task_name}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"shap_{task_name}_modality.png"))
    plt.close(fig)

    print(f"    SHAP figures saved to {out_dir}/")

    return {
        "shap_values":    shap_vals,
        "mean_abs_shap":  mean_abs_shap,
        "mri_shap":       mri_shap,
        "gene_shap":      gene_shap,
        "feature_names":  all_names,
        "mri_names":      mri_names,
        "gene_names":     sel_gene_names,
        "pct_mri":        pct_mri,
        "pct_gene":       pct_gene,
    }


# ── CSV output ────────────────────────────────────────────────────────────────

def save_results_csv(all_results: dict, out_dir: str) -> None:
    """Save summary table and per-fold metrics to CSV."""
    os.makedirs(out_dir, exist_ok=True)

    # Summary table
    rows = []
    for task_name, res in all_results.items():
        r   = res["result_summary"]
        row = {"Task": task_name}
        for m in ["Acc", "Sens", "Spec", "Prec", "F1", "Gm", "AUC", "MCC", "BA"]:
            row[m]            = f"{r[m]:.4f}"
            row[f"{m}_std"]   = f"{r[f'{m}_std']:.4f}"
        rows.append(row)
    df_summary = pd.DataFrame(rows)
    summary_path = os.path.join(out_dir, "results_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"  Summary → {summary_path}")

    # Per-fold metrics
    for task_name, res in all_results.items():
        df_folds = pd.DataFrame(res["all_metrics"])
        df_folds.insert(0, "fold", range(1, len(df_folds) + 1))
        fold_path = os.path.join(out_dir, f"metrics_per_fold_{task_name}.csv")
        df_folds.to_csv(fold_path, index=False)


def save_shap_tables(shap_results: dict, out_dir: str) -> None:
    """Save SHAP feature-importance rankings to CSV for each task."""
    os.makedirs(out_dir, exist_ok=True)
    for task_name, sr in shap_results.items():
        if sr is None:
            continue
        rows = []
        for i, name in enumerate(sr["feature_names"]):
            modality = "MRI" if i < len(sr["mri_names"]) else "Gene"
            rows.append({
                "Feature": name,
                "Modality": modality,
                "Mean_Abs_SHAP": round(float(sr["mean_abs_shap"][i]), 6),
            })
        df = pd.DataFrame(rows).sort_values("Mean_Abs_SHAP", ascending=False)
        path = os.path.join(out_dir, f"shap_importance_{task_name}.csv")
        df.to_csv(path, index=False)
    print(f"  SHAP tables saved to {out_dir}/")
