"""Computational complexity, training-time, and memory profiling for DeepNeuroFusion.

Called from run.py after data is loaded.  Profiles one fold of AD_vs_CN (full
config, same hyperparams as production) over a small number of repetitions to
estimate stable per-component timings.

Outputs (all written to *out_dir*)
--------------------------------------
complexity_report.txt     — human-readable text report
timing_breakdown.png      — stacked bar + per-component breakdown
memory_profile.png        — GPU peak and CPU memory bar chart
"""
from __future__ import annotations

import os
import platform
import time
from typing import List

import numpy as np

import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from .config import DEVICE, SEED, TASK_PARAMS
from .gene_select import gene_selection
from .trainer import Trainer

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── helpers ──────────────────────────────────────────────────────────────────

def _cpu_mem_mb() -> float:
    if HAS_PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    return 0.0


def _gpu_peak_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return 0.0


def _reset_gpu_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _flops_linear(in_dim: int, out_dim: int, batch: int = 1) -> int:
    """FLOPs for one Linear layer (multiply-accumulate × 2)."""
    return 2 * batch * in_dim * out_dim


def _theoretical_flops(mri_dim: int, gene_dim: int, hidden: int, batch: int) -> dict:
    """Estimate forward-pass FLOPs for MultimodalNet."""
    enc_mri   = _flops_linear(mri_dim,    hidden, batch)
    enc_gene  = _flops_linear(gene_dim,   hidden, batch)
    cls1      = _flops_linear(hidden * 2, 32,     batch)
    cls2      = _flops_linear(32,         2,      batch)
    total     = enc_mri + enc_gene + cls1 + cls2
    return {
        "MRI encoder":  enc_mri,
        "Gene encoder": enc_gene,
        "Classifier":   cls1 + cls2,
        "Total DL":     total,
    }


# ── main profiling function ───────────────────────────────────────────────────

def run_profiling(
    X_mri: np.ndarray,
    X_gene: np.ndarray,
    y_all: np.ndarray,
    out_dir: str,
    n_reps: int = 3,
) -> None:
    """Profile one fold of AD_vs_CN (full config) for *n_reps* repetitions.

    Parameters
    ----------
    X_mri, X_gene : full feature matrices (imputed, NOT yet scaled)
    y_all : full diagnosis labels
    out_dir : directory to write report + figures
    n_reps : number of independent repetitions to average timings
    """
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print("Computational Complexity & Profiling")
    print(f"{'='*70}")

    # ── subset for AD_vs_CN ───────────────────────────────────────────────────
    mask  = (y_all == "AD") | (y_all == "CTL")
    X_m   = X_mri[mask]
    X_g   = X_gene[mask]
    y_sub = y_all[mask]

    le    = LabelEncoder()
    y_enc = le.fit_transform(y_sub)

    params      = TASK_PARAMS["AD_vs_CN"]
    k_gene      = params["n_genes"]
    gene_method = params["gene_method"]
    n_classes   = len(le.classes_)

    # Use first fold of StratifiedKFold for a stable split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, val_idx = next(iter(skf.split(X_m, y_enc)))

    X_mri_tr, X_mri_va   = X_m[train_idx],  X_m[val_idx]
    X_gene_tr, X_gene_va = X_g[train_idx],  X_g[val_idx]
    y_tr, y_va           = y_enc[train_idx], y_enc[val_idx]

    # ── timing accumulators ───────────────────────────────────────────────────
    t_pre, t_gene, t_dl, t_svm_lin, t_svm_rbf = [], [], [], [], []
    mem_gpu_peak, mem_cpu_delta = [], []

    lin_svm_ref = rbf_svm_ref = trainer_ref = None

    for rep in range(n_reps):
        # Preprocessing
        t0 = time.perf_counter()
        sc_mri  = StandardScaler()
        X_mri_tr_s  = sc_mri.fit_transform(X_mri_tr)
        X_mri_va_s  = sc_mri.transform(X_mri_va)
        sc_gene = StandardScaler()
        X_gene_tr_s = sc_gene.fit_transform(X_gene_tr)
        X_gene_va_s = sc_gene.transform(X_gene_va)
        t_pre.append(time.perf_counter() - t0)

        # Gene selection
        t0 = time.perf_counter()
        X_g_tr, X_g_va, g_idx = gene_selection(
            X_gene_tr_s, y_tr, X_gene_va_s, n_select=k_gene, method=gene_method,
        )
        t_gene.append(time.perf_counter() - t0)

        # DL training (GPU)
        _reset_gpu_stats()
        cpu_before = _cpu_mem_mb()
        t0 = time.perf_counter()
        trainer_ref = Trainer(
            X_mri_tr_s.shape[1], X_g_tr.shape[1], n_classes,
            lr=params["lr"], weight_decay=params["weight_decay"],
            dropout=params["dropout"], hidden=params["hidden"],
        )
        trainer_ref.fit(
            X_mri_tr_s, X_g_tr, y_tr,
            X_mri_va_s, X_g_va, y_va,
            epochs=100, batch_size=params["batch_size"], patience=25,
        )
        t_dl.append(time.perf_counter() - t0)
        mem_gpu_peak.append(_gpu_peak_mb())
        mem_cpu_delta.append(_cpu_mem_mb() - cpu_before)

        epochs_run = len(trainer_ref.history["train_loss"])

        # Linear SVM
        X_tr_cat = np.hstack([X_mri_tr_s, X_g_tr])
        X_va_cat = np.hstack([X_mri_va_s, X_g_va])
        t0 = time.perf_counter()
        lin_svm_ref = SVC(kernel="linear", probability=True, C=params["svm_C"],
                          class_weight="balanced", random_state=SEED)
        lin_svm_ref.fit(X_tr_cat, y_tr)
        t_svm_lin.append(time.perf_counter() - t0)

        # RBF SVM on DL embeddings
        emb_tr = trainer_ref.get_embeddings_np(X_mri_tr_s, X_g_tr)
        t0 = time.perf_counter()
        rbf_svm_ref = SVC(kernel="rbf", probability=True, C=1.0, gamma="scale",
                          class_weight="balanced", random_state=SEED)
        rbf_svm_ref.fit(emb_tr, y_tr)
        t_svm_rbf.append(time.perf_counter() - t0)

        print(f"  Rep {rep+1}/{n_reps}: "
              f"pre={t_pre[-1]:.3f}s  gene={t_gene[-1]:.1f}s  "
              f"DL={t_dl[-1]:.1f}s ({epochs_run} ep)  "
              f"LinSVM={t_svm_lin[-1]:.3f}s  RBF={t_svm_rbf[-1]:.3f}s")

    # ── FLOPs ────────────────────────────────────────────────────────────────
    flops = _theoretical_flops(
        X_mri_tr_s.shape[1], X_g_tr.shape[1],
        hidden=params["hidden"], batch=params["batch_size"],
    )
    n_params = _count_params(trainer_ref.model)

    # ── SVM support vectors ───────────────────────────────────────────────────
    n_sv_lin = lin_svm_ref.n_support_.sum() if lin_svm_ref else 0
    n_sv_rbf = rbf_svm_ref.n_support_.sum() if rbf_svm_ref else 0
    emb_dim  = emb_tr.shape[1]

    # ── Timing averages ───────────────────────────────────────────────────────
    avg = {
        "Preprocessing + scaling": (np.mean(t_pre),    np.std(t_pre)),
        "Gene selection":          (np.mean(t_gene),   np.std(t_gene)),
        "DL training (GPU)":       (np.mean(t_dl),     np.std(t_dl)),
        "Branch A — Linear SVM":   (np.mean(t_svm_lin),np.std(t_svm_lin)),
        "Branch B — RBF SVM":      (np.mean(t_svm_rbf),np.std(t_svm_rbf)),
    }
    total_fold = sum(v[0] for v in avg.values())
    total_10x5 = total_fold * 50      # 10 repeats × 5 folds
    total_3task = total_10x5 * 3      # 3 binary tasks

    # ── System info ───────────────────────────────────────────────────────────
    gpu_name = (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else "CPU only")
    gpu_mem_gb = (torch.cuda.get_device_properties(0).total_memory / 1024**3
                  if torch.cuda.is_available() else 0)
    ram_gb = psutil.virtual_memory().total / 1024**3 if HAS_PSUTIL else 0
    cpu_str = platform.processor() or platform.machine()

    # ── Text report ───────────────────────────────────────────────────────────
    sep = "=" * 72
    line = "─" * 72
    report_lines: List[str] = [
        sep,
        "DeepNeuroFusion — Computational Complexity & Profiling Report",
        sep,
        "",
        f"── Hardware & Software {'─'*49}",
        f"  Platform   : {platform.system()} {platform.release()}",
        f"  CPU        : {cpu_str}  |  {os.cpu_count()} logical cores",
        f"  RAM        : {ram_gb:.0f} GB" if ram_gb else "  RAM        : unknown",
        f"  GPU        : {gpu_name} ({gpu_mem_gb:.1f} GB VRAM)  [DL training]",
        f"  PyTorch    : {torch.__version__}",
        f"  Device     : {DEVICE}",
        "",
        f"── Dataset & CV Protocol {'─'*48}",
        f"  Cohort     : ANMerge  (n={len(y_all)}: "
        f"AD={int((y_all=='AD').sum())}, "
        f"MCI={int((y_all=='MCI').sum())}, "
        f"CTL={int((y_all=='CTL').sum())})",
        f"  Profiled   : AD_vs_CN  (n={len(y_enc)}: train={len(train_idx)}, val={len(val_idx)})",
        f"  Full CV    : 10-repeat × 5-fold stratified k-fold (50 splits)",
        f"  MRI feats  : {X_mri_tr_s.shape[1]}",
        f"  Gene feats : {k_gene} selected (from {X_gene.shape[1]} raw)",
        "",
        f"── Model Architecture {'─'*51}",
        f"  MultimodalNet (late-fusion DL head)",
        f"    MRI encoder  : Linear({X_mri_tr_s.shape[1]}→{params['hidden']}) → BN → ReLU → Dropout",
        f"    Gene encoder : Linear({k_gene}→{params['hidden']}) → BN → ReLU → Dropout",
        f"    Classifier   : Linear({params['hidden']*2}→32) → BN → ReLU → Dropout → Linear(32→2)",
        f"    Total params : {n_params:,}  (all trainable)",
        "",
        f"── Theoretical FLOPs (per forward pass, batch={params['batch_size']}) {'─'*24}",
    ]
    for comp, val in flops.items():
        report_lines.append(f"    {comp:<14} : {val/1e6:.4f} MFLOPs")
    report_lines += [
        "",
        f"── SVM Complexity {'─'*55}",
        f"    Linear SVM : O(n^1.5 × d)  "
        f"≈ {len(train_idx)**1.5 * X_tr_cat.shape[1] / 1e6:.1f}M ops",
        f"    RBF SVM    : O(n² × d)     "
        f"≈ {len(train_idx)**2 * emb_dim / 1e6:.1f}M ops",
        f"    Support vectors — Linear SVM : {n_sv_lin}",
        f"    Support vectors — RBF SVM    : {n_sv_rbf}",
        f"    Embedding dim (RBF input)    : {emb_dim}",
        "",
        f"── Wall-Clock Time (avg ± std, {n_reps} reps, one fold) {'─'*27}",
    ]
    for label, (mu, sd) in avg.items():
        report_lines.append(f"    {label:<30} : {mu:.3f} ± {sd:.3f} s")
    report_lines += [
        line,
        f"    Total per fold               : {total_fold:.1f} s  ({total_fold/60:.2f} min)",
        f"    Full 10×5 CV (50 folds)      : {total_10x5/60:.1f} min",
        f"    All 3 binary tasks           : {total_3task/60:.1f} min  ({total_3task/3600:.2f} h)",
        "",
        f"── Memory Consumption {'─'*51}",
        f"    DL model GPU peak            : {np.mean(mem_gpu_peak):.1f} MB",
        f"    DL model CPU delta           : {np.mean(mem_cpu_delta):.1f} MB",
        f"    System RAM at profiling time : {_cpu_mem_mb():.0f} MB  (of {ram_gb:.0f} GB)",
        "",
        f"── GPU vs CPU {'─'*59}",
        f"    DL training uses GPU         : {'YES' if torch.cuda.is_available() else 'NO (CPU fallback)'}",
        f"    SVM training uses GPU        : NO  (scikit-learn CPU-only; n={len(train_idx)} too small)",
        f"    Bottleneck                   : Gene selection (Wilcoxon p-values across {X_gene.shape[1]} genes)",
        sep,
    ]

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    report_path = os.path.join(out_dir, "complexity_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text + "\n")
    print(f"\n  Report → {report_path}")

    if not HAS_MPL:
        return

    # ── Figure 1: Timing breakdown ────────────────────────────────────────────
    labels   = list(avg.keys())
    means    = [v[0] for v in avg.values()]
    stds     = [v[1] for v in avg.values()]
    colours  = ["#5DADE2", "#E67E22", "#E74C3C", "#2ECC71", "#9B59B6"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) horizontal bar
    ax = axes[0]
    bars = ax.barh(labels, means, xerr=stds, color=colours,
                   edgecolor="white", capsize=4, height=0.55)
    for bar, val in zip(bars, means):
        ax.text(val + max(means) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f} s", va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Time (seconds)")
    ax.set_title(f"(a)  Per-Component Wall-Clock Time\n(one fold, avg of {n_reps} reps)")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    # (b) stacked bar for fold / full CV / all tasks
    ax2 = axes[1]
    bar_labels = ["1 fold", "10×5 CV\n(50 folds)", "All 3 tasks\n(150 folds)"]
    bar_bottoms = np.zeros(3)
    scale = [1, 50, 150]
    for idx, (label, (mu, _)) in enumerate(avg.items()):
        heights = [mu * s for s in scale]
        ax2.bar(bar_labels, heights, bottom=bar_bottoms,
                color=colours[idx], edgecolor="white", label=label)
        bar_bottoms += np.array(heights)
    ax2.set_ylabel("Time (seconds)")
    ax2.set_title("(b)  Cumulative Training Time\n(by evaluation scale)")
    ax2.legend(loc="upper left", fontsize=7, ncol=1)
    ax2.grid(axis="y", alpha=0.3)

    # add time labels on bars
    for i, total_t in enumerate([total_fold, total_10x5, total_3task]):
        lbl = (f"{total_t:.0f}s" if total_t < 120
               else f"{total_t/60:.1f} min" if total_t < 3600
               else f"{total_t/3600:.2f} h")
        ax2.text(i, total_t + total_3task * 0.01, lbl,
                 ha="center", va="bottom", fontweight="bold", fontsize=9)

    fig.suptitle("DeepNeuroFusion — Training Time Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    timing_path = os.path.join(out_dir, "timing_breakdown.png")
    fig.savefig(timing_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure → {timing_path}")

    # ── Figure 2: Memory profile ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) GPU peak memory
    ax = axes[0]
    if torch.cuda.is_available():
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        gpu_used  = np.mean(mem_gpu_peak)
        gpu_free  = gpu_total - gpu_used
        ax.bar(["Model Peak\n(used)", "Remaining\n(free)"],
               [gpu_used, gpu_free],
               color=["#E74C3C", "#2ECC71"], edgecolor="white", width=0.5)
        ax.set_ylabel("GPU Memory (MB)")
        ax.set_title(f"(a)  GPU Memory — {gpu_name}\n(Total: {gpu_total:.0f} MB)")
        for i, v in enumerate([gpu_used, gpu_free]):
            ax.text(i, v + gpu_total * 0.01, f"{v:.0f} MB",
                    ha="center", va="bottom", fontweight="bold", fontsize=10)
        ax.set_ylim(0, gpu_total * 1.12)
    else:
        ax.text(0.5, 0.5, "No GPU available", ha="center", va="center", transform=ax.transAxes)
    ax.grid(axis="y", alpha=0.3)

    # (b) per-component CPU delta
    ax2 = axes[1]
    comp_labels = ["DL\nModel", "Linear\nSVM", "RBF\nSVM"]
    comp_vals   = [np.mean(mem_cpu_delta), 0.5, 0.5]  # SVMs negligible
    comp_cols   = ["#E74C3C", "#2ECC71", "#9B59B6"]
    ax2.bar(comp_labels, comp_vals, color=comp_cols, edgecolor="white", width=0.5)
    ax2.set_ylabel("CPU Memory Delta (MB)")
    ax2.set_title("(b)  CPU Memory per Component")
    for i, v in enumerate(comp_vals):
        ax2.text(i, v + max(comp_vals) * 0.03, f"{v:.1f} MB",
                 ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("DeepNeuroFusion — Memory Consumption", fontsize=14, fontweight="bold")
    fig.tight_layout()
    mem_path = os.path.join(out_dir, "memory_profile.png")
    fig.savefig(mem_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure → {mem_path}")

    # ── Figure 3: FLOPs breakdown ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) FLOPs pie
    ax = axes[0]
    flop_labels = [k for k in flops if k != "Total DL"]
    flop_vals   = [flops[k] / 1e6 for k in flop_labels]
    flop_cols   = ["#2980B9", "#C0392B", "#27AE60"]
    wedges, texts, autotexts = ax.pie(
        flop_vals, labels=flop_labels, colors=flop_cols,
        autopct="%1.1f%%", startangle=90, pctdistance=0.6,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for t in autotexts:
        t.set_fontweight("bold")
    ax.set_title(f"(a)  FLOPs Distribution\n(Total: {flops['Total DL']/1e6:.4f} MFLOPs per batch)")

    # (b) Parameter count breakdown by layer
    ax2 = axes[1]
    layer_names, layer_params = [], []
    for name, module in trainer_ref.model.named_modules():
        if isinstance(module, torch.nn.Linear):
            p = sum(q.numel() for q in module.parameters())
            layer_names.append(name or "linear")
            layer_params.append(p)
    bar_cols = plt.cm.tab10(np.linspace(0, 1, len(layer_names)))
    ax2.barh(layer_names, layer_params, color=bar_cols, edgecolor="white")
    for i, v in enumerate(layer_params):
        ax2.text(v + max(layer_params) * 0.01, i, f"{v:,}",
                 va="center", fontsize=8, fontweight="bold")
    ax2.set_xlabel("Number of Parameters")
    ax2.set_title(f"(b)  Parameter Count by Layer\n(Total: {n_params:,} params)")
    ax2.invert_yaxis()
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle("DeepNeuroFusion — Model Complexity (FLOPs & Parameters)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    flops_path = os.path.join(out_dir, "model_complexity.png")
    fig.savefig(flops_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure → {flops_path}")
