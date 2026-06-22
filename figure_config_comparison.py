"""
DeepNeuroFusion — Configuration Comparison Figure

Two-panel publication figure:
  Panel A  — MCC (solid) and BA (dashed) performance trend ± 1 SD shading
             across 4 feature configurations for all three binary tasks,
             with per-point MCC value labels.
  Panel B  — Per-fold MCC distribution (box + jitter + mean diamond) for
             the full configuration (MRI + GE + Clin) across 50 folds.

Usage
-----
    cd DeepNeuroFusion_V1/
    python figure_config_comparison.py
    python figure_config_comparison.py --tables-dir path/to/tables \\
                                       --out-dir path/to/figures
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":         "serif",
    "font.size":           11,
    "axes.titlesize":      12,
    "axes.labelsize":      11,
    "xtick.labelsize":     10,
    "ytick.labelsize":     10,
    "legend.fontsize":     9.5,
    "legend.framealpha":   0.93,
    "legend.edgecolor":    "#cccccc",
    "legend.borderpad":    0.65,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.linewidth":      0.9,
    "axes.grid":           True,
    "grid.color":          "#e6e6e6",
    "grid.linewidth":      0.7,
    "xtick.direction":     "out",
    "ytick.direction":     "out",
    "xtick.major.size":    4,
    "ytick.major.size":    4,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.minor.size":    2,
    "ytick.minor.size":    2,
    "figure.dpi":          300,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.15,
})

# ── Constants ─────────────────────────────────────────────────────────────────
CONFIG_ORDER = [
    "Config1_MRI_only_no_clinical",
    "Config2_Gene_only_no_clinical",
    "Config3_MRI_Gene_no_clinical",
    "Config4_MRI_Gene_with_clinical",
]
CONFIG_LABELS = ["MRI only", "GE only", "MRI + GE", "MRI + GE\n+ Clin"]
FULL_CONFIG   = "Config4_MRI_Gene_with_clinical"

TASKS  = ["AD_vs_CN",  "AD_vs_MCI",  "MCI_vs_CN"]
LABELS = ["AD vs. CN", "AD vs. MCI", "MCI vs. CN"]

COLORS = {
    "AD_vs_CN":  "#2E86AB",
    "AD_vs_MCI": "#E07B39",
    "MCI_vs_CN": "#3BB273",
}
MEAN_COLOR = "#C0392B"
X = np.arange(len(CONFIG_ORDER))


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_summary(tables_dir: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(tables_dir, "table1_config_comparison.csv"))
    df["Config"] = df["Config"].str.strip()
    return df


def load_fold_mcc(tables_dir: str, task: str, config: str) -> np.ndarray:
    fname = f"metrics_per_fold_{task}_{config}.csv"
    return pd.read_csv(os.path.join(tables_dir, fname))["MCC"].to_numpy()


def extract_trend(df: pd.DataFrame, task: str, metric: str):
    means, stds = [], []
    for cfg in CONFIG_ORDER:
        row = df[(df["Task"] == task) & (df["Config"] == cfg)]
        if row.empty:
            means.append(np.nan); stds.append(0.0)
        else:
            means.append(float(row[metric].iloc[0]))
            stds.append(float(row[f"{metric}_std"].iloc[0]))
    return np.array(means), np.array(stds)


# ── Panel A — MCC + BA trend ──────────────────────────────────────────────────

def draw_panel_a(ax: plt.Axes, df: pd.DataFrame) -> None:

    for task, label in zip(TASKS, LABELS):
        c = COLORS[task]
        mcc_mu, mcc_sd = extract_trend(df, task, "MCC")
        ba_mu,  ba_sd  = extract_trend(df, task, "BA")

        # ── MCC — solid line, filled circle markers ───────────────────────
        ax.plot(X, mcc_mu, color=c, lw=2.5, ls="-",
                marker="o", markersize=7.5,
                markerfacecolor=c, markeredgecolor="white",
                markeredgewidth=1.3, zorder=4, label=label)
        ax.fill_between(X,
                        np.clip(mcc_mu - mcc_sd, 0, 1),
                        np.clip(mcc_mu + mcc_sd, 0, 1),
                        color=c, alpha=0.14, zorder=2)

        # MCC value labels at x=0 and x=3
        for xi, va_dir, dy in [(0, "top", -0.030), (3, "bottom", 0.025)]:
            if not np.isnan(mcc_mu[xi]):
                ax.text(xi, mcc_mu[xi] + dy, f"{mcc_mu[xi]:.3f}",
                        ha="center", va=va_dir, fontsize=8.2,
                        color=c, fontweight="bold", zorder=7)

        # ── BA — dashed line, hollow square markers ───────────────────────
        ax.plot(X, ba_mu, color=c, lw=1.9, ls="--",
                marker="s", markersize=6,
                markerfacecolor="white", markeredgecolor=c,
                markeredgewidth=1.7, zorder=3)
        ax.fill_between(X,
                        np.clip(ba_mu - ba_sd, 0, 1),
                        np.clip(ba_mu + ba_sd, 0, 1),
                        color=c, alpha=0.07, zorder=1)

    # ── Axes ─────────────────────────────────────────────────────────────────
    ax.set_xticks(X)
    ax.set_xticklabels(CONFIG_LABELS, fontsize=10.5)
    ax.set_ylabel("Performance Score", labelpad=9)
    ax.set_ylim(0.16, 1.07)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.set_xlim(-0.45, len(CONFIG_ORDER) - 0.55)

    # ── Legend: task colours (lower right) ───────────────────────────────────
    task_handles = [
        mpatches.Patch(facecolor=COLORS[t], label=l, edgecolor="none")
        for t, l in zip(TASKS, LABELS)
    ]
    # Legend: line styles (lower left)
    style_handles = [
        plt.Line2D([0], [0], color="#444", lw=2.3, ls="-",
                   marker="o", markersize=6,
                   markerfacecolor="#444", markeredgecolor="white",
                   markeredgewidth=1.2, label="MCC  (solid)"),
        plt.Line2D([0], [0], color="#444", lw=1.9, ls="--",
                   marker="s", markersize=5.5,
                   markerfacecolor="white", markeredgecolor="#444",
                   markeredgewidth=1.6, label="BA    (dashed)"),
    ]
    leg1 = ax.legend(handles=task_handles, loc="lower right",
                     title="Task", title_fontsize=9.5,
                     frameon=True, handlelength=1.3)
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, loc="lower left",
              frameon=True, handlelength=2.6)

    ax.set_title("A.   Performance across feature configurations",
                 loc="left", fontweight="bold", pad=10)


# ── Panel B — Per-fold MCC boxplot ────────────────────────────────────────────

def draw_panel_b(ax: plt.Axes, tables_dir: str) -> None:

    data      = [load_fold_mcc(tables_dir, task, FULL_CONFIG) for task in TASKS]
    n_folds   = len(data[0])
    positions = np.arange(len(TASKS))
    rng       = np.random.default_rng(42)

    # ── Boxplot ──────────────────────────────────────────────────────────────
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        notch=False,
        showfliers=False,
        medianprops=dict(color="#111111",  lw=2.6, solid_capstyle="round"),
        whiskerprops=dict(color="#444444", lw=1.6, ls="-"),
        capprops=dict(color="#444444",      lw=2.1),
        boxprops=dict(lw=1.5),
        zorder=2,
    )
    for patch, task in zip(bp["boxes"], TASKS):
        patch.set_facecolor(COLORS[task])
        patch.set_alpha(0.32)

    # ── Jittered individual fold points ──────────────────────────────────────
    for i, (vals, task) in enumerate(zip(data, TASKS)):
        jitter = rng.uniform(-0.17, 0.17, size=len(vals))
        ax.scatter(i + jitter, vals,
                   s=24, color=COLORS[task], alpha=0.52,
                   linewidths=0.3, edgecolors="white", zorder=3)

    # ── Red-diamond fold means ────────────────────────────────────────────────
    for i, vals in enumerate(data):
        ax.scatter(i, np.mean(vals), marker="D", s=85,
                   color=MEAN_COLOR, zorder=6,
                   linewidths=0.9, edgecolors="white")

    # ── Median + mean annotations above each distribution ────────────────────
    for i, vals in enumerate(data):
        med  = np.median(vals)
        mean = np.mean(vals)
        q3   = np.percentile(vals, 75)
        hi_w = min(vals.max(), q3 + 1.5 * (q3 - np.percentile(vals, 25)))
        top  = max(hi_w, vals.max()) + 0.04
        ax.text(i, top,
                f"med={med:.3f}\nmean={mean:.3f}",
                ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color="#222222",
                linespacing=1.4)

    # ── Axes ─────────────────────────────────────────────────────────────────
    ax.set_xticks(positions)
    ax.set_xticklabels(LABELS, fontsize=10.5)
    ax.set_ylabel("MCC  (per fold)", labelpad=9)
    ax.set_ylim(-0.08, 1.22)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.set_xlim(-0.60, len(TASKS) - 0.40)

    # ── Legend ────────────────────────────────────────────────────────────────
    fold_h = ax.scatter([], [], marker="o", s=24, color="#777",
                        alpha=0.52, linewidths=0.3, edgecolors="white",
                        label=f"Individual folds (n={n_folds})")
    mean_h = ax.scatter([], [], marker="D", s=70, color=MEAN_COLOR,
                        linewidths=0.9, edgecolors="white", label="Fold mean")
    ax.legend(handles=[fold_h, mean_h],
              loc="lower right", frameon=True,
              handletextpad=0.5, labelspacing=0.5)

    ax.set_title(
        f"B.   Per-fold MCC distribution  (MRI + GE + Clin · {n_folds} folds)",
        loc="left", fontweight="bold", pad=10,
    )


# ── Compose & save ────────────────────────────────────────────────────────────

def generate_figure(tables_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = load_summary(tables_dir)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(15, 5.5),
        gridspec_kw={"width_ratios": [1.75, 1.0], "wspace": 0.32},
    )

    draw_panel_a(axes[0], df)
    draw_panel_b(axes[1], tables_dir)

    out_path = os.path.join(out_dir, "figure_config_performance.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate config-comparison figure (Panel A + B)"
    )
    p.add_argument("--tables-dir",
                   default="results/config_comparison/tables",
                   help="Directory with table1_config_comparison.csv and per-fold CSVs")
    p.add_argument("--out-dir",
                   default="results/figures",
                   help="Output directory for the figure")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Loading tables from : {os.path.abspath(args.tables_dir)}/")
    generate_figure(args.tables_dir, args.out_dir)
    print("Done.")
