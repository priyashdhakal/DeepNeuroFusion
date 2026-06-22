"""DeepNeuroFusion — single-config pipeline.

Usage
-----
    python run.py                          # defaults
    python run.py --output-dir my_results
    python run.py --n-repeats 3 --n-folds 3   # quick smoke-test
    python run.py --gpu 1                  # use GPU 1
    python run.py --skip-profiling         # skip Step 8

Output layout
-------------
results/
  figures/
    exploratory_data_analysis/  — 6 EDA figures
    performance.png
    confusion_matrices.png
    roc_curves.png
    metric_distributions.png
    training_history_*.png
    shap_*_bar.png  shap_*_beeswarm.png  shap_*_modality.png
  tables/
    results_summary.csv
    metrics_per_fold_*.csv
    shap_importance_*.csv
  profile/
    complexity_report.txt
    timing_breakdown.png
    memory_profile.png
    model_complexity.png
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deepneurofusion import load_data, run_task, set_seed, set_device, TASK_NAMES
from deepneurofusion.profiling import run_profiling
from deepneurofusion.plots import (
    plot_data_overview,
    plot_performance,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_metric_distributions,
    plot_training_history,
    run_shap_analysis,
    save_results_csv,
    save_shap_tables,
)

MRI_PATH  = "/media/8TB_hardisk/priyash/ANMERGE_PROJ/data/mri_common_under_90.csv"
GENE_PATH = "/media/8TB_hardisk/priyash/ANMERGE_PROJ/data/gene_common_under_90.csv"

_TASK_DEFS = [
    ("AD_vs_CN",  lambda y: (y == "AD") | (y == "CTL")),
    ("AD_vs_MCI", lambda y: (y == "AD") | (y == "MCI")),
    ("MCI_vs_CN", lambda y: (y == "MCI") | (y == "CTL")),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DeepNeuroFusion pipeline")
    p.add_argument("--output-dir", default="results",
                   help="Root output directory (default: results/)")
    p.add_argument("--n-repeats", type=int, default=10,
                   help="CV repeat count (default: 10)")
    p.add_argument("--n-folds", type=int, default=5,
                   help="CV fold count (default: 5)")
    p.add_argument("--gpu", type=int, default=0,
                   help="GPU device index (default: 0)")
    p.add_argument("--skip-profiling", action="store_true",
                   help="Skip Step 8 complexity/timing profiling")
    p.add_argument("--profile-reps", type=int, default=3,
                   help="Repetitions for timing averaging in profiling (default: 3)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    active_device = set_device(args.gpu)
    set_seed()

    output_dir  = args.output_dir
    fig_dir     = os.path.join(output_dir, "figures")
    eda_dir     = os.path.join(output_dir, "figures", "exploratory_data_analysis")
    csv_dir     = os.path.join(output_dir, "tables")
    profile_dir = os.path.join(output_dir, "profile")
    for d in [fig_dir, eda_dir, csv_dir]:
        os.makedirs(d, exist_ok=True)

    n_steps = 7 if args.skip_profiling else 8

    print("=" * 70)
    print("DeepNeuroFusion — Multimodal AD Classification Pipeline")
    print(f"Device  : {active_device}")
    print(f"Repeats : {args.n_repeats} × {args.n_folds}-fold CV")
    print(f"Output  : {os.path.abspath(output_dir)}/")
    print("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print(f"\n[1/{n_steps}] Loading data…")
    X_mri, X_gene, y_all, mri_names, gene_names = load_data(MRI_PATH, GENE_PATH)

    # ── 2. EDA figures ────────────────────────────────────────────────────────
    print(f"\n[2/{n_steps}] Generating EDA figures…")
    plot_data_overview(X_mri, X_gene, y_all, mri_names, gene_names, eda_dir)

    # ── 3. Classification tasks ───────────────────────────────────────────────
    print(f"\n[3/{n_steps}] Running classification tasks…")
    t0 = time.time()
    all_results: dict = {}
    for task_name, mask_fn in _TASK_DEFS:
        mask = mask_fn(y_all)
        all_results[task_name] = run_task(
            task_name, X_mri, X_gene, y_all, mask,
            mri_names, gene_names,
            n_repeats=args.n_repeats,
            n_folds=args.n_folds,
        )
    print(f"\n  All tasks complete in {(time.time()-t0)/60:.1f} min")

    # ── 4. Result figures ─────────────────────────────────────────────────────
    print(f"\n[4/{n_steps}] Generating result figures…")
    plot_performance(all_results, fig_dir)
    plot_confusion_matrices(all_results, fig_dir)
    plot_roc_curves(all_results, fig_dir)
    plot_metric_distributions(all_results, fig_dir)
    for task_name in all_results:
        trainer = all_results[task_name].get("last_trainer")
        if trainer:
            plot_training_history(trainer, task_name, fig_dir)

    # ── 5. SHAP analysis ──────────────────────────────────────────────────────
    print(f"\n[5/{n_steps}] Running SHAP analysis…")
    shap_results: dict = {}
    for task_name in all_results:
        shap_results[task_name] = run_shap_analysis(
            all_results[task_name], mri_names, gene_names, fig_dir,
        )

    # ── 6. Save CSV tables ────────────────────────────────────────────────────
    print(f"\n[6/{n_steps}] Saving result tables…")
    save_results_csv(all_results, csv_dir)
    save_shap_tables(shap_results, csv_dir)

    # ── 7. Final summary ──────────────────────────────────────────────────────
    print(f"\n[7/{n_steps}] Summary")
    print("\n" + "=" * 70)
    print("DeepNeuroFusion — Final Results")
    print("=" * 70)
    for task_name in TASK_NAMES:
        if task_name not in all_results:
            continue
        r = all_results[task_name]["result_summary"]
        print(f"\n  {task_name}:")
        for m in ["Acc", "Sens", "Spec", "BA", "AUC", "MCC", "F1"]:
            print(f"    {m:<6}: {r[m]:.4f} ± {r[f'{m}_std']:.4f}")

    # ── 8. Computational complexity profiling ─────────────────────────────────
    if not args.skip_profiling:
        print(f"\n[8/{n_steps}] Computational complexity & profiling…")
        run_profiling(
            X_mri, X_gene, y_all,
            out_dir=profile_dir,
            n_reps=args.profile_reps,
        )

    print(f"\n{'='*70}")
    print(f"All outputs saved to: {os.path.abspath(output_dir)}/")
    print(f"  figures/exploratory_data_analysis/  — EDA figures")
    print(f"  figures/      — Performance, ROC, SHAP figures")
    print(f"  tables/       — CSV summary & per-fold metrics")
    if not args.skip_profiling:
        print(f"  profile/      — Complexity report, timing & memory figures")
    print(f"{'='*70}")
    print("Done!")


if __name__ == "__main__":
    main()
