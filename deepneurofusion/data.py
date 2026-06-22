from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def load_data(
    mri_path: str,
    gene_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """Load and align MRI volumetric and gene-expression CSVs.

    Returns
    -------
    X_mri : ndarray  (n_subjects, n_mri_features + n_clinical)
        Imputed MRI features with Age and APOE appended.
    X_gene : ndarray  (n_subjects, n_gene_features)
        Imputed gene-expression features.
    y : ndarray  (n_subjects,)
        Diagnosis labels ('AD', 'MCI', 'CTL').
    mri_feature_names : list[str]
    gene_feature_names : list[str]
    """
    mri_df = pd.read_csv(mri_path)
    gene_df = pd.read_csv(gene_path)

    print(f"  MRI  shape : {mri_df.shape}")
    print(f"  Gene shape : {gene_df.shape}")
    print(f"  Class dist : {mri_df['Diagnosis'].value_counts().to_dict()}")

    # APOE allele → ε4 count (0, 1, or 2)
    if "APOE" in mri_df.columns:
        mri_df["APOE"] = mri_df["APOE"].apply(
            lambda v: np.nan if pd.isna(v) else str(v).upper().replace(" ", "").count("4")
        )

    metadata_exclude = {
        "Unnamed: 0", "Visit", "Month", "Site", "Diagnosis",
        "Sex", "Age", "APOE", "MMSE", "CDRSB", "ADAS11",
        "ADAS13", "Gexp_batch",
    }

    mri_cols = [c for c in mri_df.columns if c not in metadata_exclude]
    gene_cols = [c for c in gene_df.columns if c not in metadata_exclude]

    X_mri_raw = mri_df[mri_cols].values.astype(float)

    clinical_cols = [c for c in ("Age", "APOE") if c in mri_df.columns]
    if clinical_cols:
        X_mri_raw = np.hstack([X_mri_raw, mri_df[clinical_cols].values.astype(float)])
    print(f"  Clinical covariates appended: {clinical_cols}")

    X_gene_raw = gene_df[gene_cols].values.astype(float)

    X_mri_raw = SimpleImputer(strategy="mean").fit_transform(X_mri_raw)
    X_gene_raw = SimpleImputer(strategy="mean").fit_transform(X_gene_raw)

    y_all = mri_df["Diagnosis"].values
    mri_feature_names = mri_cols + clinical_cols
    gene_feature_names = gene_cols

    print(f"  Final MRI features  : {X_mri_raw.shape[1]}")
    print(f"  Final Gene features : {X_gene_raw.shape[1]}")

    return X_mri_raw, X_gene_raw, y_all, mri_feature_names, gene_feature_names
