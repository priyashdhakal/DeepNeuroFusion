# DeepNeuroFusion
**A Deep Learning Ensemble Framework for Multimodal Classification of Alzheimer's Disease Stages Using MRI and Transcriptomic Data**
DeepNeuroFusion fuses structural MRI (with age and APOE-ε4 dosage) and whole-blood gene expression into a shared 64-dimensional latent space, then combines a linear SVM, an RBF-SVM on the learned embedding, and a deep MLP head through task-specific weighted probability fusion. All preprocessing, gene selection, and training are recomputed inside each fold (fold-safe) under repeated 10 × 5 stratified cross-validation. Evaluated on the [ANMerge](https://www.synapse.org/#!Synapse:syn22252881) cohort for three tasks: **AD vs. CN**, **AD vs. MCI**, **MCI vs. CN**.
## Installation

Requires Python 3.10 (GPU optional).

```bash
conda create -n deepneurofusion python=3.10.5 -y && conda activate deepneurofusion
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Data

ANMerge is access-controlled (Synapse `syn22252881`) and **not** included here. Provide two row-aligned CSVs — an MRI file (FreeSurfer features + `Diagnosis`, `Age`, `APOE`) and a gene-expression file — then set the paths in `run.py`:

```python
MRI_PATH  = "/path/to/mri_data.csv"
GENE_PATH = "/path/to/gene_data.csv"
```

## Usage

```bash
python run.py                          # full pipeline (CV, figures, SHAP, profiling)
python run.py --n-repeats 3 --n-folds 3   # quick smoke test
python run.py --gpu 1 --skip-profiling
python figure_config_comparison.py     # configuration-comparison figure
```

Outputs are written to `results/` (`figures/`, `tables/`, `profile/`). Per-task hyperparameters live in `deepneurofusion/config.py`. Runs are fully deterministic (`SEED = 42`).

## Acknowledgements

We thank the ANMerge and AddNeuroMed consortium for making the harmonised multimodal dataset publicly available. AddNeuroMed and InnoMed were funded by the European Commission. The views expressed here are those of the authors and do not necessarily reflect those of the data providers.
