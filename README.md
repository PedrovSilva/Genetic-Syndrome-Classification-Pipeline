# Genetic Syndrome Classification Pipeline

> A reproducible machine-learning pipeline for evaluating KNN classification of genetic syndrome image embeddings.

[![CI](https://github.com/PedrovSilva/Genetic-Syndrome-Classification-Pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/PedrovSilva/Genetic-Syndrome-Classification-Pipeline/actions/workflows/ci.yml)

## Overview

This project investigates whether pre-computed image embeddings can be used to classify genetic syndromes with **K-Nearest Neighbors (KNN)**.

The pipeline covers:

- hierarchical dataset loading and flattening;
- embedding inspection and t-SNE visualization;
- KNN evaluation with **Euclidean** and **Cosine** distance;
- cross-validation with **subject-level grouping**;
- Top-1, Top-3, ROC-AUC, macro F1, weighted F1 and balanced accuracy;
- reproducible result tables and plots;
- automated tests and GitHub Actions CI.

The project is intentionally focused on the **ML evaluation pipeline**, rather than serving a model through an API.

## Pipeline

```text
Dataset (.p)
    │
    ▼
Dataset loader
    │
    ├── class distribution
    ├── subject/image metadata
    └── embeddings
    │
    ▼
Subject-level cross-validation
    │
    ├── KNN + Euclidean
    └── KNN + Cosine
    │
    ▼
Evaluation
    │
    ├── ROC-AUC
    ├── Macro / Weighted F1
    ├── Balanced Accuracy
    └── Top-1 / Top-3
    │
    ▼
CSV results + visualizations
```

## Why subject-level validation?

The dataset contains multiple images per subject. A random image-level split can place images from the same subject in both training and validation folds, which can make the reported performance overly optimistic.

The refactored pipeline therefore uses `StratifiedGroupKFold`, with `subject_id` as the grouping variable:

```python
splitter.split(X, y, groups=subject_ids)
```

This keeps all images belonging to the same subject inside a single fold.

> **Important:** historical image-level results are kept separate from the subject-level benchmark below and should not be compared as if they used the same validation protocol.

## Dataset

The repository uses the `mini_gm_public_v0.1.p` dataset supplied with the original project.

Current benchmark dataset:

| Property | Value |
|---|---:|
| Samples / images | **1,116** |
| Subjects | **941** |
| Syndromes / classes | **10** |
| Missing values | **0** |

The loader expects:

```text
syndrome_id
└── subject_id
    └── image_id
        └── embedding
```

Because the input is a Python pickle file, only use datasets from trusted sources.

## Results

### Subject-level benchmark

The refactored evaluation uses **StratifiedGroupKFold**, keeping all images from the same subject inside a single fold.

| Distance | Top-1 | Top-3 | F1 Macro | F1 Weighted | Balanced Accuracy | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Euclidean | 68.86% | 86.25% | 65.53% | 67.93% | 65.36% | 90.58% |
| **Cosine** | **74.66%** | **89.84%** | **71.99%** | **74.68%** | **71.56%** | **93.29%** |

Difference in the current benchmark summary:

| Metric | Cosine − Euclidean |
|---|---:|
| Top-1 | +5.80 pp |
| Top-3 | +3.60 pp |
| F1 Macro | +6.45 pp |
| F1 Weighted | +6.20 pp |
| Balanced Accuracy | +6.20 pp |
| ROC-AUC | +2.71 pp |

These results describe this dataset and validation protocol; they are not clinical diagnostic performance.

### Cosine KNN: effect of k

The pipeline evaluates `k=1..15`. If **macro F1** is used as the selection criterion, its highest value in the current sweep occurs at **k=8**.

| k | ROC-AUC | F1 Macro | F1 Weighted | Balanced Acc. | Top-1 | Top-3 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 77.03% | 56.15% | 59.23% | 56.55% | 59.53% | 67.97% |
| 2 | 88.29% | 63.95% | 66.63% | 64.92% | 66.61% | 84.24% |
| 3 | 91.00% | 69.66% | 72.75% | 69.94% | 72.07% | 88.43% |
| 4 | 92.32% | 70.43% | 73.14% | 70.50% | 73.32% | 89.26% |
| 5 | 93.63% | 72.30% | 74.69% | 71.86% | 74.76% | 90.32% |
| 6 | 94.24% | 73.17% | 75.50% | 72.66% | 75.80% | 90.23% |
| 7 | 94.72% | 74.49% | 76.92% | 73.91% | 75.70% | 91.37% |
| **8** | **94.95%** | **75.16%** | **77.40%** | **74.62%** | 77.10% | 91.70% |
| 9 | 94.99% | 75.13% | 77.28% | 74.40% | **77.41%** | 91.75% |
| 10 | 95.21% | 73.82% | 76.53% | 73.19% | 76.46% | 92.01% |
| 11 | 95.25% | 73.48% | 76.35% | 72.78% | 76.58% | 92.20% |
| 12 | 95.49% | 73.49% | 76.65% | 72.63% | 76.82% | 92.77% |
| 13 | 95.72% | 73.02% | 76.12% | 72.06% | 76.77% | 92.77% |
| 14 | 95.87% | 73.44% | 76.21% | 72.57% | 76.02% | 92.64% |
| **15** | **95.88%** | 73.75% | 76.42% | 72.59% | 76.58% | **93.11%** |

The metric optimum depends on the criterion: macro F1 peaks at `k=8`, Top-1 at `k=9`, while ROC-AUC and Top-3 reach their highest values at `k=15`.

### Historical baseline

The original implementation reported image-level cross-validation results:

| Distance | Accuracy | F1 | AUC |
|---|---:|---:|---:|
| Euclidean | 72.01% | 70.78% | 92.27% |
| Cosine | 77.31% | 76.96% | 94.25% |

These values are retained as historical results and are **not directly comparable** with the subject-level benchmark because the validation protocol changed.

### Generated artifacts

```text
data/output/
├── knn_results_statistics.csv
├── knn_summary_table.csv
├── results_knn_comparison.csv
├── results_knn_cosine.csv
├── results_knn_euclidean.csv
└── plots/
    ├── auc_comparison.png
    └── tsne_visualization.png
```

## Project structure

```text
.
├── .devcontainer/
├── .github/workflows/ci.yml
├── data/
├── scripts/run_pipeline.py
├── src/
│   ├── analysis.py
│   ├── dataset.py
│   └── models/knn_classifier.py
├── tests/test_knn_classifier.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Installation

The project targets **Python 3.12** for reproducible scientific Python dependencies.

```bash
git clone https://github.com/PedrovSilva/Genetic-Syndrome-Classification-Pipeline.git
cd Genetic-Syndrome-Classification-Pipeline

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running

```bash
python -m scripts.run_pipeline
pytest -q
```

## Reproducibility

The evaluation uses:

- fixed random seed: `42`;
- stratified grouped cross-validation;
- deterministic t-SNE configuration;
- pinned direct Python dependencies;
- automated tests;
- GitHub Actions CI.

CI validates source compilation and tests rather than executing the complete ML evaluation.

## Methodology

### KNN

The classifier evaluates `k=1..15` using Euclidean and Cosine distance. For every fold, training contains only subjects assigned to that training partition.

### Metrics

- **Top-1 accuracy**
- **Top-3 accuracy**
- **ROC-AUC (weighted, one-vs-rest)**
- **F1 macro**
- **F1 weighted**
- **Balanced accuracy**

Macro F1 is included because weighted F1 can hide poor performance on minority classes.

### t-SNE

t-SNE is used only as an exploratory visualization of the embedding space. It is not evidence of model generalization.

## Limitations

- Pre-computed embeddings are evaluated; the image encoder is not trained here.
- The dataset is relatively small and class-imbalanced.
- KNN is an interpretable baseline rather than a final model.
- t-SNE is exploratory and sensitive to its parameters.
- Subject-level validation reduces leakage risk, but performance still depends on dataset composition and provenance.
- This is a research/engineering benchmark, **not a clinical diagnostic validation**.
- An independent subject-level holdout would provide a stronger external check.

## Future work

1. Add confusion matrices and per-class precision/recall/F1.
2. Compare KNN with SVM, logistic regression and tree-based baselines.
3. Add confidence intervals across folds.
4. Add experiment tracking.
5. Add configuration for dataset paths and model parameters.
6. Document data provenance and dataset/license.
7. Evaluate an independent subject-level holdout.
8. Investigate embedding normalization and dimensionality-reduction strategies.

## Development

The repository includes a Dev Container and GitHub Actions CI.

```text
Checkout
   ↓
Python 3.12
   ↓
Install dependencies
   ↓
Compile source
   ↓
Run pytest
```

## License

Add the appropriate license and dataset attribution before publishing the project as a reusable research artifact.
