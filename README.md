# DecodeLabs AI — Project 2: Data Classification Using AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Datasets-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-22C55E?style=for-the-badge)
![DecodeLabs](https://img.shields.io/badge/DecodeLabs-Batch%202026-7C3AED?style=for-the-badge)

**Industrial Training Kit · Batch 2026 · Powered by DecodeLabs**

*A production-grade supervised learning pipeline for multi-class data classification*

</div>

---

## 📄 Abstract

This project implements a complete, end-to-end supervised learning pipeline for multi-class data classification using the Iris benchmark dataset. It covers every stage of the ML engineering lifecycle — from raw data ingestion with source fallback logic and SHA-256 integrity hashing, through exploratory analysis, outlier detection, dimensionality reduction, preprocessing, model training, hyperparameter tuning, cross-validation with statistical significance testing, and model explainability — culminating in a persisted sklearn Pipeline and a real-time interactive prediction interface.

Rather than satisfying the minimum specification of training a single KNN classifier, this project trains and rigorously compares five models: KNN (optimal k via elbow method), Decision Tree, Support Vector Machine, Logistic Regression, and a soft-voting Ensemble. Each model is evaluated across five metrics, subjected to stratified 5-fold cross-validation, and ranked by statistical significance using a paired t-test. SHAP explainability, calibration analysis, learning curves, and a custom domain-aware weighted scorer are applied to the best-performing model.

All code is modular, fully documented, and organised as a deployable Python package (`pipeline/`) with 22 pytest unit tests. The project is designed to function as both an internship submission and a portfolio artifact demonstrating production-level ML engineering practice.

---

## 📋 Table of Contents

- [Project Architecture](#-project-architecture)
- [Dataset](#-dataset)
- [Notebook Structure](#-notebook-structure)
- [Results](#-results)
- [Visualizations](#-visualizations)
- [Reproducibility](#-reproducibility)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Known Limitations](#-known-limitations)
- [Future Work](#-future-work)
- [License](#-license)
- [Author](#-author)

---

## 🏗 Project Architecture

```
Raw Data
  ├── Primary   : HuggingFace  →  scikit-learn/iris
  └── Fallback  : sklearn.datasets.load_iris()
          │
          ▼
SHA-256 Data Hash  (integrity versioning per run)
          │
          ▼
Data Validation
  ├── Shape assertion     (150 rows · 4 features)
  ├── Class assertion     (3 unique targets)
  ├── Null check          (zero missing values)
  └── Duplicate check     (zero full duplicates)
          │
          ▼
Exploratory Data Analysis
  ├── Descriptive statistics · class distribution
  ├── KDE + rug plots · correlation heatmap
  ├── Pairplot · boxplots (IQR outlier overlay)
  └── Violin plots
          │
          ▼
Dimensionality Reduction
  ├── PCA  (2 components — explained variance annotated)
  └── t-SNE  (perplexity=30 — cluster separability)
          │
          ▼
Preprocessing
  ├── Feature / Target split
  ├── Stratified Train-Test Split  (80 / 20 · seed=42)
  └── StandardScaler  (fit on X_train ONLY → transform both)
          │
          ▼
KNN Hyperparameter Tuning
  └── Elbow Method  (k = 1→15 · optimal k = 3)
          │
          ▼
Model Training
  ├── KNN                  (n_neighbors = optimal_k)
  ├── Decision Tree        (max_depth=5)
  ├── SVM                  (kernel=rbf · probability=True)
  ├── Logistic Regression  (max_iter=1000)
  └── Voting Ensemble      (soft voting · all 4 base models)
          │
          ▼
Stratified K-Fold Cross-Validation  (5 folds · f1_macro)
          │
          ▼
Statistical Significance  (paired t-test · best vs 2nd best)
          │
          ▼
GridSearchCV  (best model · f1_macro · cv=5 · n_jobs=-1)
          │
          ▼
Full Evaluation
  ├── Accuracy · F1 · Precision · Recall · ROC-AUC
  ├── Per-class classification report
  ├── Confusion matrices · ROC curves
  ├── SHAP explainability  (KernelExplainer)
  ├── Calibration curves   (reliability diagrams)
  ├── Learning curves      (overfitting diagnosis)
  ├── Feature importance   (permutation-based)
  ├── Custom weighted scorer  (domain-aware F1)
  └── Class imbalance demonstration  (accuracy mirage proof)
          │
          ▼
sklearn Pipeline Object  (StandardScaler + best classifier)
          │
          ▼
Model Persistence  (joblib · best_model_pipeline.pkl)
          │
          ▼
Experiment Logging  (CSV · timestamp · metrics · params · hash)
          │
          ▼
Interactive Prediction Interface  (ipywidgets · confidence threshold)
          │
          ▼
Requirements Freeze  (exact environment snapshot)
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Name | Iris Benchmark |
| Primary Source | HuggingFace — `scikit-learn/iris` |
| Fallback Source | `sklearn.datasets.load_iris()` |
| Samples | 150 — balanced (50 per class) |
| Classes | 3 — Setosa · Versicolor · Virginica |
| Features | 4 — sepal length · sepal width · petal length · petal width |
| Missing Values | None |
| Duplicates | None |
| Train Split | 120 samples (80%) — stratified |
| Test Split | 30 samples (20%) — stratified |
| Data Hash (SHA-256) | `67037d1d5dd80fe8dcd69abe1ead3074...` |
| Random State | 42 |

---

## 📓 Notebook Structure

The main notebook (`notebooks/ai_classification_project2.ipynb`) contains **54 cells** — 38 code, 16 markdown — organised across 13 sections:

| Section | Title | Key Cells |
|---|---|---|
| 1 | Data Loading | HF loader · fallback · SHA-256 hash · validation |
| 2 | Exploratory Data Analysis | 6 visualization types · IQR outlier detection |
| 3 | Preprocessing | Feature split · stratified split · StandardScaler |
| 3.5 | Dimensionality Reduction | PCA + t-SNE projections |
| 4 | KNN Hyperparameter Tuning | Elbow method · optimal k = 3 |
| 5 | Model Training | 4 base classifiers + Voting Ensemble |
| 6 | Model Evaluation | 5 metrics · classification reports · confusion matrices · ROC |
| 7 | Cross-Validation & Significance | StratifiedKFold · paired t-test |
| 8 | Decision Tree & Feature Importance | Tree structure · permutation importance |
| 9 | Custom Scorer & Calibration | Domain-aware weighted F1 · reliability diagrams |
| 10 | SHAP Explainability | KernelExplainer · summary + waterfall plots |
| 11 | Learning Curve Analysis | Overfitting diagnosis · shaded std bands |
| 12 | Pipeline & Persistence | sklearn Pipeline · joblib save/reload · ipywidgets UI |
| 13 | Model Card & Experiment Log | Structured model card · CSV experiment tracker |

---

## 📈 Results

> All metrics computed on the held-out test set (30 samples · stratified 20% split).
> Best model determined by F1 macro score.

### Model Comparison — Test Set

| Model | Accuracy | F1 (macro) | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|
| **SVM (RBF) ✦ Best** | **0.9667** | **0.9666** | **0.9697** | **0.9667** | **0.9967** |
| KNN (k=3) | 0.9667 | 0.9666 | 0.9697 | 0.9667 | 0.9967 |
| Logistic Regression | 0.9667 | 0.9666 | 0.9697 | 0.9667 | 0.9967 |
| Voting Ensemble | 0.9667 | 0.9666 | 0.9697 | 0.9667 | 0.9967 |
| Decision Tree | 0.9333 | 0.9333 | 0.9352 | 0.9333 | 0.9780 |

### Cross-Validation Results — 5-Fold Stratified (F1 macro)

| Model | Mean F1 | Std F1 |
|---|---|---|
| SVM (RBF) | ~0.973 | ±0.025 |
| KNN (k=3) | ~0.960 | ±0.030 |
| Logistic Regression | ~0.960 | ±0.030 |
| Voting Ensemble | ~0.967 | ±0.025 |
| Decision Tree | ~0.940 | ±0.040 |

### Statistical Significance

Paired t-test between SVM (best) and KNN (second-best) CV fold scores.
Results logged in `logs/experiment_log.csv` per run.
Winner declared with t-statistic and p-value printed in Cell 38.

### Best Model Summary

| Property | Value |
|---|---|
| Algorithm | SVC (RBF kernel) |
| Accuracy | 0.9667 |
| F1 (macro) | 0.9666 |
| Precision (macro) | 0.9697 |
| Recall (macro) | 0.9667 |
| ROC-AUC (OvR) | 0.9967 |
| Weighted F1 | 0.9599 |
| Calibration Deviation | 0.0990 |
| Optimal KNN k | 3 |
| Data Source | HuggingFace |

---

## 🖼 Visualizations

22 plots generated and saved to `assets/` on every run:

| File | Description |
|---|---|
| `class_distribution.png` | Bar chart — sample count per species |
| `feature_distributions.png` | KDE + rug plots per feature colored by class |
| `correlation_heatmap.png` | Pearson r matrix — all 4 features annotated |
| `pairplot.png` | All feature combinations colored by species |
| `boxplots.png` | IQR bounds per feature with outlier markers overlaid |
| `violin_plots.png` | Distribution shape + IQR per feature per class |
| `scaling_comparison.png` | Raw vs StandardScaler feature distribution |
| `pca_tsne_projection.png` | PCA + t-SNE 2D projections — class separability |
| `knn_elbow.png` | Error rate vs k — optimal k marked with dashed line |
| `model_comparison.png` | Grouped bar chart — 5 metrics × 5 models |
| `confusion_matrix_knn.png` | Annotated heatmap — KNN |
| `confusion_matrix_decision_tree.png` | Annotated heatmap — Decision Tree |
| `confusion_matrix_svm.png` | Annotated heatmap — SVM |
| `confusion_matrix_logistic_regression.png` | Annotated heatmap — Logistic Regression |
| `confusion_matrix_ensemble.png` | Annotated heatmap — Voting Ensemble |
| `roc_curve_svm.png` | ROC curves per class with AUC — best model |
| `decision_tree.png` | Decision Tree structure + feature importances |
| `feature_importance.png` | Permutation importance horizontal bars — all models |
| `calibration_curves.png` | Reliability diagrams — predicted vs actual probability |
| `cv_boxplot.png` | CV score distribution with error bars per model |
| `learning_curve_svm.png` | Train vs validation score — overfitting diagnosis |
| `shap_summary_svm.png` | SHAP summary + waterfall plot — best model |

---

## 🔁 Reproducibility

```bash
# 1. Clone the repository
git clone https://github.com/whozahm3d/decodelabs-ai-project2.git
cd decodelabs-ai-project2

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open the notebook
jupyter notebook notebooks/ai_classification_project2.ipynb

# 4. Run all cells
# Kernel → Restart & Run All
```

For an exact environment match using the frozen snapshot:

```bash
pip install -r requirements_frozen.txt
```

All randomness is seeded with `RANDOM_STATE = 42`.
Every run logs a SHA-256 data hash to `logs/experiment_log.csv` for full traceability.

> **Windows users:** The `BASE_DIR` constant in Cell 4 is set to the project root.
> Update this path if you move the project folder.

---

## 📁 Project Structure

```
decodelabs-ai-project2/
│
├── notebooks/
│   └── ai_classification_project2.ipynb   ← 54-cell main notebook
│
├── pipeline/                               ← Importable Python package
│   ├── __init__.py                         ← Package init
│   ├── data_loader.py                      ← HF→sklearn fallback · SHA-256 hash
│   ├── preprocessor.py                     ← Split · scale · IQR outlier detection
│   ├── trainer.py                          ← 4 models · ensemble · sklearn Pipeline
│   ├── evaluator.py                        ← Metrics · CV · t-test · experiment log
│   └── visualizer.py                       ← 22 plot functions → assets/
│
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py                    ← 22 pytest unit tests
│
├── models/                                 ← Generated at runtime
│   ├── best_model_pipeline.pkl             ← Saved sklearn Pipeline (SVM)
│   └── model_card.txt                      ← Auto-generated structured model card
│
├── assets/                                 ← Generated at runtime
│   └── *.png                               ← 22 plot exports
│
├── logs/                                   ← Generated at runtime
│   └── experiment_log.csv                  ← Per-run metrics · params · data hash
│
├── .gitignore                              ← Python · Jupyter · pkl · csv
├── requirements.txt                        ← Minimum version pins
└── requirements_frozen.txt                 ← Exact environment snapshot (runtime)
```

---

## 🛠 Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core language |
| scikit-learn | 1.8.0 | Models · metrics · Pipeline · GridSearchCV |
| pandas | 3.0.3 | Data manipulation · experiment logging |
| numpy | 1.26.4 | Numerical computing · array operations |
| matplotlib | 3.10.9 | Base plotting engine |
| seaborn | 0.13.2 | Statistical visualizations |
| HuggingFace datasets | 4.8.5 | Primary data source |
| scipy | 1.17.1 | Paired t-test · statistical significance |
| shap | 0.51.0 | Model explainability · KernelExplainer |
| joblib | 1.5.3 | Model serialization · persistence |
| ipywidgets | 7.8.1 | Interactive real-time prediction UI |
| nbformat | 5.10.4 | Notebook format utilities |
| pytest | 7.4.4 | Unit testing framework |
| Jupyter Notebook | 7.0+ | Development environment |

---

## ⚠ Known Limitations

- **KNN does not scale** — inference complexity is O(n × d) at prediction time. Unsuitable for datasets beyond ~100k rows without approximate nearest-neighbour methods such as FAISS or Annoy.
- **Iris is a benchmark, not a real-world problem** — near-perfect class separability (Setosa is linearly separable) and a balanced 150-sample dataset are not representative of production data with noise, missing values, and class imbalance.
- **GridSearchCV covers a fixed param grid** — production pipelines would use Bayesian optimisation (Optuna, Ray Tune) for more efficient hyperparameter search across a larger space.
- **SHAP approximation** — KernelExplainer uses 50 background samples for speed. TreeExplainer (exact) is available for tree-based models but not for SVM.
- **No external validation** — all evaluation is in-distribution. A geographically or temporally distinct held-out dataset would be required to confirm generalisation.

---

## 🚀 Future Work

- **Project 3 — Deep Learning & CNNs** — transition from tabular classification to image classification using convolutional neural networks, building on the supervised learning foundations established here.
- **Scalable inference** — replace KNN with a neural network classifier and deploy the best model as a REST API using FastAPI + Docker for production serving.
- **Experiment tracking** — integrate MLflow or Weights & Biases to replace the manual CSV logger with a full experiment management system supporting run comparison, artifact versioning, and hyperparameter visualisation.
- **AutoML benchmark** — compare the manual pipeline against TPOT or H2O AutoML to quantify the performance gains from human-guided feature engineering and domain-aware evaluation.

---

## 📜 License

MIT License — Copyright (c) 2026 Ali Ahmad

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## 👤 Author

**Ali Ahmad**
BS Data Science — FAST NUCES Lahore (6th Semester)
AI Engineering Intern — DecodeLabs Batch 2026

[![GitHub](https://img.shields.io/badge/GitHub-whozahm3d-181717?style=flat-square&logo=github)](https://github.com/whozahm3d)

---

<div align="center">

*Built with precision as part of the DecodeLabs Industrial AI Training Program · Batch 2026*

</div>
