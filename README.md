# AMR Resistance Predictor

> **XGBoost + SHAP for predicting aminoglycoside resistance in *Acinetobacter baumannii***

[![Python Version](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](https://opensource.org/licenses/MIT)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![DOI](https://img.shields.io/badge/DOI-pending-orange)]()

---

## Overview

Antimicrobial resistance (AMR) is a global health crisis, with **Acinetobacter baumannii** emerging as a critical priority pathogen. Aminoglycoside-modifying enzymes (AMEs) like **ANT(3'')-Ia** confer resistance to clinically important antibiotics, making treatment increasingly difficult.

This project builds a **machine learning pipeline** that predicts aminoglycoside resistance from phenotypic metadata alone (drug identity and testing method), using **XGBoost** with **SHAP** interpretability.

### Why This Matters

- **Clinical relevance**: Predict resistance patterns to guide antibiotic selection
- **Interpretability**: SHAP reveals *why* the model makes predictions
- **Reproducibility**: Full pipeline from data download to model interpretation
- **Biological context**: Directly relevant to AME-mediated resistance mechanisms

---

## Key Results

| Metric | Value | 95% Confidence Interval |
|--------|-------|-------------------------|
| **ROC-AUC** | 0.742 | [0.706, 0.777] |
| **F1 Score** | 0.820 | - |
| **Accuracy** | 0.67 | - |

### Feature Importance (SHAP)

![SHAP Summary](outputs/figures/06_shap_summary_beeswarm.png)

**Gentamicin** is the strongest predictor of resistance, consistent with the prevalence of ANT(3'')-Ia enzymes in *A. baumannii*. Amikacin shows weaker signal, reflecting its structural modifications that evade common resistance mechanisms.

### Model Performance

![ROC Curve](outputs/figures/05_roc_curve.png)

---

## Methods

### Data Source

- **Database**: BV-BRC (Bacterial and Viral Bioinformatics Resource Center)
- **Organism**: *Acinetobacter baumannii* (NCBI Taxon ID: 470)
- **Antibiotics**: Amikacin, Gentamicin, Tobramycin
- **Records**: 4,172 laboratory-confirmed phenotypes
- **Filtered to**: Laboratory-confirmed evidence only

### Feature Engineering

| Feature Type | Features Created |
|--------------|------------------|
| Drug identity | One-hot encoded antibiotics (3) |
| Testing method | One-hot encoded methods (3) |
| **Total features** | **6 binary features** |

### Models Evaluated

| Model | ROC-AUC | F1 | Training Time |
|-------|---------|----|---------------|
| **XGBoost** | 0.7423 | 0.8204 | 0.50s |
| **Random Forest** | 0.7423 | 0.8204 | 0.78s |
| **LightGBM** | 0.7423 | 0.8204 | 0.29s |
| **Logistic Regression** | 0.7372 | 0.7323 | 0.17s |

**Why XGBoost?**
- Matches or exceeds performance of other models
- SHAP interpretability is built-in
- Industry standard for tabular data

### Advanced Validation

| Method | Result |
|--------|--------|
| **Leave-One-Group-Out CV** | ROC-AUC: 0.887 ± 0.231 |
| **Bootstrap (95% CI)** | [0.706, 0.777] |
| **XGBoost vs Logistic Regression** | p = 0.922 (not significant) |

---

## Installation

### Prerequisites

- Python 3.11+
- Conda (recommended) or pip

### Setup with Conda

```bash
# Clone the repository
git clone https://github.com/menzisk/amr-resistance-predictor.git
cd amr-resistance-predictor

# Create and activate environment
conda env create -f environment.yml
conda activate amr-predictor
