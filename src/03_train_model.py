#!/usr/bin/env python3
"""
03_train_model.py

Trains an XGBoost classifier for aminoglycoside resistance prediction.

This script:
    1. Loads the engineered feature set
    2. Splits data into stratified train/test sets (80/20)
    3. Computes class weights to handle imbalance (scale_pos_weight)
    4. Trains XGBoost with cross-validation (5-fold stratified)
    5. Evaluates on held-out test set
    6. Saves model and evaluation results

Inputs:
    data/processed/amr_features.tsv
        - Engineered features from 02_clean_and_engineer.py

Outputs:
    outputs/models/evaluation_results.json
        - ROC-AUC, F1, and CV results
    outputs/models/xgboost_model.pkl
        - Trained model (saved with joblib)
    outputs/figures/04_confusion_matrix.png
    outputs/figures/05_roc_curve.png

Usage:
    python src/03_train_model.py

References:
    Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System. KDD 2016.
    https://arxiv.org/abs/1603.02754

Author: Menzi Sikakane (menzisk)
Date:   2026-06-17
License: MIT
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # For headless Linux/WSL
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import yaml
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
)
from xgboost import XGBClassifier

# ── Load Configuration ─────────────────────────────────────────────────────
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

# Extract settings
PROCESSED_DIR = CONFIG['data']['processed_dir']
MODEL_DIR = CONFIG['outputs']['model_dir']
FIG_DIR = CONFIG['outputs']['figure_dir']
RANDOM_STATE = CONFIG['model']['random_state']
TEST_SIZE = CONFIG['model']['test_size']
N_FOLDS = CONFIG['model']['n_folds']
DPI = CONFIG['outputs']['dpi']

# ── Constants ──────────────────────────────────────────────────────────────

INPUT_FILE = os.path.join(PROCESSED_DIR, "amr_features.tsv")
RESULTS_FILE = os.path.join(MODEL_DIR, "evaluation_results.json")
MODEL_FILE = os.path.join(MODEL_DIR, "xgboost_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.2)


# ── Functions ──────────────────────────────────────────────────────────────

def load_features(filepath: str):
    """Load the engineered feature set. Separates features (X) from label (y)."""
    df = pd.read_csv(filepath, sep="\t")
    print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols]
    y = df["label"]

    print(f"Features (X): {X.shape}")
    print(f"Label    (y): {y.shape}")
    print(f"\nClass distribution:")
    counts = y.value_counts()
    for val, count in counts.items():
        name = "Resistant" if val == 1 else "Susceptible"
        print(f"  {name} ({val}): {count:,}  ({count/len(y)*100:.1f}%)")

    return X, y


def compute_class_weight(y: pd.Series) -> float:
    """Compute the scale_pos_weight parameter for XGBoost."""
    n_resistant = (y == 1).sum()
    n_susceptible = (y == 0).sum()
    weight = n_susceptible / n_resistant

    print(f"\nClass weight (scale_pos_weight):")
    print(f"  n_susceptible / n_resistant = {n_susceptible} / {n_resistant} = {weight:.4f}")

    return weight


def split_data(X: pd.DataFrame, y: pd.Series):
    """Split data into training and test sets with stratification."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\nTrain/test split (stratified {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}):")
    print(f"  Training set : {X_train.shape[0]:,} records")
    print(f"  Test set     : {X_test.shape[0]:,} records")

    train_rate = y_train.mean() * 100
    test_rate = y_test.mean() * 100
    print(f"  Resistance rate in train: {train_rate:.1f}%")
    print(f"  Resistance rate in test : {test_rate:.1f}%")

    return X_train, X_test, y_train, y_test


def build_model(scale_pos_weight: float) -> XGBClassifier:
    """Define the XGBoost classifier with hyperparameters from config."""
    model = XGBClassifier(
        n_estimators=CONFIG['model']['n_estimators'],
        max_depth=CONFIG['model']['max_depth'],
        learning_rate=CONFIG['model']['learning_rate'],
        subsample=CONFIG['model']['subsample'],
        colsample_bytree=CONFIG['model']['colsample_bytree'],
        scale_pos_weight=scale_pos_weight,
        eval_metric=CONFIG['model']['eval_metric'],
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    return model


def run_cross_validation(model: XGBClassifier, X_train, y_train) -> dict:
    """Run stratified k-fold cross-validation on the training set."""
    print(f"\nRunning {N_FOLDS}-fold stratified cross-validation...")

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    cv_results = cross_validate(
        model, X_train, y_train,
        cv=cv,
        scoring=["roc_auc", "f1", "accuracy"],
        return_train_score=True,
    )

    print(f"\nCross-validation results (training set, {N_FOLDS} folds):")
    print(f"  {'Metric':<15} {'Mean':>8}  {'Std':>8}")
    print(f"  {'-'*35}")

    summary = {}
    for metric in ["roc_auc", "f1", "accuracy"]:
        test_scores = cv_results[f"test_{metric}"]
        train_scores = cv_results[f"train_{metric}"]
        mean_test = test_scores.mean()
        std_test = test_scores.std()
        mean_train = train_scores.mean()

        print(f"  {metric:<15} {mean_test:>8.4f}  {std_test:>8.4f}  "
              f"(train: {mean_train:.4f})")

        summary[metric] = {
            "mean_test": round(mean_test, 4),
            "std_test": round(std_test, 4),
            "mean_train": round(mean_train, 4),
        }

    roc_gap = summary["roc_auc"]["mean_train"] - summary["roc_auc"]["mean_test"]
    print(f"\n  Train-test ROC-AUC gap: {roc_gap:.4f}")
    if roc_gap > 0.05:
        print("  ⚠ Gap > 0.05 — possible overfitting. Consider reducing max_depth.")
    else:
        print("  ✓ Gap ≤ 0.05 — model generalises well.")

    return summary


def train_final_model(model: XGBClassifier, X_train, y_train) -> XGBClassifier:
    """Train the final model on the full training set."""
    print("\nTraining final model on full training set...")
    model.fit(X_train, y_train)
    print("✓ Training complete.")
    return model


def evaluate_on_test_set(model: XGBClassifier, X_test, y_test) -> dict:
    """Evaluate the trained model on the held-out test set."""
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_pred_prob)
    f1 = f1_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("TEST SET EVALUATION (held-out, never seen during training)")
    print("=" * 60)
    print(f"\nROC-AUC : {roc_auc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Susceptible (0)", "Resistant (1)"]
    ))

    results = {
        "roc_auc": round(roc_auc, 4),
        "f1": round(f1, 4),
    }
    return results, y_pred, y_pred_prob


def plot_confusion_matrix(y_test, y_pred) -> None:
    """Plot and save the confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Susceptible", "Resistant"],
        yticklabels=["Susceptible", "Resistant"],
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(
        "Confusion Matrix — XGBoost AMR Classifier\n"
        "Acinetobacter baumannii Aminoglycoside Resistance",
        fontsize=12, fontweight="bold", pad=12,
    )
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "04_confusion_matrix.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {path}")


def plot_roc_curve(y_test, y_pred_prob) -> None:
    """Plot and save the ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc = roc_auc_score(y_test, y_pred_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="steelblue", lw=2,
            label=f"XGBoost (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1,
            linestyle="--", label="Random classifier (AUC = 0.50)")
    ax.fill_between(fpr, tpr, alpha=0.1, color="steelblue")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title(
        "ROC Curve — XGBoost AMR Classifier\n"
        "Acinetobacter baumannii Aminoglycoside Resistance",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "05_roc_curve.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {path}")


def save_results(cv_summary: dict, test_results: dict, model: XGBClassifier) -> None:
    """Save all evaluation results and model for reproducibility."""
    results = {
        "cross_validation": cv_summary,
        "test_set": test_results,
        "config": {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "n_folds": N_FOLDS,
            "model_params": model.get_params(),
        }
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {RESULTS_FILE}")

    # Save the model
    joblib.dump(model, MODEL_FILE)
    print(f"Model saved → {MODEL_FILE}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("XGBoost AMR Resistance Classifier — Training Pipeline")
    print("=" * 60)

    # Step 1: Load data
    X, y = load_features(INPUT_FILE)

    # Step 2: Compute class weight
    scale_pos_weight = compute_class_weight(y)

    # Step 3: Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 4: Build model
    model = build_model(scale_pos_weight)
    print(f"\nModel configuration:")
    print(f"  {model.get_params()}")

    # Step 5: Cross-validation
    cv_summary = run_cross_validation(model, X_train, y_train)

    # Step 6: Train final model
    model = train_final_model(model, X_train, y_train)

    # Step 7: Evaluate on test set
    test_results, y_pred, y_pred_prob = evaluate_on_test_set(model, X_test, y_test)

    # Step 8: Plots
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_pred_prob)

    # Step 9: Save results
    save_results(cv_summary, test_results, model)

    print("\n Training pipeline complete.")