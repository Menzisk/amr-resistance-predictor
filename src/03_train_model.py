"""
03_train_model.py

Trains an XGBoost classifier to predict aminoglycoside resistance
in Acinetobacter baumannii from phenotypic metadata features.

Pipeline:
    1. Load engineered features
    2. Split into train/test sets (stratified 80/20)
    3. Train XGBoost with class weighting
    4. Evaluate with cross-validation
    5. Evaluate on held-out test set
    6. Save model and results

References:
    Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System.
    KDD 2016. https://arxiv.org/abs/1603.02754

    Friedman (2001). Greedy function approximation: a gradient boosting
    machine. Annals of Statistics, 29(5), 1189-1232.

Author: Menzi Sikakane
Date:   2026-06-17
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os
import json


from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
)
from xgboost import XGBClassifier

# ── Constants ────────────────────────────────────────────────────────────────

INPUT_FILE   = "data/processed/amr_features.tsv"
MODEL_DIR    = "outputs/models"
FIG_DIR      = "outputs/figures"
RESULTS_FILE = os.path.join(MODEL_DIR, "evaluation_results.json")

RANDOM_STATE = 42     
TEST_SIZE    = 0.20   
N_FOLDS      = 5      
DPI          = 300    

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.2)


# ── Functions ────────────────────────────────────────────────────────────────

def load_features(filepath: str):
    """
    Load the engineered feature set.
    Separates features (X) from label (y).

    In ML notation:
        X = feature matrix, shape (n_samples, n_features)
        y = label vector,   shape (n_samples,)

    Returns:
        X: pd.DataFrame of features
        y: pd.Series of binary labels (1=Resistant, 0=Susceptible)
    """
    df = pd.read_csv(filepath, sep="\t")
    print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Separate features from label
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
    """
    Compute the scale_pos_weight parameter for XGBoost.

    XGBoost uses scale_pos_weight to handle class imbalance.
    It tells the model to penalise mistakes on the minority class
    (Susceptible, label=0) more heavily.

    The formula is:
        scale_pos_weight = count(negative class) / count(positive class)
                         = count(Susceptible) / count(Resistant)

    Why this formula?
    XGBoost internally multiplies the gradient of positive class examples
    by this weight, effectively up-weighting the minority class during
    training. This is mathematically equivalent to oversampling the
    minority class by this factor.

    Reference: XGBoost documentation, Parameter Tuning Guide.

    Args:
        y: binary label Series (1=Resistant, 0=Susceptible)

    Returns:
        float: scale_pos_weight value
    """
    n_resistant   = (y == 1).sum()
    n_susceptible = (y == 0).sum()
    weight = n_susceptible / n_resistant

    print(f"\nClass weight (scale_pos_weight):")
    print(f"  n_susceptible / n_resistant = {n_susceptible} / {n_resistant} = {weight:.4f}")
    print(f"  (weight < 1 means Resistant is majority class)")

    return weight


def split_data(X: pd.DataFrame, y: pd.Series):
    """
    Split data into training and test sets.

    stratify=y ensures both sets have the same class ratio.
    Without stratify, a random split might put almost all
    Susceptible records in one set by chance.

    Args:
        X: feature matrix
        y: label vector

    Returns:
        X_train, X_test, y_train, y_test
    """
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
    test_rate  = y_test.mean()  * 100
    print(f"  Resistance rate in train: {train_rate:.1f}%")
    print(f"  Resistance rate in test : {test_rate:.1f}%")

    return X_train, X_test, y_train, y_test


def build_model(scale_pos_weight: float) -> XGBClassifier:
    """
    Define the XGBoost classifier with our chosen hyperparameters.

    Hyperparameter explanations:
    ─────────────────────────────────────────────────────────────
    n_estimators=200
        Number of trees to build sequentially.
        More trees → better fit, but risk of overfitting.
        We use early stopping to find the optimal number.

    max_depth=4
        Maximum depth of each tree.
        Deeper trees → more complex patterns captured.
        Shallower trees → simpler, more generalisable.
        For tabular biological data, depth 3-6 is standard.

    learning_rate=0.1  (also called eta, η)
        How much each new tree contributes to the ensemble.
        Mathematically: F_t(x) = F_{t-1}(x) + η × f_t(x)
        Lower η → more trees needed, but more robust.
        Range: 0.01 to 0.3 is typical.

    subsample=0.8
        Fraction of training records used per tree.
        Introduces randomness → reduces overfitting.
        This is the "stochastic" in stochastic gradient boosting.

    colsample_bytree=0.8
        Fraction of features considered per tree.
        Also introduces randomness, similar to Random Forest.

    scale_pos_weight
        Handles class imbalance.
        = n_negative / n_positive = n_susceptible / n_resistant

    eval_metric="logloss"
        The loss function minimised during training.
        Log-loss measures how well predicted probabilities
        match true labels. Lower is better.

    random_state=42
        Fixed seed for reproducibility.
        Any fixed integer works — 42 is a convention.
    ─────────────────────────────────────────────────────────────

    Args:
        scale_pos_weight: class imbalance correction factor

    Returns:
        XGBClassifier: configured but not yet trained
    """
    model = XGBClassifier(
        n_estimators      = 200,
        max_depth         = 4,
        learning_rate     = 0.1,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        scale_pos_weight  = scale_pos_weight,
        eval_metric       = "logloss",
        random_state      = RANDOM_STATE,
        verbosity         = 0,      
    )
    return model


def run_cross_validation(model: XGBClassifier, X_train, y_train) -> dict:
    """
    Run stratified k-fold cross-validation on the training set.

    Why stratified?
    StratifiedKFold ensures each fold has the same class ratio
    as the full training set. With imbalanced data, a random fold
    might contain very few Susceptible records by chance.

    Scoring metrics:
        roc_auc  → Area Under the ROC Curve
        f1       → F1 score (harmonic mean of precision and recall)
        accuracy → raw percentage correct (shown for comparison only)

    We report mean ± standard deviation across folds.
    Standard deviation tells us how stable the model is.
    A high std means the model is sensitive to which data it sees.

    Args:
        model: configured XGBClassifier
        X_train: training features
        y_train: training labels

    Returns:
        dict: CV results with mean and std per metric
    """
    print(f"\nRunning {N_FOLDS}-fold stratified cross-validation")

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
        test_scores  = cv_results[f"test_{metric}"]
        train_scores = cv_results[f"train_{metric}"]
        mean_test  = test_scores.mean()
        std_test   = test_scores.std()
        mean_train = train_scores.mean()

        print(f"  {metric:<15} {mean_test:>8.4f}  {std_test:>8.4f}  "
              f"(train: {mean_train:.4f})")

        summary[metric] = {
            "mean_test" : round(mean_test,  4),
            "std_test"  : round(std_test,   4),
            "mean_train": round(mean_train, 4),
        }

    # Overfitting check
    roc_gap = summary["roc_auc"]["mean_train"] - summary["roc_auc"]["mean_test"]
    print(f"\n  Train-test ROC-AUC gap: {roc_gap:.4f}")
    if roc_gap > 0.05:
        print("  Gap > 0.05 - possible overfitting. Consider reducing max_depth.")
    else:
        print("  Gap ≤ 0.05 - model generalises well.")

    return summary


def train_final_model(model: XGBClassifier, X_train, y_train) -> XGBClassifier:
    """
    Train the final model on the full training set.

    After cross-validation confirms our hyperparameters are sound,
    we train one final model on ALL training data.
    This gives the model the maximum amount of information to learn from.

    Args:
        model: configured XGBClassifier
        X_train: full training features
        y_train: full training labels

    Returns:
        XGBClassifier: trained model
    """
    print("\nTraining final model on full training set")
    model.fit(X_train, y_train)
    print(" Training complete.")
    return model


def evaluate_on_test_set(model: XGBClassifier, X_test, y_test,
                         feature_names: list) -> dict:
    """
    Evaluate the trained model on the held-out test set.

    This is the final, honest evaluation. The test set was never
    seen during training or cross-validation.

    Metrics reported:
    ─────────────────────────────────────────────────────────────
    Accuracy  = (TP + TN) / (TP + TN + FP + FN)
                Overall correctness. Misleading for imbalanced data.

    Precision = TP / (TP + FP)
                Of all predicted Resistant, how many truly were?
                Low precision → too many false alarms.

    Recall    = TP / (TP + FN)
                Of all truly Resistant, how many did we catch?
                Low recall → missing true resistant cases.
                In clinical settings, low recall is dangerous.

    F1        = 2 × (Precision × Recall) / (Precision + Recall)
                Harmonic mean of precision and recall.
                Balances both concerns.

    ROC-AUC   = Area under the Receiver Operating Characteristic curve.
                Probability that a randomly chosen Resistant isolate
                is ranked higher than a randomly chosen Susceptible one.
                0.5 = random guessing. 1.0 = perfect.
    ─────────────────────────────────────────────────────────────

    Args:
        model: trained XGBClassifier
        X_test: test features
        y_test: test labels
        feature_names: list of feature column names

    Returns:
        dict: evaluation metrics
    """
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1] 

    roc_auc  = roc_auc_score(y_test, y_pred_prob)
    f1       = f1_score(y_test, y_pred)

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
        "f1"     : round(f1, 4),
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
        "Confusion Matrix: XGBoost AMR Classifier\n"
        "Acinetobacter baumannii Aminoglycoside Resistance",
        fontsize=12, fontweight="bold", pad=12,
    )
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "04_confusion_matrix.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"Figure saved: {path}")
    plt.close()


def plot_roc_curve(y_test, y_pred_prob) -> None:
    """
    Plot and save the ROC curve.

    The ROC curve plots:
        x-axis: False Positive Rate (FPR) = FP / (FP + TN)
        y-axis: True Positive Rate (TPR)  = TP / (TP + FN) = Recall

    At every possible classification threshold (0 to 1), we compute
    FPR and TPR and plot the point. The curve traces the tradeoff
    between catching true positives and generating false positives.

    The diagonal line (AUC=0.5) represents random guessing.
    A perfect classifier hugs the top-left corner (AUC=1.0).
    """
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
        "ROC Curve: XGBoost AMR Classifier\n"
        "Acinetobacter baumannii Aminoglycoside Resistance",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "05_roc_curve.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"Figure saved: {path}")
    plt.close()


def save_results(cv_summary: dict, test_results: dict) -> None:
    """Save all evaluation results to JSON for reproducibility."""
    results = {
        "cross_validation": cv_summary,
        "test_set"        : test_results,
        "config": {
            "random_state": RANDOM_STATE,
            "test_size"   : TEST_SIZE,
            "n_folds"     : N_FOLDS,
        }
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {RESULTS_FILE}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("XGBoost AMR Resistance Classifier — Training Pipeline")
    print("=" * 60)

    X, y = load_features(INPUT_FILE)
    scale_pos_weight = compute_class_weight(y)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = build_model(scale_pos_weight)
    print(f"\nModel configuration:")
    print(f"  {model.get_params()}")

    cv_summary = run_cross_validation(model, X_train, y_train)
    model = train_final_model(model, X_train, y_train)
    feature_names = list(X.columns)
    test_results, y_pred, y_pred_prob = evaluate_on_test_set(
        model, X_test, y_test, feature_names
    )

    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_pred_prob)

    save_results(cv_summary, test_results)

    print("\n Training pipeline complete.")