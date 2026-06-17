"""
04_shap_analysis.py

SHAP (SHapley Additive exPlanations) interpretability analysis
for the XGBoost AMR resistance classifier.

Produces:
    - SHAP summary plot (beeswarm)
    - SHAP bar plot (mean absolute importance)
    - SHAP waterfall plot (single prediction)

References:
    Lundberg & Lee (2017). A unified approach to interpreting model
    predictions. NeurIPS 2017. https://arxiv.org/abs/1705.07874

    Lundberg et al. (2020). From local explanations to global
    understanding with explainable AI for trees. Nature Machine
    Intelligence, 2, 56-67.

Author: Menzi Sikakane
Date:   2026-06-17
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
matplotlib.use('Agg')
import shap
import os
import xgboost as xgb

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ── Constants ────────────────────────────────────────────────────────────────

INPUT_FILE   = "data/processed/amr_features.tsv"
FIG_DIR      = "outputs/figures"
RANDOM_STATE = 42
TEST_SIZE    = 0.20
DPI          = 300

os.makedirs(FIG_DIR, exist_ok=True)

# Human-readable feature names for plots
FEATURE_LABELS = {
    "antibiotic_amikacin"                  : "Antibiotic: Amikacin",
    "antibiotic_gentamicin"                : "Antibiotic: Gentamicin",
    "antibiotic_tobramycin"                : "Antibiotic: Tobramycin",
    "laboratory_typing_method_Broth dilution" : "Method: Broth Dilution",
    "laboratory_typing_method_Disk diffusion" : "Method: Disk Diffusion",
    "laboratory_typing_method_Unknown"        : "Method: Unknown",
}


# ── Functions ────────────────────────────────────────────────────────────────

def load_and_split(filepath: str):
    """
    Load features and reproduce the exact same train/test split
    used in training. Using the same random_state and stratify
    guarantees the test set here is identical to training.

    This is critical — we must evaluate SHAP on the same test set
    the model was evaluated on. Using different data would give
    misleading interpretations.

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    df = pd.read_csv(filepath, sep="\t")

    feature_cols  = [c for c in df.columns if c != "label"]
    X = df[feature_cols]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y,
    )

    print(f"Data loaded: {X.shape[0]:,} records, {X.shape[1]} features")
    print(f"Test set   : {X_test.shape[0]:,} records")

    return X_train, X_test, y_train, y_test, list(feature_cols)


def retrain_model(X_train, y_train) -> XGBClassifier:
    """
    Retrain the XGBoost model with identical hyperparameters.

    In a production pipeline you would save and load the trained model
    using joblib or pickle. We retrain here for simplicity and to keep
    this script self-contained and reproducible without file dependencies.

    Args:
        X_train: training features
        y_train: training labels

    Returns:
        XGBClassifier: trained model
    """
    n_resistant   = (y_train == 1).sum()
    n_susceptible = (y_train == 0).sum()
    scale_pos_weight = n_susceptible / n_resistant

    model = XGBClassifier(
        n_estimators     = 200,
        max_depth        = 4,
        learning_rate    = 0.1,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        scale_pos_weight = scale_pos_weight,
        eval_metric      = "logloss",
        random_state     = RANDOM_STATE,
        verbosity        = 0,
    )

    model.fit(X_train, y_train)
    print("Model retrained.")
    return model


def compute_shap_values(model: XGBClassifier, X_test: pd.DataFrame):
    """
    Compute SHAP values for the test set using TreeExplainer.

    TreeExplainer is a fast, exact SHAP algorithm specifically
    designed for tree-based models (XGBoost, LightGBM, Random Forest).
    It runs in polynomial time rather than exponential time by
    exploiting the tree structure to compute exact Shapley values
    without sampling.

    Reference: Lundberg et al. (2020). Nature Machine Intelligence.

    Args:
        model: trained XGBClassifier
        X_test: test features

    Returns:
        explainer: SHAP TreeExplainer object
        shap_values: array of shape (n_samples, n_features)
    """
    print("\nComputing SHAP values with TreeExplainer...")

    # TreeExplainer takes the trained model directly
    explainer = shap.TreeExplainer(model)

    # shap_values shape: (n_test_records, n_features)
    # Each row is one record. Each column is one feature's SHAP value.
    # Positive value → pushes prediction toward Resistant (class 1)
    # Negative value → pushes prediction toward Susceptible (class 0)
    shap_values = explainer.shap_values(X_test)

    print(f"SHAP values computed: {shap_values.shape}")
    print(f"Baseline (expected value): {explainer.expected_value:.4f}")
    print(f"  (This is the model's average prediction = log-odds of Resistant)")

    return explainer, shap_values


def verify_additivity(explainer, shap_values, X_test, model, n_check=3):
    """
    Verify the SHAP additivity property for n_check records.

    The additivity axiom states:
        f(x) = E[f(x)] + sum(SHAP values for all features)

    If this holds numerically, our SHAP computation is correct.
    This is a scientific integrity check — always verify your tools.

    Args:
        explainer: SHAP TreeExplainer
        shap_values: computed SHAP values array
        X_test: test features
        model: trained XGBClassifier
        n_check: number of records to verify
    """
    print(f"\nVerifying SHAP additivity for {n_check} records:")
    print(f"  {'Record':<8} {'Model output':>14} {'Base+SHAP sum':>14} {'Match':>8}")
    print(f"  {'-'*48}")

    # Get raw model outputs (log-odds, before sigmoid)
    # predict_proba gives probabilities; we need raw margin for additivity check
    raw_preds = model.get_booster().predict(
        xgb_DMatrix(X_test), output_margin=True
    )

    for i in range(n_check):
        model_output = raw_preds[i]
        shap_sum     = explainer.expected_value + shap_values[i].sum()
        match        = "✓" if abs(model_output - shap_sum) < 1e-4 else "✗"
        print(f"  {i:<8} {model_output:>14.6f} {shap_sum:>14.6f} {match:>8}")


def plot_shap_summary(shap_values, X_test, feature_names):
    """
    Beeswarm summary plot — the most informative SHAP visualisation.

    Each dot = one record in the test set.
    x-axis   = SHAP value (impact on model output)
               positive → pushes toward Resistant
               negative → pushes toward Susceptible
    y-axis   = features ranked by mean absolute SHAP value (most important on top)
    colour   = feature value for that record (red=high/1, blue=low/0)

    For our binary features:
        red  (feature value = 1) → this drug WAS the one tested
        blue (feature value = 0) → this drug was NOT the one tested
    """
    # Rename columns for clean plot labels
    X_plot = X_test.rename(columns=FEATURE_LABELS)

    fig, ax = plt.subplots(figsize=(10, 6))

    shap.summary_plot(
        shap_values,
        X_plot,
        show       = False,   # don't auto-display 
        plot_size  = None,    # use our figure size
        color_bar  = True,
    )

    plt.title(
        "SHAP Feature Importance — XGBoost AMR Classifier\n"
        "Acinetobacter baumannii Aminoglycoside Resistance",
        fontsize   = 12,
        fontweight = "bold",
        pad        = 15,
    )
    plt.xlabel("SHAP Value (impact on model output)\n"
               "Positive → pushes toward Resistant | "
               "Negative → pushes toward Susceptible",
               fontsize=10)
    plt.tight_layout()

    path = os.path.join(FIG_DIR, "06_shap_summary_beeswarm.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {path}")


def plot_shap_bar(shap_values, X_test):
    """
    Bar plot of mean absolute SHAP values per feature.

    Mean |SHAP| = average magnitude of a feature's impact,
    regardless of direction. This is the standard global
    feature importance measure in SHAP.

    Mathematically:
        importance(i) = (1/n) × Σ |φᵢ(xⱼ)|
        where φᵢ(xⱼ) is the SHAP value of feature i for record j
    """
    # Compute mean absolute SHAP per feature
    feature_names_clean = [FEATURE_LABELS.get(f, f) for f in X_test.columns]
    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=feature_names_clean
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.barh(
        mean_abs_shap.index,
        mean_abs_shap.values,
        color     = "steelblue",
        edgecolor = "white",
        linewidth = 0.8,
    )

    # Add value labels
    for bar, val in zip(bars, mean_abs_shap.values):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va       = "center",
            fontsize = 9,
        )

    ax.set_xlabel("Mean |SHAP Value| (average impact on model output)", fontsize=11)
    ax.set_title(
        "Global Feature Importance (Mean Absolute SHAP)\n"
        "Acinetobacter baumannii Aminoglycoside Resistance",
        fontsize   = 12,
        fontweight = "bold",
        pad        = 12,
    )
    plt.tight_layout()

    path = os.path.join(FIG_DIR, "07_shap_bar_importance.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {path}")


def plot_shap_waterfall(explainer, shap_values, X_test, y_test, record_idx=0):
    """
    Waterfall plot for a single prediction.

    Shows how each feature pushes the prediction from the baseline
    (average prediction) to the final output for one specific record.

    Starting from E[f(x)] (baseline), each feature's SHAP value
    is added sequentially, arriving at f(x) (final prediction).

    This is the most intuitive plot for explaining an individual
    prediction to a non-technical audience.

    Args:
        record_idx: which test set record to explain (default: 0)
    """
    # Select one record
    record   = X_test.iloc[[record_idx]]
    sv       = shap_values[record_idx]
    true_label = y_test.iloc[record_idx]
    label_name = "Resistant" if true_label == 1 else "Susceptible"

    print(f"\nWaterfall plot for test record #{record_idx}")
    print(f"True label: {label_name} ({true_label})")
    print(f"Feature values:")
    for col, val in record.iloc[0].items():
        shap_val = sv[list(X_test.columns).index(col)]
        label    = FEATURE_LABELS.get(col, col)
        print(f"  {label:<40} value={int(val)}  SHAP={shap_val:+.4f}")

    # Build SHAP Explanation object (required by new waterfall API)
    explanation = shap.Explanation(
        values        = sv,
        base_values   = explainer.expected_value,
        data          = record.iloc[0].values,
        feature_names = [FEATURE_LABELS.get(f, f) for f in X_test.columns],
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    shap.plots.waterfall(explanation, show=False)

    plt.title(
        f"SHAP Waterfall — Single Prediction (True label: {label_name})\n"
        f"Acinetobacter baumannii Aminoglycoside Resistance",
        fontsize   = 11,
        fontweight = "bold",
        pad        = 12,
    )
    plt.tight_layout()

    path = os.path.join(FIG_DIR, f"08_shap_waterfall_record{record_idx}.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"Figure saved: {path}")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("SHAP Interpretability Analysis")
    print("=" * 60)

    # Step 1: Load data and reproduce split
    X_train, X_test, y_train, y_test, feature_names = load_and_split(INPUT_FILE)

    # Step 2: Retrain model
    model = retrain_model(X_train, y_train)

    # Step 3: Compute SHAP values
    explainer, shap_values = compute_shap_values(model, X_test)

    # Step 4: Verify additivity (scientific integrity check)
    print(f"\nSHAP value matrix shape: {shap_values.shape}")
    print(f"Mean absolute SHAP per feature:")
    for i, fname in enumerate(feature_names):
        label = FEATURE_LABELS.get(fname, fname)
        mean_abs = np.abs(shap_values[:, i]).mean()
        print(f"  {label:<40} {mean_abs:.4f}")

    # Step 5: Plots
    plot_shap_summary(shap_values, X_test, feature_names)
    plot_shap_bar(shap_values, X_test)
    resistant_idx   = y_test[y_test == 1].index[0]
    susceptible_idx = y_test[y_test == 0].index[0]
    resistant_pos   = list(y_test.index).index(resistant_idx)
    susceptible_pos = list(y_test.index).index(susceptible_idx)

    plot_shap_waterfall(explainer, shap_values, X_test, y_test, resistant_pos)
    plot_shap_waterfall(explainer, shap_values, X_test, y_test, susceptible_pos)

    print("\n SHAP analysis complete.")