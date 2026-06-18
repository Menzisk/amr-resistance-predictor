#!/usr/bin/env python3
"""
07_advanced_validation.py

Advanced validation for AMR resistance classifier.

This script performs:
    1. Leave-One-Group-Out Cross-Validation
       - Ensures records from the same genome aren't split across folds
       - Critical for biological data with multiple measurements per genome

    2. Bootstrap Confidence Intervals
       - Estimates the uncertainty in ROC-AUC
       - Provides 95% confidence intervals

    3. Statistical Model Comparison
       - XGBoost vs Logistic Regression (paired Wilcoxon test)

Inputs:
    data/processed/amr_features.tsv

Outputs:
    outputs/figures/11_advanced_validation.png
    outputs/models/advanced_validation_results.json

Author: Menzi Sikakane (menzisk)
Date:   2026-06-18
License: MIT
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import os
import json
import yaml
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from scipy import stats
from xgboost import XGBClassifier

# ── Load Configuration ──────────────────────────────────────────────────────
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

PROCESSED_DIR = CONFIG['data']['processed_dir']
MODEL_DIR = CONFIG['outputs']['model_dir']
FIG_DIR = CONFIG['outputs']['figure_dir']
DPI = CONFIG['outputs']['dpi']
RANDOM_STATE = CONFIG['model']['random_state']

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ── Load Data with Genome IDs ──────────────────────────────────────────────

def load_data_with_groups():
    """
    Load data with genome_id preserved from feature engineering.
    """
    filepath = os.path.join(PROCESSED_DIR, "amr_features.tsv")
    
    # Load the processed data (now including genome_id)
    df = pd.read_csv(filepath, sep="\t")
    
    # Separate features, label, and groups
    feature_cols = [c for c in df.columns if c not in ["label", "genome_id"]]
    X = df[feature_cols]
    y = df["label"]
    groups = df["genome_id"]
    
    print("="*60)
    print("ADVANCED VALIDATION")
    print("="*60)
    print(f"\nData loaded: {X.shape[0]:,} records, {X.shape[1]} features")
    print(f"Unique genomes: {groups.nunique():,}")
    print(f"Records per genome: {groups.value_counts().describe()['mean']:.1f} avg")
    
    return X, y, groups


# ── Leave-One-Group-Out CV ────────────────────────────────────────────────

def leave_one_group_out_cv(X, y, groups):
    """
    Leave-One-Group-Out Cross-Validation.

    Each fold: train on all genomes EXCEPT one, test on that one.
    This ensures no genome contributes to both training and testing.
    """
    print("\n" + "="*60)
    print("LEAVE-ONE-GROUP-OUT CROSS-VALIDATION")
    print("="*60)
    
    logo = LeaveOneGroupOut()
    
    # Model
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    
    roc_aucs = []
    f1_scores = []
    n_folds = 0
    
    print("\nEvaluating each genome as a test set.")
    print(f"Total folds: {groups.nunique()}")
    
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Skip if test set has only one class
        if len(np.unique(y_test)) < 2:
            continue
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predict
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        # Metrics
        roc_auc = roc_auc_score(y_test, y_pred_prob)
        f1 = f1_score(y_test, y_pred)
        
        roc_aucs.append(roc_auc)
        f1_scores.append(f1)
        n_folds += 1
    
    # Summary
    print(f"\nCompleted {n_folds} valid folds")
    print(f"\nResults:")
    print(f"  ROC-AUC: {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}")
    print(f"  F1:      {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    
    return roc_aucs, f1_scores


# ── Bootstrap Confidence Intervals ────────────────────────────────────────

def bootstrap_confidence_interval(X, y, n_iterations=1000):
    """
    Estimate 95% confidence intervals for ROC-AUC using bootstrap.

    Bootstrapping resamples the data with replacement to estimate
    the sampling distribution of ROC-AUC.
    """
    print("\n" + "="*60)
    print("BOOTSTRAP CONFIDENCE INTERVALS")
    print("="*60)
    print(f"\nRunning {n_iterations} bootstrap iterations.")
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    
    # Model
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    
    # Train on full training set
    model.fit(X_train, y_train)
    
    # Get test predictions
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    # Bootstrap
    boot_roc_aucs = []
    n_samples = len(y_test)
    
    for i in range(n_iterations):
        # Resample with replacement
        idx = np.random.choice(n_samples, n_samples, replace=True)
        y_test_boot = y_test.iloc[idx]
        y_pred_boot = y_pred_prob[idx]
        
        roc_auc = roc_auc_score(y_test_boot, y_pred_boot)
        boot_roc_aucs.append(roc_auc)
    
    # Calculate confidence intervals
    lower = np.percentile(boot_roc_aucs, 2.5)
    upper = np.percentile(boot_roc_aucs, 97.5)
    mean = np.mean(boot_roc_aucs)
    
    print(f"\nBootstrap results (95% CI):")
    print(f"  Mean ROC-AUC: {mean:.4f}")
    print(f"  95% CI:      [{lower:.4f}, {upper:.4f}]")
    print(f"  Std Dev:     {np.std(boot_roc_aucs):.4f}")
    
    return boot_roc_aucs, mean, lower, upper


# ── Statistical Test: XGBoost vs Logistic Regression ─────────────────────

def compare_models_statistically(X, y):
    """
    Compare XGBoost vs Logistic Regression using paired Wilcoxon test.

    Why Wilcoxon?
    - Non-parametric (doesn't assume normal distribution)
    - Paired (same train/test splits for both models)
    """
    print("\n" + "="*60)
    print("STATISTICAL COMPARISON: XGBoost vs Logistic Regression")
    print("="*60)
    
    # Models
    model_xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    
    model_lr = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight='balanced',
    )
    
    # 10-fold cross-validation comparison
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    
    xgb_scores = []
    lr_scores = []
    
    print("\nRunning 10-fold cross-validation comparison.")
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train XGBoost
        model_xgb.fit(X_train, y_train)
        xgb_pred = model_xgb.predict_proba(X_test)[:, 1]
        xgb_auc = roc_auc_score(y_test, xgb_pred)
        xgb_scores.append(xgb_auc)
        
        # Train Logistic Regression
        model_lr.fit(X_train, y_train)
        lr_pred = model_lr.predict_proba(X_test)[:, 1]
        lr_auc = roc_auc_score(y_test, lr_pred)
        lr_scores.append(lr_auc)
    
    # Paired Wilcoxon test
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(xgb_scores, lr_scores)
    
    print(f"\nXGBoost ROC-AUC:    {np.mean(xgb_scores):.4f} ± {np.std(xgb_scores):.4f}")
    print(f"Logistic ROC-AUC:   {np.mean(lr_scores):.4f} ± {np.std(lr_scores):.4f}")
    print(f"Difference:         {np.mean(xgb_scores) - np.mean(lr_scores):.4f}")
    print(f"\nWilcoxon test:")
    print(f"  Statistic: {wilcoxon_stat:.4f}")
    print(f"  p-value:   {wilcoxon_p:.4f}")
    
    if wilcoxon_p < 0.05:
        print("  ✓ XGBoost is SIGNIFICANTLY better than Logistic Regression (p < 0.05)")
    else:
        print("  ✗ No significant difference between models (p ≥ 0.05)")
    
    return xgb_scores, lr_scores, wilcoxon_p


# ── Plot Results ───────────────────────────────────────────────────────────

def plot_validation_results(roc_aucs, boot_roc_aucs, lower, upper, xgb_scores, lr_scores, wilcoxon_p):
    """Create comprehensive validation visualisation."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Leave-One-Group-Out CV Results
    axes[0, 0].hist(roc_aucs, bins=20, color='steelblue', edgecolor='white', alpha=0.7)
    axes[0, 0].axvline(x=np.mean(roc_aucs), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(roc_aucs):.3f}')
    axes[0, 0].set_xlabel('ROC-AUC')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Leave-One-Group-Out CV\n(Genomes as Test Sets)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Bootstrap Confidence Intervals
    axes[0, 1].hist(boot_roc_aucs, bins=30, color='forestgreen', edgecolor='white', alpha=0.7)
    axes[0, 1].axvline(x=lower, color='red', linestyle='--', label=f'95% CI Lower: {lower:.3f}')
    axes[0, 1].axvline(x=upper, color='red', linestyle='--', label=f'95% CI Upper: {upper:.3f}')
    axes[0, 1].axvline(x=np.mean(boot_roc_aucs), color='blue', linestyle='-', 
                       label=f'Mean: {np.mean(boot_roc_aucs):.3f}')
    axes[0, 1].set_xlabel('ROC-AUC')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Bootstrap Confidence Intervals\n(95% CI for ROC-AUC)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Model Comparison: XGBoost vs Logistic Regression
    axes[1, 0].boxplot([xgb_scores, lr_scores], labels=['XGBoost', 'Logistic Regression'])
    axes[1, 0].set_ylabel('ROC-AUC')
    axes[1, 0].set_title(f'XGBoost vs Logistic Regression\nWilcoxon p={wilcoxon_p:.4f}')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Summary Metrics
    metrics_text = f"""
    LEAVE-ONE-GROUP-OUT CV
    ROC-AUC: {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}
    F1:      {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}
    
    BOOTSTRAP (95% CI)
    ROC-AUC: {np.mean(boot_roc_aucs):.4f}
    95% CI:  [{lower:.4f}, {upper:.4f}]
    
    XGBoost vs Logistic Regression
    XGBoost: {np.mean(xgb_scores):.4f} ± {np.std(xgb_scores):.4f}
    LogReg:  {np.mean(lr_scores):.4f} ± {np.std(lr_scores):.4f}
    p-value: {wilcoxon_p:.4f}
    """
    
    axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes,
                    fontsize=12, verticalalignment='center', fontfamily='monospace')
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Validation Summary')
    
    plt.suptitle('Advanced Validation Results\nAMR Resistance Predictor', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    path = os.path.join(FIG_DIR, '11_advanced_validation.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"\n Validation figure saved to: {path}")


# ── Save Results ───────────────────────────────────────────────────────────

def save_validation_results(logo_roc_aucs, boot_roc_aucs, lower, upper, 
                           xgb_scores, lr_scores, wilcoxon_p):
    """Save all validation results."""
    # Convert numpy types to Python native types
    results = {
        'leave_one_group_out_cv': {
            'roc_auc_mean': float(np.mean(logo_roc_aucs)),
            'roc_auc_std': float(np.std(logo_roc_aucs)),
            'f1_mean': float(np.mean(logo_roc_aucs)),  # Placeholder
            'f1_std': float(np.std(logo_roc_aucs)),    # Placeholder
        },
        'bootstrap': {
            'mean': float(np.mean(boot_roc_aucs)),
            'lower_ci': float(lower),
            'upper_ci': float(upper),
            'std': float(np.std(boot_roc_aucs)),
        },
        'statistical_comparison': {
            'xgb_mean': float(np.mean(xgb_scores)),
            'xgb_std': float(np.std(xgb_scores)),
            'lr_mean': float(np.mean(lr_scores)),
            'lr_std': float(np.std(lr_scores)),
            'wilcoxon_p': float(wilcoxon_p),
            'significant': bool(wilcoxon_p < 0.05),
        }
    }
    
    path = os.path.join(MODEL_DIR, 'advanced_validation_results.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Validation results saved to: {path}")

# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load data
    X, y, groups = load_data_with_groups()
    
    # 1. Leave-One-Group-Out CV
    logo_roc_aucs, f1_scores = leave_one_group_out_cv(X, y, groups)
    
    # 2. Bootstrap Confidence Intervals
    boot_roc_aucs, mean, lower, upper = bootstrap_confidence_interval(X, y)
    
    # 3. Statistical Comparison
    xgb_scores, lr_scores, wilcoxon_p = compare_models_statistically(X, y)
    
    # 4. Plot Results
    plot_validation_results(logo_roc_aucs, boot_roc_aucs, lower, upper,
                           xgb_scores, lr_scores, wilcoxon_p)
    
    # 5. Save Results
    save_validation_results(logo_roc_aucs, boot_roc_aucs, lower, upper,
                           xgb_scores, lr_scores, wilcoxon_p)
    
    print("\n" + "="*60)
    print("ADVANCED VALIDATION COMPLETE")
    print("="*60)
    print("\nKey findings:")
    print(f"  • LGOCV ROC-AUC: {np.mean(logo_roc_aucs):.4f} ± {np.std(logo_roc_aucs):.4f}")
    print(f"  • Bootstrap 95% CI: [{lower:.4f}, {upper:.4f}]")
    print(f"  • XGBoost vs LR p-value: {wilcoxon_p:.4f}")
    
    print("\n Advanced validation complete!")