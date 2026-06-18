#!/usr/bin/env python3
"""
06_model_comparison.py

Compare XGBoost against other models for AMR resistance prediction.

Models evaluated:
    1. XGBoost (gradient boosting)
    2. Random Forest (bagging)
    3. Logistic Regression (linear baseline)
    4. LightGBM (alternative gradient boosting)

Metrics:
    - ROC-AUC
    - F1 Score
    - Training Time

Inputs:
    data/processed/amr_features.tsv

Outputs:
    outputs/figures/10_model_comparison.png
    outputs/models/comparison_results.json

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
import time
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from xgboost import XGBClassifier

# Try importing LightGBM (optional)
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("LightGBM not installed. Skipping.")

# ── Load Configuration ──────────────────────────────────────────────────────
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

PROCESSED_DIR = CONFIG['data']['processed_dir']
MODEL_DIR = CONFIG['outputs']['model_dir']
FIG_DIR = CONFIG['outputs']['figure_dir']
DPI = CONFIG['outputs']['dpi']
RANDOM_STATE = CONFIG['model']['random_state']
TEST_SIZE = CONFIG['model']['test_size']

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ── Load Data ──────────────────────────────────────────────────────────────

def load_data():
    """Load features and split into train/test."""
    filepath = os.path.join(PROCESSED_DIR, "amr_features.tsv")
    df = pd.read_csv(filepath, sep="\t")
    
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols]
    y = df["label"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    
    print("="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(f"\nData loaded: {X.shape[0]:,} records, {X.shape[1]} features")
    print(f"Training set: {X_train.shape[0]:,} records")
    print(f"Test set: {X_test.shape[0]:,} records")
    
    # Class distribution
    n_resistant = (y_train == 1).sum()
    n_susceptible = (y_train == 0).sum()
    print(f"\nClass distribution (train):")
    print(f"  Resistant: {n_resistant:,} ({n_resistant/len(y_train)*100:.1f}%)")
    print(f"  Susceptible: {n_susceptible:,} ({n_susceptible/len(y_train)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test


# ── Define Models ──────────────────────────────────────────────────────────

def get_models():
    """Define all models to compare."""
    models = {}
    
    # 1. XGBoost (our current model)
    models['XGBoost'] = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    
    # 2. Random Forest (bagging)
    models['Random Forest'] = RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    
    # 3. Logistic Regression (linear baseline)
    models['Logistic Regression'] = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight='balanced',
    )
    
    # 4. LightGBM (if available)
    if HAS_LGBM:
        models['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbosity=-1,
        )
    
    return models


# ── Evaluate Models ────────────────────────────────────────────────────────

def evaluate_models(X_train, X_test, y_train, y_test, models):
    """Train and evaluate all models."""
    results = []
    
    print("\n" + "="*60)
    print("TRAINING AND EVALUATING MODELS")
    print("="*60)
    
    for name, model in models.items():
        print(f"\n{'-'*60}")
        print(f"Model: {name}")
        print(f"{'-'*60}")
        
        # Train
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predict
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        roc_auc = roc_auc_score(y_test, y_pred_prob)
        f1 = f1_score(y_test, y_pred)
        
        # Store results
        results.append({
            'model': name,
            'roc_auc': round(roc_auc, 4),
            'f1': round(f1, 4),
            'train_time': round(train_time, 3),
        })
        
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Training time: {train_time:.2f} seconds")
        
        # Save model if it's the best
        if name == 'XGBoost' or name == 'Random Forest':
            joblib.dump(model, os.path.join(MODEL_DIR, f"{name.lower().replace(' ', '_')}.pkl"))
            print(f" Model saved to {MODEL_DIR}/")
    
    return results


# ── Plot Results ───────────────────────────────────────────────────────────

def plot_comparison(results):
    """Create bar charts comparing models."""
    df_results = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. ROC-AUC Comparison
    axes[0].bar(df_results['model'], df_results['roc_auc'], color='steelblue')
    axes[0].axhline(y=0.7423, color='red', linestyle='--', label='XGBoost Baseline')
    axes[0].set_ylabel('ROC-AUC')
    axes[0].set_title('Model Performance: ROC-AUC')
    axes[0].set_ylim(0.5, 0.8)
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=15)
    
    # Add value labels
    for i, v in enumerate(df_results['roc_auc']):
        axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    
    # 2. F1 Score Comparison
    axes[1].bar(df_results['model'], df_results['f1'], color='forestgreen')
    axes[1].axhline(y=0.7323, color='red', linestyle='--', label='XGBoost Baseline')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('Model Performance: F1 Score')
    axes[1].set_ylim(0.4, 0.8)
    axes[1].legend()
    axes[1].tick_params(axis='x', rotation=15)
    
    for i, v in enumerate(df_results['f1']):
        axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    
    # 3. Training Time
    axes[2].bar(df_results['model'], df_results['train_time'], color='orange')
    axes[2].set_ylabel('Training Time (seconds)')
    axes[2].set_title('Model Efficiency: Training Time')
    axes[2].tick_params(axis='x', rotation=15)
    
    for i, v in enumerate(df_results['train_time']):
        axes[2].text(i, v + 0.1, f'{v:.1f}s', ha='center', fontsize=9)
    
    plt.suptitle(
        'Model Comparison for AMR Resistance Prediction\n'
        'Acinetobacter baumannii Aminoglycosides',
        fontsize=13,
        fontweight='bold',
        y=1.02,
    )
    plt.tight_layout()
    
    path = os.path.join(FIG_DIR, '10_model_comparison.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"\n Comparison plot saved to: {path}")
    
    return df_results


# ── Save Results ───────────────────────────────────────────────────────────

def save_comparison_results(df_results):
    """Save results to JSON."""
    results = {
        'models': df_results.to_dict('records'),
        'baseline': {
            'roc_auc': 0.7423,
            'f1': 0.7323,
        },
        'config': {
            'random_state': RANDOM_STATE,
            'test_size': TEST_SIZE,
        }
    }
    
    path = os.path.join(MODEL_DIR, 'comparison_results.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f" Comparison results saved to: {path}")


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Get models
    models = get_models()
    print(f"\nModels to compare: {list(models.keys())}")
    
    # Evaluate models
    results = evaluate_models(X_train, X_test, y_train, y_test, models)
    
    # Create comparison DataFrame
    df_results = pd.DataFrame(results)
    
    # Plot results
    plot_comparison(df_results)
    
    # Save results
    save_comparison_results(df_results)
    
    # Summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(df_results.to_string(index=False))
    print("\n Model comparison complete!")