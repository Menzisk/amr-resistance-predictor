#!/usr/bin/env python3
"""
05_hyperparameter_tuning.py

Hyperparameter tuning for XGBoost AMR resistance classifier.

This script performs a systematic search over hyperparameter space
to find the optimal configuration for our model.

Strategy:
    1. Use RandomizedSearchCV for broad exploration
    2. Then refine with GridSearchCV on promising regions
    3. Save the best parameters to config.yaml

Inputs:
    data/processed/amr_features.tsv

Outputs:
    outputs/models/best_params.json
    outputs/figures/09_tuning_results.png

Author: Menzi Sikakane (menzisk)
Date:   2026-06-17
License: MIT
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import os
import yaml
import json
import joblib
from time import time

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier

# ── Load Configuration ──────────────────────────────────────────────────────
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

PROCESSED_DIR = CONFIG['data']['processed_dir']
MODEL_DIR = CONFIG['outputs']['model_dir']
FIG_DIR = CONFIG['outputs']['figure_dir']
RANDOM_STATE = CONFIG['model']['random_state']
TEST_SIZE = CONFIG['model']['test_size']
N_FOLDS = CONFIG['model']['n_folds']
DPI = CONFIG['outputs']['dpi']

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ── Functions ──────────────────────────────────────────────────────────────

def load_data(filepath: str):
    """Load features and split into train/test."""
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

    print(f"Data loaded: {X.shape[0]:,} records, {X.shape[1]} features")
    print(f"Training set: {X_train.shape[0]:,} records")
    print(f"Test set: {X_test.shape[0]:,} records")

    return X_train, X_test, y_train, y_test


def get_class_weight(y_train):
    """Compute scale_pos_weight for imbalanced data."""
    n_resistant = (y_train == 1).sum()
    n_susceptible = (y_train == 0).sum()
    return n_susceptible / n_resistant


def randomized_search(X_train, y_train, scale_pos_weight):
    """
    Run RandomizedSearchCV for broad parameter exploration.

    Why RandomizedSearchCV?
    - GridSearchCV on 30+ parameter combinations would take hours
    - RandomizedSearchCV samples random combinations, finds good regions faster
    """
    print("\n" + "="*60)
    print("RANDOMIZED SEARCH - Phase 1")
    print("="*60)

    # Define parameter distributions to sample from
    param_dist = {
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2],
        'n_estimators': [100, 150, 200, 300, 400, 500],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4],
    }

    # Base model with fixed parameters
    base_model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        verbosity=0,
    )

    # Random search with 50 iterations (tries 50 random combinations)
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=50,  # Try 50 random combinations
        cv=N_FOLDS,
        scoring='roc_auc',
        n_jobs=-1,  # Use all CPU cores
        random_state=RANDOM_STATE,
        verbose=1,
    )

    print("\nSearching 50 random parameter combinations.")
    print("This will take 5-15 minutes depending on your machine.")

    start_time = time()
    random_search.fit(X_train, y_train)
    elapsed = time() - start_time

    print(f"\n Search completed in {elapsed/60:.1f} minutes")
    print(f"\nBest parameters found:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"Best CV ROC-AUC: {random_search.best_score_:.4f}")

    return random_search


def grid_search_refine(X_train, y_train, scale_pos_weight, best_params):
    """
    Run GridSearchCV to refine around the best parameters found.

    Why GridSearchCV?
    - Exhaustively tries all combinations in a smaller grid
    - More precise than random search for final tuning
    """
    print("\n" + "="*60)
    print("GRID SEARCH - Phase 2 (Refinement)")
    print("="*60)

    # Create a refined grid around the best parameters
    # Take the best value and check neighbours
    def refine_param(current, values):
        """Get values around the current best."""
        idx = values.index(current) if current in values else len(values)//2
        start = max(0, idx - 1)
        end = min(len(values), idx + 2)
        return values[start:end]

    # Define the full parameter space from phase 1
    max_depth_values = [3, 4, 5, 6, 7, 8]
    learning_rate_values = [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
    n_estimators_values = [100, 150, 200, 300, 400, 500]
    subsample_values = [0.6, 0.7, 0.8, 0.9, 1.0]
    colsample_bytree_values = [0.6, 0.7, 0.8, 0.9, 1.0]
    min_child_weight_values = [1, 3, 5, 7]
    gamma_values = [0, 0.1, 0.2, 0.3, 0.4]

    # Refine each parameter
    refined_grid = {
        'max_depth': refine_param(best_params['max_depth'], max_depth_values),
        'learning_rate': refine_param(best_params['learning_rate'], learning_rate_values),
        'n_estimators': refine_param(best_params['n_estimators'], n_estimators_values),
        'subsample': refine_param(best_params['subsample'], subsample_values),
        'colsample_bytree': refine_param(best_params['colsample_bytree'], colsample_bytree_values),
        'min_child_weight': refine_param(best_params['min_child_weight'], min_child_weight_values),
        'gamma': refine_param(best_params['gamma'], gamma_values),
    }

    print("\nRefined parameter grid:")
    for param, values in refined_grid.items():
        print(f"  {param}: {values}")

    # Base model
    base_model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        verbosity=0,
    )

    # Grid search with all combinations
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=refined_grid,
        cv=N_FOLDS,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
    )

    print(f"\nSearching {len(refined_grid['max_depth']) * len(refined_grid['learning_rate']) * len(refined_grid['n_estimators'])} combinations.")

    start_time = time()
    grid_search.fit(X_train, y_train)
    elapsed = time() - start_time

    print(f"\n✓ Search completed in {elapsed/60:.1f} minutes")
    print(f"\nBest parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

    return grid_search


def evaluate_best_model(model, X_test, y_test):
    """Evaluate the best model on held-out test set."""
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_pred_prob)
    f1 = f1_score(y_test, y_pred)

    print("\n" + "="*60)
    print("BEST MODEL EVALUATION (Test Set)")
    print("="*60)
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return roc_auc, f1


def save_results(random_search, grid_search, X_test, y_test):
    """Save best parameters and results."""
    # Get best model (grid search if available, otherwise random)
    if grid_search:
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_cv_score = grid_search.best_score_
        search_type = "GridSearchCV"
    else:
        best_model = random_search.best_estimator_
        best_params = random_search.best_params_
        best_cv_score = random_search.best_score_
        search_type = "RandomizedSearchCV"

    # Evaluate on test set
    y_pred_prob = best_model.predict_proba(X_test)[:, 1]
    y_pred = best_model.predict(X_test)
    test_roc_auc = roc_auc_score(y_test, y_pred_prob)
    test_f1 = f1_score(y_test, y_pred)

    # Save results
    results = {
        "best_params": best_params,
        "best_cv_roc_auc": round(best_cv_score, 4),
        "test_roc_auc": round(test_roc_auc, 4),
        "test_f1": round(test_f1, 4),
        "search_type": search_type,
    }

    results_path = os.path.join(MODEL_DIR, "best_params.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Best parameters saved to: {results_path}")

    # Save the best model
    model_path = os.path.join(MODEL_DIR, "xgboost_model_tuned.pkl")
    joblib.dump(best_model, model_path)
    print(f" Tuned model saved to: {model_path}")

    return results, best_params


def plot_tuning_results(random_search, grid_search):
    """Visualise tuning results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Random search results
    if random_search:
        results = pd.DataFrame(random_search.cv_results_)
        axes[0].scatter(
            range(len(results)),
            results['mean_test_score'],
            alpha=0.6,
            s=20,
        )
        axes[0].axhline(y=random_search.best_score_, color='red', linestyle='--',
                        label=f"Best: {random_search.best_score_:.4f}")
        axes[0].set_xlabel("Random Search Iteration")
        axes[0].set_ylabel("CV ROC-AUC")
        axes[0].set_title("Phase 1: Randomized Search")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # Plot 2: Grid search results
    if grid_search:
        results = pd.DataFrame(grid_search.cv_results_)
        axes[1].scatter(
            range(len(results)),
            results['mean_test_score'],
            alpha=0.6,
            s=20,
            color='green',
        )
        axes[1].axhline(y=grid_search.best_score_, color='red', linestyle='--',
                        label=f"Best: {grid_search.best_score_:.4f}")
        axes[1].set_xlabel("Grid Search Combination")
        axes[1].set_ylabel("CV ROC-AUC")
        axes[1].set_title("Phase 2: Grid Search Refinement")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        "Hyperparameter Tuning Results\n"
        "XGBoost AMR Resistance Classifier",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    path = os.path.join(FIG_DIR, "09_tuning_results.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"✓ Tuning visualisation saved to: {path}")


def update_config_yaml(best_params):
    """Update config.yaml with the best parameters."""
    config_path = "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Update model parameters
    config['model']['max_depth'] = best_params.get('max_depth', config['model']['max_depth'])
    config['model']['learning_rate'] = best_params.get('learning_rate', config['model']['learning_rate'])
    config['model']['n_estimators'] = best_params.get('n_estimators', config['model']['n_estimators'])
    config['model']['subsample'] = best_params.get('subsample', config['model']['subsample'])
    config['model']['colsample_bytree'] = best_params.get('colsample_bytree', config['model']['colsample_bytree'])

    # Save updated config
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\n config.yaml updated with best parameters")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*60)
    print("XGBOOST HYPERPARAMETER TUNING")
    print("="*60)
    print("\nThis script will find the best hyperparameters for your model.")
    print("It runs in two phases:")
    print("  1. RandomizedSearchCV - broad exploration (50 iterations)")
    print("  2. GridSearchCV - refinement around best parameters")
    print("\nThis may take 15-30 minutes. Please be patient.\n")

    # Load data
    X_train, X_test, y_train, y_test = load_data(
        os.path.join(PROCESSED_DIR, "amr_features.tsv")
    )

    # Compute class weight
    scale_pos_weight = get_class_weight(y_train)
    print(f"\nClass weight (scale_pos_weight): {scale_pos_weight:.4f}")

    # Phase 1: Randomized Search
    random_search = randomized_search(X_train, y_train, scale_pos_weight)

    # Phase 2: Grid Search Refinement
    grid_search = grid_search_refine(
        X_train, y_train, scale_pos_weight, random_search.best_params_
    )

    # Evaluate on test set
    best_model = grid_search.best_estimator_
    test_roc_auc, test_f1 = evaluate_best_model(best_model, X_test, y_test)

    # Save results
    results, best_params = save_results(random_search, grid_search, X_test, y_test)

    # Update config.yaml
    update_config_yaml(best_params)

    # Plot results
    plot_tuning_results(random_search, grid_search)

    # Compare with baseline
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    print(f"Baseline ROC-AUC: 0.7423")
    print(f"Tuned ROC-AUC   : {test_roc_auc:.4f}")
    print(f"Improvement     : {(test_roc_auc - 0.7423)*100:.2f}%")

    print("\n Hyperparameter tuning complete!")
    print("\nYour config.yaml has been updated with the best parameters.")
    print("Run 'python src/03_train_model.py' to train with the new settings.")