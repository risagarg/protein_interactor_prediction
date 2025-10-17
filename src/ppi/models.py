"""
Machine learning models for PPI prediction.

This module contains the model definitions and training pipelines
for protein interaction prediction.
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from .settings import SEED, N_FEATURES


def build_pipeline(model_type="xgboost", pos_weight=1.0, n_features=N_FEATURES):
    """
    Build a machine learning pipeline for PPI prediction.
    
    Args:
        model_type: Type of model to use ("xgboost" or "random_forest")
        pos_weight: Weight for positive class (for imbalanced data)
        n_features: Number of features to select
        
    Returns:
        pipeline: Fitted sklearn Pipeline
    """
    if model_type == "xgboost":
        classifier = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=SEED,
            scale_pos_weight=pos_weight,
            n_jobs=2,
            eval_metric="logloss"
        )
    elif model_type == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=SEED,
            class_weight="balanced",
            n_jobs=2
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Determine number of features to select
    # We'll set this dynamically based on actual features available
    pipeline = Pipeline([
        ("select", SelectKBest(f_classif, k="all")),  # Select all features for now
        ("clf", classifier)
    ])
    
    return pipeline


def build_ensemble_pipeline(pos_weight=1.0):
    """
    Build an ensemble pipeline with multiple models.
    
    Args:
        pos_weight: Weight for positive class
        
    Returns:
        ensemble: Dictionary of fitted pipelines
    """
    ensemble = {
        "core": build_pipeline("xgboost", pos_weight, n_features=10),
        "specialist": build_pipeline("random_forest", pos_weight, n_features=15),
        "expansion": build_pipeline("xgboost", pos_weight * 0.5, n_features=12)
    }
    
    return ensemble


def train_ensemble(ensemble, X, y):
    """
    Train an ensemble of models.
    
    Args:
        ensemble: Dictionary of pipelines
        X: Feature matrix
        y: Labels
        
    Returns:
        trained_ensemble: Dictionary of trained pipelines
    """
    trained_ensemble = {}
    
    for name, pipeline in ensemble.items():
        print(f"Training {name} model...")
        pipeline.fit(X, y)
        trained_ensemble[name] = pipeline
    
    return trained_ensemble


def predict_ensemble(trained_ensemble, X):
    """
    Make predictions using an ensemble of models.
    
    Args:
        trained_ensemble: Dictionary of trained pipelines
        X: Feature matrix
        
    Returns:
        predictions: Dictionary of predictions from each model
        ensemble_pred: Combined ensemble prediction
    """
    predictions = {}
    
    for name, pipeline in trained_ensemble.items():
        pred_proba = pipeline.predict_proba(X)[:, 1]
        predictions[name] = pred_proba
    
    # Simple ensemble: average probabilities
    ensemble_pred = np.mean(list(predictions.values()), axis=0)
    
    return predictions, ensemble_pred


def calculate_pos_weight(y):
    """
    Calculate appropriate positive class weight for imbalanced data.
    
    Args:
        y: Labels array
        
    Returns:
        pos_weight: Weight for positive class
    """
    n_neg = np.sum(y == 0)
    n_pos = np.sum(y == 1)
    
    if n_pos == 0:
        return 1.0
    
    return n_neg / n_pos
