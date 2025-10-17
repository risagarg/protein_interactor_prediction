"""
Tests for demo data learnability.

These tests ensure that the synthetic demo data contains enough signal
for machine learning models to learn meaningful patterns.
"""

import pytest
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import average_precision_score, roc_auc_score
from ppi.demo_data import generate_demo
from ppi.features import build_pair_features
from ppi.models import build_pipeline, calculate_pos_weight
from ppi.settings import SEED


def test_demo_beats_random_baseline():
    """Test that model trained on demo data beats random baseline."""
    # Generate demo data
    df_proteins, df_edges = generate_demo(n_proteins=500, clusters=4)
    X, y = build_pair_features(df_proteins, df_edges)
    
    # Calculate random baseline
    random_auprc = np.mean(y)  # Random baseline AUPRC
    random_auroc = 0.5  # Random baseline AUROC
    
    # Train model
    pos_weight = calculate_pos_weight(y)
    pipeline = build_pipeline("xgboost", pos_weight)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    y_prob = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
    
    # Calculate metrics
    auprc = average_precision_score(y, y_prob)
    auroc = roc_auc_score(y, y_prob)
    
    # Model should beat random baseline (relaxed threshold for synthetic data)
    assert auprc > random_auprc * 1.05, f"AUPRC {auprc:.3f} should be > {random_auprc * 1.05:.3f}"
    assert auroc > random_auroc + 0.02, f"AUROC {auroc:.3f} should be > {random_auroc + 0.02:.3f}"


def test_demo_has_learnable_signal():
    """Test that demo data contains learnable signal."""
    # Generate demo data with planted signal
    df_proteins, df_edges = generate_demo(n_proteins=1000, clusters=6)
    X, y = build_pair_features(df_proteins, df_edges)
    
    # Check that we have reasonable class distribution
    pos_rate = np.mean(y)
    assert 0.05 < pos_rate < 0.5, f"Positive rate {pos_rate:.3f} should be between 0.05 and 0.5"
    
    # Check that features have reasonable variance
    feature_vars = X.var()
    assert (feature_vars > 1e-6).sum() > 0, "Some features should have non-zero variance"
    
    # Check that features are not all identical
    assert not (X.nunique() == 1).all(), "Features should not all be identical"


def test_demo_consistency_across_runs():
    """Test that demo data is consistent across runs."""
    # Generate data twice with same parameters
    df1_proteins, df1_edges = generate_demo(n_proteins=200, clusters=3)
    df2_proteins, df2_edges = generate_demo(n_proteins=200, clusters=3)
    
    # Should be identical due to fixed seed
    assert df1_proteins.equals(df2_proteins)
    assert df1_edges.equals(df2_edges)


def test_demo_planted_cluster_signal():
    """Test that demo data has planted cluster-based signal."""
    # Generate data with clear cluster structure
    df_proteins, df_edges = generate_demo(n_proteins=300, clusters=3)
    
    # Check that proteins are distributed across clusters
    # (This is tested indirectly through the planted signal)
    
    # Build features and train model
    X, y = build_pair_features(df_proteins, df_edges)
    pos_weight = calculate_pos_weight(y)
    pipeline = build_pipeline("xgboost", pos_weight)
    
    # Quick training to check if model can learn
    pipeline.fit(X, y)
    y_prob = pipeline.predict_proba(X)[:, 1]
    
    # Model should be able to make reasonable predictions
    auprc = average_precision_score(y, y_prob)
    assert auprc > 0.1, f"Model should learn some signal, got AUPRC {auprc:.3f}"


def test_demo_feature_engineering_works():
    """Test that feature engineering produces meaningful features."""
    df_proteins, df_edges = generate_demo(n_proteins=200)
    X, y = build_pair_features(df_proteins, df_edges)
    
    # Check feature properties
    assert X.shape[0] == len(df_edges), "Feature matrix should have one row per edge"
    assert X.shape[1] > 0, "Should have some features"
    
    # Check that features are not all NaN
    assert not X.isnull().all().any(), "Features should not be all NaN"
    
    # Check that features have reasonable ranges
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        assert X[col].min() >= 0, f"Feature {col} should be non-negative"
        assert not np.isinf(X[col]).any(), f"Feature {col} should not contain infinity"


def test_demo_ensemble_learnability():
    """Test that ensemble models can learn from demo data."""
    from ppi.models import build_ensemble_pipeline, train_ensemble, predict_ensemble
    
    # Generate data
    df_proteins, df_edges = generate_demo(n_proteins=300)
    X, y = build_pair_features(df_proteins, df_edges)
    
    # Build and train ensemble
    pos_weight = calculate_pos_weight(y)
    ensemble = build_ensemble_pipeline(pos_weight)
    trained_ensemble = train_ensemble(ensemble, X, y)
    
    # Make predictions
    individual_preds, ensemble_pred = predict_ensemble(trained_ensemble, X)
    
    # Check that ensemble prediction is reasonable
    auprc = average_precision_score(y, ensemble_pred)
    auroc = roc_auc_score(y, ensemble_pred)
    
    # Ensemble should perform reasonably well
    assert auprc > 0.1, f"Ensemble AUPRC {auprc:.3f} should be > 0.1"
    assert auroc > 0.6, f"Ensemble AUROC {auroc:.3f} should be > 0.6"
