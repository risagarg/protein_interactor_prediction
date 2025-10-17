"""
Tests for feature engineering.
"""

import pytest
import pandas as pd
import numpy as np
from ppi.demo_data import generate_demo
from ppi.features import build_pair_features, build_single_protein_features, preprocess_features


def test_build_pair_features():
    """Test pair feature building."""
    df_proteins, df_edges = generate_demo(n_proteins=100)
    X, y = build_pair_features(df_proteins, df_edges)
    
    # Check output types
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, np.ndarray)
    
    # Check dimensions
    assert len(X) == len(df_edges)
    assert len(y) == len(df_edges)
    
    # Check feature columns
    expected_features = [
        "core_prob_diff", "spec_prob_diff", "exp_prob_diff",
        "core_prob_mean", "spec_prob_mean", "exp_prob_mean",
        "core_prob_product", "spec_prob_product", "exp_prob_product",
        "both_mapped", "src_mapped", "dst_mapped",
        "src_has_expr", "dst_has_expr", "both_have_expr"
    ]
    
    for feature in expected_features:
        assert feature in X.columns
    
    # Check that features are numeric
    assert X.select_dtypes(include=[np.number]).shape[1] == X.shape[1]
    
    # Check that labels are binary
    assert set(y) <= {0, 1}


def test_build_single_protein_features():
    """Test single protein feature building."""
    df_proteins, _ = generate_demo(n_proteins=100)
    X = build_single_protein_features(df_proteins)
    
    # Check output type
    assert isinstance(X, pd.DataFrame)
    
    # Check dimensions
    assert len(X) == len(df_proteins)
    
    # Check feature columns
    expected_features = [
        "core_prob", "spec_prob", "exp_prob",
        "prob_sum", "prob_max", "prob_min", "prob_std",
        "is_mapped", "has_expression"
    ]
    
    for feature in expected_features:
        assert feature in X.columns
    
    # Check that features are numeric
    assert X.select_dtypes(include=[np.number]).shape[1] == X.shape[1]


def test_preprocess_features():
    """Test feature preprocessing."""
    df_proteins, df_edges = generate_demo(n_proteins=100)
    X, _ = build_pair_features(df_proteins, df_edges)
    
    # Test fitting scaler
    X_scaled, scaler = preprocess_features(X)
    
    # Check output types
    assert isinstance(X_scaled, pd.DataFrame)
    assert hasattr(scaler, 'transform')
    
    # Check dimensions
    assert X_scaled.shape == X.shape
    assert list(X_scaled.columns) == list(X.columns)
    
    # Check that scaling worked (mean should be close to 0, std close to 1)
    numeric_cols = X_scaled.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        assert abs(X_scaled[col].mean()) < 0.1  # Mean close to 0
        assert 0.8 < X_scaled[col].std() < 1.2  # Std close to 1


def test_preprocess_features_with_fitted_scaler():
    """Test feature preprocessing with pre-fitted scaler."""
    df_proteins, df_edges = generate_demo(n_proteins=100)
    X, _ = build_pair_features(df_proteins, df_edges)
    
    # Fit scaler on first half
    X_train = X.iloc[:len(X)//2]
    X_test = X.iloc[len(X)//2:]
    
    X_train_scaled, scaler = preprocess_features(X_train)
    X_test_scaled, _ = preprocess_features(X_test, fit_scaler=scaler)
    
    # Check that both are scaled
    assert isinstance(X_train_scaled, pd.DataFrame)
    assert isinstance(X_test_scaled, pd.DataFrame)
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape


def test_feature_engineering_consistency():
    """Test that feature engineering is consistent across runs."""
    df_proteins, df_edges = generate_demo(n_proteins=50)
    
    # Run feature engineering twice
    X1, y1 = build_pair_features(df_proteins, df_edges)
    X2, y2 = build_pair_features(df_proteins, df_edges)
    
    # Should be identical
    pd.testing.assert_frame_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_feature_engineering_with_empty_data():
    """Test feature engineering with empty data."""
    # Create empty dataframes with correct structure
    df_proteins = pd.DataFrame(columns=[
        "actual_protein_id", "ensembl_id", "status", 
        "hpa_brain_expression_summary", "core_prob", 
        "spec_prob", "exp_prob", "ensemble_pred"
    ])
    
    df_edges = pd.DataFrame(columns=[
        "src_protein_id", "dst_protein_id", "label"
    ])
    
    # Should handle empty data gracefully
    X, y = build_pair_features(df_proteins, df_edges)
    assert len(X) == 0
    assert len(y) == 0
