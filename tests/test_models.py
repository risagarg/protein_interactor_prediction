"""
Tests for machine learning models.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from ppi.demo_data import generate_demo
from ppi.features import build_pair_features
from ppi.models import (
    build_pipeline, build_ensemble_pipeline, train_ensemble, 
    predict_ensemble, calculate_pos_weight
)


def test_build_pipeline():
    """Test pipeline building."""
    # Test XGBoost pipeline
    pipeline = build_pipeline("xgboost", pos_weight=1.0)
    assert hasattr(pipeline, 'fit')
    assert hasattr(pipeline, 'predict')
    assert hasattr(pipeline, 'predict_proba')
    
    # Test Random Forest pipeline
    pipeline = build_pipeline("random_forest", pos_weight=1.0)
    assert hasattr(pipeline, 'fit')
    assert hasattr(pipeline, 'predict')
    assert hasattr(pipeline, 'predict_proba')


def test_build_pipeline_invalid_model():
    """Test that invalid model type raises error."""
    with pytest.raises(ValueError, match="Unknown model type"):
        build_pipeline("invalid_model")


def test_calculate_pos_weight():
    """Test positive class weight calculation."""
    # Balanced data
    y_balanced = np.array([0, 1, 0, 1])
    weight = calculate_pos_weight(y_balanced)
    assert weight == 1.0
    
    # Imbalanced data (more negatives)
    y_imbalanced = np.array([0, 0, 0, 1])
    weight = calculate_pos_weight(y_imbalanced)
    assert weight == 3.0
    
    # No positives
    y_no_pos = np.array([0, 0, 0, 0])
    weight = calculate_pos_weight(y_no_pos)
    assert weight == 1.0


def test_pipeline_training():
    """Test that pipeline can be trained."""
    # Generate synthetic data
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    
    # Build and train pipeline
    pipeline = build_pipeline("xgboost", pos_weight=1.0)
    pipeline.fit(X, y)
    
    # Test predictions
    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)
    
    assert len(predictions) == len(y)
    assert probabilities.shape == (len(y), 2)
    assert set(predictions) <= {0, 1}


def test_ensemble_pipeline():
    """Test ensemble pipeline building."""
    ensemble = build_ensemble_pipeline(pos_weight=1.0)
    
    # Check that ensemble has expected models
    expected_models = ["core", "specialist", "expansion"]
    assert set(ensemble.keys()) == set(expected_models)
    
    # Check that each model is a pipeline
    for model_name, pipeline in ensemble.items():
        assert hasattr(pipeline, 'fit')
        assert hasattr(pipeline, 'predict_proba')


def test_ensemble_training():
    """Test ensemble training."""
    # Generate synthetic data
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    
    # Build and train ensemble
    ensemble = build_ensemble_pipeline(pos_weight=1.0)
    trained_ensemble = train_ensemble(ensemble, X, y)
    
    # Check that all models are trained
    assert len(trained_ensemble) == len(ensemble)
    
    # Check that models can make predictions
    for model_name, pipeline in trained_ensemble.items():
        predictions = pipeline.predict(X)
        probabilities = pipeline.predict_proba(X)
        assert len(predictions) == len(y)
        assert probabilities.shape == (len(y), 2)


def test_ensemble_predictions():
    """Test ensemble prediction functionality."""
    # Generate synthetic data
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    
    # Build, train, and predict with ensemble
    ensemble = build_ensemble_pipeline(pos_weight=1.0)
    trained_ensemble = train_ensemble(ensemble, X, y)
    individual_preds, ensemble_pred = predict_ensemble(trained_ensemble, X)
    
    # Check individual predictions
    assert len(individual_preds) == len(ensemble)
    for model_name, preds in individual_preds.items():
        assert len(preds) == len(y)
        assert all(0 <= p <= 1 for p in preds)  # Probabilities between 0 and 1
    
    # Check ensemble prediction
    assert len(ensemble_pred) == len(y)
    assert all(0 <= p <= 1 for p in ensemble_pred)


def test_model_with_real_data_structure():
    """Test models with data that matches real PPI structure."""
    # Generate demo data
    df_proteins, df_edges = generate_demo(n_proteins=100)
    X, y = build_pair_features(df_proteins, df_edges)
    
    # Test single model
    pipeline = build_pipeline("xgboost", pos_weight=1.0)
    pipeline.fit(X, y)
    
    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)
    
    assert len(predictions) == len(y)
    assert probabilities.shape == (len(y), 2)
    
    # Test ensemble
    ensemble = build_ensemble_pipeline(pos_weight=1.0)
    trained_ensemble = train_ensemble(ensemble, X, y)
    individual_preds, ensemble_pred = predict_ensemble(trained_ensemble, X)
    
    assert len(ensemble_pred) == len(y)
    assert all(0 <= p <= 1 for p in ensemble_pred)


def test_model_reproducibility():
    """Test that models are reproducible with fixed seeds."""
    # Generate data
    df_proteins, df_edges = generate_demo(n_proteins=50)
    X, y = build_pair_features(df_proteins, df_edges)
    
    # Train two identical models
    pipeline1 = build_pipeline("xgboost", pos_weight=1.0)
    pipeline2 = build_pipeline("xgboost", pos_weight=1.0)
    
    pipeline1.fit(X, y)
    pipeline2.fit(X, y)
    
    # Predictions should be identical
    pred1 = pipeline1.predict(X)
    pred2 = pipeline2.predict(X)
    np.testing.assert_array_equal(pred1, pred2)
