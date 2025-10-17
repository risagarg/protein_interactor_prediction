"""
Tests for data contract validation.
"""

import pytest
import pandas as pd
import numpy as np
from ppi.demo_data import generate_demo
from ppi.data_contract import validate_contract, REQUIRED_PROTEIN_COLS, REQUIRED_EDGE_COLS


def test_contract_with_valid_data():
    """Test that valid data passes contract validation."""
    df_proteins, df_edges = generate_demo(n_proteins=100)
    
    # Should not raise any exceptions
    validate_contract(df_proteins, df_edges)


def test_contract_missing_protein_columns():
    """Test that missing protein columns raise AssertionError."""
    df_proteins, df_edges = generate_demo(n_proteins=100)
    
    # Remove a required column
    df_proteins_missing = df_proteins.drop(columns=["core_prob"])
    
    with pytest.raises(AssertionError, match="Missing required protein column: core_prob"):
        validate_contract(df_proteins_missing, df_edges)


def test_contract_missing_edge_columns():
    """Test that missing edge columns raise AssertionError."""
    df_proteins, df_edges = generate_demo(n_proteins=100)
    
    # Remove a required column
    df_edges_missing = df_edges.drop(columns=["label"])
    
    with pytest.raises(AssertionError, match="Missing required edge column: label"):
        validate_contract(df_proteins, df_edges_missing)


def test_contract_invalid_probability_ranges():
    """Test that invalid probability ranges raise AssertionError."""
    df_proteins, df_edges = generate_demo(n_proteins=100)
    
    # Set invalid probability values
    df_proteins_invalid = df_proteins.copy()
    df_proteins_invalid["core_prob"] = 2.0  # Invalid: > 1
    
    with pytest.raises(AssertionError, match="core_prob must be between 0 and 1"):
        validate_contract(df_proteins_invalid, df_edges)


def test_contract_invalid_ensemble_pred_type():
    """Test that invalid ensemble_pred type raises AssertionError."""
    df_proteins, df_edges = generate_demo(n_proteins=100)
    
    # Set invalid ensemble_pred type
    df_proteins_invalid = df_proteins.copy()
    df_proteins_invalid["ensemble_pred"] = "yes"  # Invalid: string instead of int/bool
    
    with pytest.raises(AssertionError, match="ensemble_pred must be integer or boolean"):
        validate_contract(df_proteins_invalid, df_edges)


def test_required_columns_are_defined():
    """Test that required columns are properly defined."""
    assert len(REQUIRED_PROTEIN_COLS) > 0
    assert len(REQUIRED_EDGE_COLS) > 0
    
    # Check that all required columns are strings
    assert all(isinstance(col, str) for col in REQUIRED_PROTEIN_COLS)
    assert all(isinstance(col, str) for col in REQUIRED_EDGE_COLS)
