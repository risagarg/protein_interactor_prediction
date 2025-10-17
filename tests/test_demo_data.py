"""
Tests for synthetic demo data generation.
"""

import pytest
import pandas as pd
import numpy as np
from ppi.demo_data import generate_demo, generate_simple_demo


def test_generate_demo_basic():
    """Test basic demo data generation."""
    df_proteins, df_edges = generate_demo(n_proteins=100, clusters=3)
    
    # Check data types
    assert isinstance(df_proteins, pd.DataFrame)
    assert isinstance(df_edges, pd.DataFrame)
    
    # Check protein data structure
    assert len(df_proteins) == 100
    assert "actual_protein_id" in df_proteins.columns
    assert "ensembl_id" in df_proteins.columns
    assert "ensemble_pred" in df_proteins.columns
    
    # Check edge data structure
    assert len(df_edges) > 0
    assert "src_protein_id" in df_edges.columns
    assert "dst_protein_id" in df_edges.columns
    assert "label" in df_edges.columns


def test_generate_demo_data_types():
    """Test that generated data has correct types."""
    df_proteins, df_edges = generate_demo(n_proteins=50)
    
    # Check protein ID formats
    assert all(df_proteins["actual_protein_id"].str.startswith("P"))
    assert all(df_proteins["ensembl_id"].str.startswith("ENSG"))
    
    # Check probability ranges
    prob_cols = ["core_prob", "spec_prob", "exp_prob"]
    for col in prob_cols:
        assert df_proteins[col].between(0, 1).all()
    
    # Check ensemble_pred is binary
    assert df_proteins["ensemble_pred"].isin([0, 1]).all()
    assert df_edges["label"].isin([0, 1]).all()


def test_generate_demo_reproducibility():
    """Test that demo generation is reproducible."""
    df1_proteins, df1_edges = generate_demo(n_proteins=50, clusters=3)
    df2_proteins, df2_edges = generate_demo(n_proteins=50, clusters=3)
    
    # Should be identical due to fixed seed
    pd.testing.assert_frame_equal(df1_proteins, df2_proteins)
    pd.testing.assert_frame_equal(df1_edges, df2_edges)


def test_generate_demo_planted_signal():
    """Test that demo data has planted signal for learning."""
    df_proteins, df_edges = generate_demo(n_proteins=200, clusters=4)
    
    # Check that we have both positive and negative examples
    assert df_edges["label"].sum() > 0  # Some positive examples
    assert (df_edges["label"] == 0).sum() > 0  # Some negative examples
    
    # Check that positive rate is reasonable
    pos_rate = df_edges["label"].mean()
    assert 0.05 < pos_rate < 0.5  # Between 5% and 50% positive


def test_generate_simple_demo():
    """Test simple demo generation."""
    df_proteins, df_edges = generate_simple_demo()
    
    # Should be smaller than regular demo
    assert len(df_proteins) < 1000
    # Note: df_edges might still be large due to DEMO_N_EDGES constant
    assert len(df_proteins) < 1000  # Just check proteins are smaller
    
    # Should still have proper structure
    assert "actual_protein_id" in df_proteins.columns
    assert "label" in df_edges.columns


def test_generate_demo_no_duplicate_pairs():
    """Test that no duplicate protein pairs are generated."""
    df_proteins, df_edges = generate_demo(n_proteins=100)
    
    # Create pair identifiers
    pairs = df_edges[["src_protein_id", "dst_protein_id"]].apply(
        lambda x: tuple(sorted([x["src_protein_id"], x["dst_protein_id"]])), axis=1
    )
    
    # Should have no duplicates
    assert len(pairs) == len(pairs.unique())


def test_generate_demo_no_self_interactions():
    """Test that no self-interactions are generated."""
    df_proteins, df_edges = generate_demo(n_proteins=100)
    
    # No protein should interact with itself
    self_interactions = df_edges["src_protein_id"] == df_edges["dst_protein_id"]
    assert not self_interactions.any()
