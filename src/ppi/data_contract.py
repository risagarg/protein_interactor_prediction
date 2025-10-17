"""
Data contract definitions for PPI prediction pipeline.

This module defines the expected structure of input data to ensure
compatibility and prevent data leakage issues.
"""

# Required columns for protein data
REQUIRED_PROTEIN_COLS = [
    "actual_protein_id",
    "ensembl_id", 
    "status",
    "hpa_brain_expression_summary",
    "core_prob",
    "spec_prob", 
    "exp_prob",
    "ensemble_pred"
]

# Required columns for interaction data (if using pair-based approach)
REQUIRED_EDGE_COLS = [
    "src_protein_id",
    "dst_protein_id", 
    "label"  # 0/1 for negative/positive interaction
]

def validate_contract(df_proteins, df_edges=None):
    """
    Validate that input data conforms to the expected contract.
    
    Args:
        df_proteins: DataFrame with protein data
        df_edges: Optional DataFrame with interaction data
        
    Raises:
        AssertionError: If required columns are missing
    """
    for col in REQUIRED_PROTEIN_COLS:
        assert col in df_proteins.columns, f"Missing required protein column: {col}"
    
    if df_edges is not None:
        for col in REQUIRED_EDGE_COLS:
            assert col in df_edges.columns, f"Missing required edge column: {col}"
    
    # Validate data types
    assert df_proteins["ensemble_pred"].dtype in ["int64", "int32", "bool"], \
        "ensemble_pred must be integer or boolean"
    
    # Validate probability ranges
    prob_cols = ["core_prob", "spec_prob", "exp_prob"]
    for col in prob_cols:
        if col in df_proteins.columns:
            assert df_proteins[col].between(0, 1).all(), f"{col} must be between 0 and 1"
