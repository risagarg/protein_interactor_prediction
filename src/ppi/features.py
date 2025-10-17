"""
Feature engineering for PPI prediction.

This module handles feature extraction and preprocessing for the
protein interaction prediction pipeline.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def build_pair_features(df_proteins: pd.DataFrame, df_edges: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Build features for protein pair interactions.
    
    Args:
        df_proteins: DataFrame with protein data
        df_edges: DataFrame with protein pair interactions
        
    Returns:
        X: Feature matrix for pairs
        y: Labels for pairs
    """
    # Set protein data as index for easy lookup
    P = df_proteins.set_index("actual_protein_id")
    
    def extract_pair_features(row):
        """Extract features for a single protein pair."""
        src_id, dst_id = row["src_protein_id"], row["dst_protein_id"]
        
        # Get protein data
        src_protein = P.loc[src_id]
        dst_protein = P.loc[dst_id]
        
        # Extract features
        features = {
            # Probability-based features
            "core_prob_diff": abs(src_protein["core_prob"] - dst_protein["core_prob"]),
            "spec_prob_diff": abs(src_protein["spec_prob"] - dst_protein["spec_prob"]),
            "exp_prob_diff": abs(src_protein["exp_prob"] - dst_protein["exp_prob"]),
            
            # Combined probability features
            "core_prob_mean": (src_protein["core_prob"] + dst_protein["core_prob"]) / 2,
            "spec_prob_mean": (src_protein["spec_prob"] + dst_protein["spec_prob"]) / 2,
            "exp_prob_mean": (src_protein["exp_prob"] + dst_protein["exp_prob"]) / 2,
            
            # Product features (interaction strength)
            "core_prob_product": src_protein["core_prob"] * dst_protein["core_prob"],
            "spec_prob_product": src_protein["spec_prob"] * dst_protein["spec_prob"],
            "exp_prob_product": src_protein["exp_prob"] * dst_protein["exp_prob"],
            
            # Status features
            "both_mapped": int(src_protein["status"] == "mapped" and dst_protein["status"] == "mapped"),
            "src_mapped": int(src_protein["status"] == "mapped"),
            "dst_mapped": int(dst_protein["status"] == "mapped"),
            
            # Expression features
            "src_has_expr": int(src_protein["hpa_brain_expression_summary"] != ""),
            "dst_has_expr": int(dst_protein["hpa_brain_expression_summary"] != ""),
            "both_have_expr": int(src_protein["hpa_brain_expression_summary"] != "" and 
                                dst_protein["hpa_brain_expression_summary"] != ""),
        }
        
        return pd.Series(features)
    
    # Apply feature extraction to all pairs
    X = df_edges.apply(extract_pair_features, axis=1)
    y = df_edges["label"].astype(int).values
    
    return X, y


def build_single_protein_features(df_proteins: pd.DataFrame) -> pd.DataFrame:
    """
    Build features for single protein analysis.
    
    Args:
        df_proteins: DataFrame with protein data
        
    Returns:
        X: Feature matrix for proteins
    """
    # Use the probability scores as features
    feature_cols = ["core_prob", "spec_prob", "exp_prob"]
    X = df_proteins[feature_cols].copy()
    
    # Add derived features
    X["prob_sum"] = X["core_prob"] + X["spec_prob"] + X["exp_prob"]
    X["prob_max"] = X[feature_cols].max(axis=1)
    X["prob_min"] = X[feature_cols].min(axis=1)
    X["prob_std"] = X[feature_cols].std(axis=1)
    
    # Add status features
    X["is_mapped"] = (df_proteins["status"] == "mapped").astype(int)
    X["has_expression"] = (df_proteins["hpa_brain_expression_summary"] != "").astype(int)
    
    return X


def preprocess_features(X: pd.DataFrame, fit_scaler: Optional[object] = None) -> Tuple[pd.DataFrame, object]:
    """
    Preprocess features with scaling and normalization.
    
    Args:
        X: Feature matrix
        fit_scaler: Optional pre-fitted scaler
        
    Returns:
        X_scaled: Scaled feature matrix
        scaler: Fitted scaler object
    """
    from sklearn.preprocessing import StandardScaler
    
    if fit_scaler is None:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
    else:
        scaler = fit_scaler
        X_scaled = pd.DataFrame(
            scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
    
    return X_scaled, scaler
