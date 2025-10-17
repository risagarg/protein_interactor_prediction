"""
Synthetic data generation for PPI prediction demo.

This module generates realistic synthetic data that matches the structure
of the real protein interaction data without using any proprietary information.
"""

import numpy as np
import pandas as pd
from .settings import SEED, DEMO_N_PROTEINS, DEMO_N_CLUSTERS, DEMO_N_EDGES, POS_RATE


def set_seeds(seed=SEED):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    import random
    random.seed(seed)


def generate_demo(n_proteins=DEMO_N_PROTEINS, clusters=DEMO_N_CLUSTERS):
    """
    Generate synthetic protein interaction data for demo purposes.
    
    Args:
        n_proteins: Number of proteins to generate
        clusters: Number of protein clusters (for planted signal)
        
    Returns:
        df_proteins: DataFrame with synthetic protein data
        df_edges: DataFrame with synthetic interaction data (if needed)
    """
    set_seeds()
    rng = np.random.default_rng(SEED)
    
    # Generate protein IDs
    uniprot_ids = [f"P{i:06d}" for i in range(n_proteins)]
    ensembl_ids = [f"ENSG{i:011d}.{i%20 + 1}" for i in range(n_proteins)]
    
    # Generate cluster assignments for planted signal
    cluster_ids = rng.integers(0, clusters, size=n_proteins)
    
    # Generate synthetic features with realistic distributions
    # Core probability: higher for proteins in same cluster
    base_core_prob = rng.beta(2, 3, size=n_proteins)  # Skewed towards lower values
    cluster_boost = 0.3 * (cluster_ids / clusters)  # Higher clusters get boost
    core_prob = np.clip(base_core_prob + cluster_boost, 0, 1)
    
    # Specialist probability: more uniform but with some cluster signal
    spec_prob = rng.beta(1.5, 1.5, size=n_proteins)
    spec_cluster_signal = 0.2 * (cluster_ids % 2)  # Alternating clusters
    spec_prob = np.clip(spec_prob + spec_cluster_signal, 0, 1)
    
    # Expansion probability: inverse relationship with core
    exp_prob = np.clip(1 - core_prob + rng.normal(0, 0.1, size=n_proteins), 0, 1)
    
    # Generate ensemble predictions based on probabilities
    # Higher probability proteins are more likely to be predicted as interactors
    ensemble_threshold = np.percentile(core_prob, 100 - (POS_RATE * 100))
    ensemble_pred = (core_prob > ensemble_threshold).astype(int)
    
    # Generate HPA brain expression (mostly empty like real data)
    hpa_brain_expr = [""] * n_proteins
    # Add some non-empty values for realism
    expr_indices = rng.choice(n_proteins, size=int(0.1 * n_proteins), replace=False)
    for idx in expr_indices:
        hpa_brain_expr[idx] = f"High expression in {rng.choice(['cerebral cortex', 'cerebellum', 'hippocampus'])}"
    
    # Create protein DataFrame matching your exact structure
    df_proteins = pd.DataFrame({
        "actual_protein_id": uniprot_ids,
        "ensembl_id": ensembl_ids,
        "status": ["mapped"] * (n_proteins - 2) + ["no_ensembl_mapping"] * 2,
        "hpa_brain_expression_summary": hpa_brain_expr,
        "core_prob": core_prob,
        "spec_prob": spec_prob,
        "exp_prob": exp_prob,
        "ensemble_pred": ensemble_pred,
        "_cluster": cluster_ids  # Internal use only
    })
    
    # Generate interaction pairs for pair-based analysis (optional)
    n_edges = min(DEMO_N_EDGES, n_proteins * (n_proteins - 1) // 2)
    # Generate all possible pairs and sample from them
    all_pairs = []
    for i in range(n_proteins):
        for j in range(i + 1, n_proteins):
            all_pairs.append((i, j))
    
    if len(all_pairs) >= n_edges:
        pair_indices = rng.choice(len(all_pairs), size=n_edges, replace=False)
        pairs = np.array([all_pairs[i] for i in pair_indices])
    else:
        # If we need more pairs than possible, sample with replacement
        pairs = rng.choice(n_proteins, size=(n_edges, 2), replace=True)
        # Remove self-interactions
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    
    # Plant signal: higher probability of interaction within clusters
    same_cluster = (df_proteins.loc[pairs[:, 0], "_cluster"].values == 
                   df_proteins.loc[pairs[:, 1], "_cluster"].values)
    
    # Generate interaction probabilities
    base_prob = rng.uniform(0.01, 0.1, size=n_edges)
    cluster_boost = 0.3 * same_cluster
    interaction_prob = np.clip(base_prob + cluster_boost, 0, 1)
    
    # Generate labels based on probabilities
    interaction_threshold = np.percentile(interaction_prob, 100 - (POS_RATE * 100))
    labels = (interaction_prob > interaction_threshold).astype(int)
    
    df_edges = pd.DataFrame({
        "src_protein_id": df_proteins.loc[pairs[:, 0], "actual_protein_id"].values,
        "dst_protein_id": df_proteins.loc[pairs[:, 1], "actual_protein_id"].values,
        "label": labels
    })
    
    # Remove internal cluster column
    df_proteins = df_proteins.drop("_cluster", axis=1)
    
    return df_proteins, df_edges


def generate_simple_demo():
    """Generate a smaller demo dataset for quick testing."""
    return generate_demo(n_proteins=500, clusters=3)
