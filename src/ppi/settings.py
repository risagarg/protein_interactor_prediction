"""
Configuration settings for the PPI prediction pipeline.
"""

# Random seed for reproducibility
SEED = 42

# Model parameters
N_FEATURES = 200
N_SPLITS = 5
POS_RATE = 0.15

# Demo data parameters
DEMO_N_PROTEINS = 2000
DEMO_N_CLUSTERS = 6
DEMO_N_EDGES = 8000

# File paths
ARTIFACTS_DIR = "artifacts"
MODELS_DIR = "models"
DATA_DIR = "data"
