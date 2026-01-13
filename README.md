# Machine Learning Framework for Protein-Protein Interaction Prediction (*FOR ERROR LEARNING PURPOSES ONLY - THIS REPO IS UNDER CONSTRUCTION *)
please read - https://risagarg05.wordpress.com/2026/01/01/the-anatomy-of-a-circular-prediction-a-ppi-case-study/

A comprehensive pipeline for predicting novel protein interactors using ensemble learning

## Overview

This project presents a machine learning framework for predicting protein-protein interactions (PPIs) using a multi-model ensemble approach. The framework combines:

- ESM-2 protein language model embeddings for sequence-based features
- UniProt database integration for functional annotations
- Human Protein Atlas (HPA) expression data
- XGBoost ensemble models for prediction

## Features

- Comprehensive data leakage detection and prevention
- Processes Uniprot human proteome (~20,000 proteins)
- SHAP analysis for feature importance
- Modular code structure with proper validation

## Results

- Identified 1000 novel protein interactors (tested with amyloid beta)
-  F1-score of 0.78 on holdout data
-  0.76 ± 0.03 F1-score across folds

## Project Structure
```
protein_interactor_prediction/
<<<<<<< HEAD
├── src/ppi/                    # Main package
│   ├── __init__.py            # Package initialization
│   ├── cli.py                 # Command-line interface
│   ├── demo_data.py           # Synthetic data generation
│   ├── features.py            # Feature engineering
│   ├── models.py              # ML models and pipelines
│   ├── eval.py                # Evaluation metrics and plots
│   ├── data_contract.py       # Data validation
│   └── settings.py            # Configuration
├── tests/                     # Test suite
│   ├── test_data_contract.py
│   ├── test_demo_data.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_demo_learnability.py
├── notebooks/                 # Jupyter notebooks for analysis
=======
├── notebooks/
>>>>>>> cb20ebfa38be8a9ea548eb9d7f218d31011659dd
│   ├── 01_data_collection_and_filtering.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_development.ipynb
│   ├── 04_data_leakage_analysis.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── 06_proteome_prediction.ipynb
<<<<<<< HEAD
├── .github/workflows/         # CI/CD
│   └── ci.yml
├── artifacts/                 # Generated outputs (gitignored)
├── pyproject.toml            # Package configuration
├── Makefile                  # Development commands
└── README.md
```

## Quick Start

### Installation

```bash
git clone https://github.com/risagarg/protein_interactor_prediction.git
cd protein_interactor_prediction
pip install -e .
```

### Demo Usage

```bash
# Run basic demo with synthetic data
python -m ppi.cli demo

# Run ensemble model demo
python -m ppi.cli ensemble-demo

# Run single protein prediction demo
python -m ppi.cli single-protein-demo
```

### Python API Usage

```python
from ppi import generate_demo, build_pair_features, build_pipeline, eval_metrics

# Generate synthetic data
df_proteins, df_edges = generate_demo()

# Build features
X, y = build_pair_features(df_proteins, df_edges)

# Train model
pipeline = build_pipeline("xgboost")
pipeline.fit(X, y)

# Make predictions
predictions = pipeline.predict_proba(X)

# Evaluate
metrics = eval_metrics(y, predictions[:, 1])
print(f"AUPRC: {metrics['AUPRC']:.3f}")
```

## Methodology
=======
├── README.md
└── requirements.txt
>>>>>>> cb20ebfa38be8a9ea548eb9d7f218d31011659dd

### 1. Data Collection & Filtering
- Systematic negative dataset creation using STRING network analysis
- Cellular compartment and tissue expression filtering
- Quality control for protein annotations

### 2. Feature Engineering
- ESM-2 embeddings (1280-dimensional protein representations)
- UniProt functional annotations (GO terms, domains, PTMs)
- HPA tissue expression profiles
- Multi-hot encoding for categorical features

### 3. Model Architecture
- Core Model: Trained on high-confidence positive interactions
- Specialist Model: Trained on outlier/novel interaction patterns
- Expansion Model: Trained on low-confidence positives
- Ensemble: Logical OR combination with optimized thresholds

### 4. Validation & Quality Control
- 80/20 stratified train/test split
- Comprehensive data leakage detection
- Cross-validation with proper feature alignment
- SHAP analysis for model interpretability

### 5. Reproducibility 
- This is a research project demonstrating ML techniques for protein interaction prediction. The notebooks show the complete methodology, but require access to protein databases.

## Data Leakage Prevention
We identified and corrected a data leakage issue in our pipeline. 
See Notebook 4 for detailed analysis of the problem and solution.

## Scientific Applications
This framework is meant to be applied to:

- Drug Discovery: Identifying novel drug targets
- Intrinsically Disordered Proteins: Understanding IDP interactions
- Disease Mechanisms: Mapping protein interaction networks
- Evolutionary Biology: Studying protein evolution

## Key Publications & References

- ESM-2: Lin et al., 2023
- STRING Database: Szklarczyk et al., 2021
- Human Protein Atlas: Uhlén et al., 2015

## Contributing

Please feel free to submit a Pull Request.

## Author

**Risa Garg** - risagarg@sas.upenn.edu
