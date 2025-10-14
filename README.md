# Machine Learning Framework for Protein-Protein Interaction Prediction

A comprehensive pipeline for predicting novel protein interactors using ensemble learning, with applications to intrinsically disordered proteins and drug target discovery.

## Overview

This project presents a machine learning framework for predicting protein-protein interactions (PPIs) using a multi-model ensemble approach. The framework combines:

- ESM-2 protein language model embeddings for sequence-based features
- UniProt database integration for functional annotations
- Human Protein Atlas (HPA) expression data
- DBSCAN clustering for protein grouping
- XGBoost ensemble models for prediction

## Key Features

- Rigorous Methodology: Comprehensive data leakage detection and prevention
- Scalable Pipeline: Processes Uniprot human proteome (~20,000 proteins)
- Interpretable Models: SHAP analysis for feature importance
- Production Ready: Modular code structure with proper validation

## Results

- Novel Predictions: Identified 1000 novel protein interactors (tested with amyloid beta)
- Model Performance: F1-score of 0.78 on holdout data
- Cross-Validation: 0.76 ± 0.03 F1-score across folds
- Data Leakage: Zero overlap between training and test sets

## Project Structure
protein_interactor_prediction/
├── notebooks/
│   ├── 01_data_collection_and_filtering.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_development.ipynb
│   ├── 04_data_leakage_analysis.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── 06_proteome_prediction.ipynb
├── README.md
└── requirements.txt

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

## License

This project is licensed under the MIT License.

## Author

**Risa Garg** - risagarg@sas.upenn.edu
