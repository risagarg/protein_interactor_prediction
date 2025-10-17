"""
Command-line interface for PPI prediction.

This module provides a CLI for running the protein interaction prediction
pipeline with various options and demo functionality.
"""

import argparse
import os
import sys
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import average_precision_score

from .demo_data import generate_demo, generate_simple_demo
from .features import build_pair_features, build_single_protein_features
from .models import build_pipeline, build_ensemble_pipeline, train_ensemble, predict_ensemble, calculate_pos_weight
from .eval import comprehensive_eval
from .settings import SEED, ARTIFACTS_DIR


def set_seeds(seed=SEED):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def cmd_demo():
    """Run the demo with synthetic data."""
    print("🧬 Running PPI Prediction Demo...")
    print("=" * 50)
    
    # Generate synthetic data
    print("📊 Generating synthetic protein data...")
    df_proteins, df_edges = generate_demo()
    
    print(f"   Generated {len(df_proteins)} proteins")
    print(f"   Generated {len(df_edges)} protein pairs")
    print(f"   Positive interaction rate: {df_edges['label'].mean():.3f}")
    
    # Build features
    print("\n🔧 Building features...")
    X, y = build_pair_features(df_proteins, df_edges)
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Features: {list(X.columns)}")
    
    # Calculate class weights
    pos_weight = calculate_pos_weight(y)
    print(f"   Positive class weight: {pos_weight:.3f}")
    
    # Train and evaluate model
    print("\n🤖 Training model...")
    pipeline = build_pipeline("xgboost", pos_weight)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    y_prob = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
    
    # Evaluate
    print("\n📈 Evaluating model...")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    metrics = comprehensive_eval(y, y_prob, outdir=ARTIFACTS_DIR, model_name="demo")
    
    # Print results
    print("\n🎯 Results:")
    print(f"   AUPRC: {metrics['AUPRC']:.3f}")
    print(f"   AUROC: {metrics['AUROC']:.3f}")
    print(f"   F1 Score: {metrics['F1']:.3f}")
    
    # Check if model beats random baseline
    random_auprc = metrics['pos_rate']
    if metrics['AUPRC'] > random_auprc * 1.5:
        print("✅ Model successfully learns from synthetic data!")
    else:
        print("⚠️  Model performance is close to random baseline")
    
    print(f"\n📁 Results saved to {ARTIFACTS_DIR}/")
    print("Demo completed successfully! 🎉")


def cmd_ensemble_demo():
    """Run ensemble model demo."""
    print("🧬 Running Ensemble PPI Prediction Demo...")
    print("=" * 50)
    
    # Generate synthetic data
    print("📊 Generating synthetic protein data...")
    df_proteins, df_edges = generate_simple_demo()  # Smaller dataset for ensemble
    
    print(f"   Generated {len(df_proteins)} proteins")
    print(f"   Generated {len(df_edges)} protein pairs")
    
    # Build features
    print("\n🔧 Building features...")
    X, y = build_pair_features(df_proteins, df_edges)
    
    # Calculate class weights
    pos_weight = calculate_pos_weight(y)
    
    # Build and train ensemble
    print("\n🤖 Training ensemble models...")
    ensemble = build_ensemble_pipeline(pos_weight)
    trained_ensemble = train_ensemble(ensemble, X, y)
    
    # Make predictions
    print("\n🔮 Making ensemble predictions...")
    individual_preds, ensemble_pred = predict_ensemble(trained_ensemble, X)
    
    # Evaluate ensemble
    print("\n📈 Evaluating ensemble...")
    metrics = comprehensive_eval(y, ensemble_pred, outdir=ARTIFACTS_DIR, model_name="ensemble")
    
    # Print results
    print("\n🎯 Ensemble Results:")
    print(f"   AUPRC: {metrics['AUPRC']:.3f}")
    print(f"   AUROC: {metrics['AUROC']:.3f}")
    print(f"   F1 Score: {metrics['F1']:.3f}")
    
    print(f"\n📁 Results saved to {ARTIFACTS_DIR}/")
    print("Ensemble demo completed successfully! 🎉")


def cmd_single_protein_demo():
    """Run single protein prediction demo."""
    print("🧬 Running Single Protein Prediction Demo...")
    print("=" * 50)
    
    # Generate synthetic data
    print("📊 Generating synthetic protein data...")
    df_proteins, _ = generate_demo()
    
    print(f"   Generated {len(df_proteins)} proteins")
    print(f"   Positive prediction rate: {df_proteins['ensemble_pred'].mean():.3f}")
    
    # Build features
    print("\n🔧 Building single protein features...")
    X = build_single_protein_features(df_proteins)
    y = df_proteins['ensemble_pred'].values
    
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Features: {list(X.columns)}")
    
    # Train and evaluate
    print("\n🤖 Training single protein model...")
    pos_weight = calculate_pos_weight(y)
    pipeline = build_pipeline("xgboost", pos_weight)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    y_prob = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
    
    # Evaluate
    print("\n📈 Evaluating model...")
    metrics = comprehensive_eval(y, y_prob, outdir=ARTIFACTS_DIR, model_name="single_protein")
    
    # Print results
    print("\n🎯 Single Protein Results:")
    print(f"   AUPRC: {metrics['AUPRC']:.3f}")
    print(f"   AUROC: {metrics['AUROC']:.3f}")
    print(f"   F1 Score: {metrics['F1']:.3f}")
    
    print(f"\n📁 Results saved to {ARTIFACTS_DIR}/")
    print("Single protein demo completed successfully! 🎉")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Protein-Protein Interaction Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ppi.cli demo                    # Run basic demo
  python -m ppi.cli ensemble-demo          # Run ensemble demo
  python -m ppi.cli single-protein-demo    # Run single protein demo
        """
    )
    
    parser.add_argument(
        "command",
        choices=["demo", "ensemble-demo", "single-protein-demo"],
        help="Command to run"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed (default: {SEED})"
    )
    
    args = parser.parse_args()
    
    # Set seeds
    set_seeds(args.seed)
    
    # Run command
    try:
        if args.command == "demo":
            cmd_demo()
        elif args.command == "ensemble-demo":
            cmd_ensemble_demo()
        elif args.command == "single-protein-demo":
            cmd_single_protein_demo()
    except KeyboardInterrupt:
        print("\n\n❌ Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
