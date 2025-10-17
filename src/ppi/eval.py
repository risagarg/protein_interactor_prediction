"""
Evaluation utilities for PPI prediction.

This module provides comprehensive evaluation metrics and visualization
for the protein interaction prediction pipeline.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    roc_curve, confusion_matrix, classification_report, f1_score
)
from typing import Dict, Tuple, Optional


def eval_metrics(y_true: np.ndarray, y_prob: np.ndarray, 
                outdir: str = "artifacts", model_name: str = "model") -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        outdir: Output directory for saving results
        model_name: Name of the model for file naming
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    os.makedirs(outdir, exist_ok=True)
    
    # Calculate metrics
    metrics = {
        "AUPRC": float(average_precision_score(y_true, y_prob)),
        "AUROC": float(roc_auc_score(y_true, y_prob)),
        "F1": float(f1_score(y_true, (y_prob > 0.5).astype(int))),
        "n_samples": len(y_true),
        "n_positive": int(np.sum(y_true)),
        "n_negative": int(np.sum(y_true == 0)),
        "pos_rate": float(np.mean(y_true))
    }
    
    # Save metrics
    with open(f"{outdir}/{model_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def plot_precision_recall(y_true: np.ndarray, y_prob: np.ndarray, 
                         outdir: str = "artifacts", model_name: str = "model"):
    """
    Plot precision-recall curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        outdir: Output directory
        model_name: Name for file naming
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, 'b-', linewidth=2, label=f'AUPRC = {auprc:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/{model_name}_precision_recall.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, 
                  outdir: str = "artifacts", model_name: str = "model"):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        outdir: Output directory
        model_name: Name for file naming
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUROC = {auroc:.3f}')
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/{model_name}_roc.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         outdir: str = "artifacts", model_name: str = "model"):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        outdir: Output directory
        model_name: Name for file naming
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Interaction', 'Interaction'],
                yticklabels=['No Interaction', 'Interaction'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/{model_name}_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance(feature_names: list, importances: np.ndarray, 
                           outdir: str = "artifacts", model_name: str = "model", 
                           top_n: int = 20):
    """
    Plot feature importance.
    
    Args:
        feature_names: List of feature names
        importances: Feature importance values
        outdir: Output directory
        model_name: Name for file naming
        top_n: Number of top features to show
    """
    # Sort features by importance
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importances[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Feature Importances - {model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/{model_name}_feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()


def comprehensive_eval(y_true: np.ndarray, y_prob: np.ndarray, 
                      feature_names: Optional[list] = None,
                      importances: Optional[np.ndarray] = None,
                      outdir: str = "artifacts", model_name: str = "model"):
    """
    Run comprehensive evaluation with all plots and metrics.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        feature_names: List of feature names
        importances: Feature importance values
        outdir: Output directory
        model_name: Name for file naming
    """
    print(f"Running comprehensive evaluation for {model_name}...")
    
    # Calculate metrics
    metrics = eval_metrics(y_true, y_prob, outdir, model_name)
    print(f"Metrics: {metrics}")
    
    # Generate plots
    plot_precision_recall(y_true, y_prob, outdir, model_name)
    plot_roc_curve(y_true, y_prob, outdir, model_name)
    
    # Confusion matrix
    y_pred = (y_prob > 0.5).astype(int)
    plot_confusion_matrix(y_true, y_pred, outdir, model_name)
    
    # Feature importance (if available)
    if feature_names is not None and importances is not None:
        plot_feature_importance(feature_names, importances, outdir, model_name)
    
    print(f"Evaluation complete. Results saved to {outdir}/")
    
    return metrics
