"""
Protein-Protein Interaction Prediction Package

A comprehensive pipeline for predicting novel protein interactors using ensemble learning.
"""

__version__ = "0.1.0"
__author__ = "Risa Garg"
__email__ = "risagarg@sas.upenn.edu"

from .demo_data import generate_demo
from .features import build_pair_features
from .models import build_pipeline
from .eval import eval_metrics

__all__ = [
    "generate_demo",
    "build_pair_features", 
    "build_pipeline",
    "eval_metrics",
]
