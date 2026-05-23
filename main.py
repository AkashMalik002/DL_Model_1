"""
Main Orchestrator for Deep Learning Assignment
Modular Project Structure for Multi-Domain Model Training and Analysis

Project Flow:
1. Section 0: Imports & Setup
2. Section 1: Dataset Preparation and Loading (from datasets.py)
3. Section 2: Model Definitions (from models.py)
4. Section 3: Evaluation Metrics (from metrics.py)
5. Section 4: Preprocessing Pipeline (from preprocessing.py)
6. Section 5: Multi-Domain Training (from training.py)
7. Section 6: Loss Visualization (from visualization.py)
8. Section 7: Metrics Comparison (from visualization.py)
9. Section 8: Situated Learning Analysis (from analysis.py)
10. Section 9: Final Results Summary (from results.py)
"""

# ==================== SECTION 0: IMPORTS & SETUP ====================
print("\n" + "="*100)
print("SECTION 0: IMPORTS AND SETUP")
print("="*100 + "\n")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Import custom modules
from datasets import create_domain_datasets, display_dataset_summary
from models import LogisticRegression, LinearRegression, SoftmaxRegression, MultiLayerPerceptron
from metrics import EvaluationMetrics
from preprocessing import preprocess_data
from training import train_all_domains
from visualization import visualize_training_loss, visualize_metrics_comparison, print_comprehensive_metrics
from analysis import print_situated_learning_analysis, get_domain_recommendations
from results import get_assignment_results, display_assignment_results, print_export_guide

print("✓ All required libraries imported successfully!")
print("✓ Libraries in use: NumPy, Pandas, Matplotlib, Seaborn")
print("✓ sklearn utilities: train_test_split, StandardScaler, MinMaxScaler")
print("\n✓ All custom modules imported successfully!")


# ==================== SECTION 1: DATASET PREPARATION AND LOADING ====================
print("\n" + "="*100)
print("SECTION 1: DATASET PREPARATION AND LOADING")
print("="*100 + "\n")

print("## Creating datasets for all 5 domains...")
print("## Each dataset: ≥500 samples, ≥10 features, binary classification\n")

# Create all datasets
datasets = create_domain_datasets()

# Display dataset information
display_dataset_summary(datasets)


# ==================== SECTION 2-4: MODELS, METRICS, PREPROCESSING ====================
print("\n" + "="*100)
print("SECTIONS 2-4: ML MODELS, METRICS & PREPROCESSING")
print("="*100)
print("\n✓ Logistic Regression class loaded from models.py")
print("✓ Linear Regression class loaded from models.py")
print("✓ Softmax Regression class loaded from models.py")
print("✓ MultiLayerPerceptron class loaded from models.py")
print("✓ EvaluationMetrics class loaded from metrics.py")
print("✓ Preprocessing pipeline loaded from preprocessing.py")


# ==================== SECTION 5: MULTI-DOMAIN MODEL TRAINING ====================
print("\n" + "="*100)
print("SECTION 5: MULTI-DOMAIN MODEL TRAINING AND EVALUATION")
print("="*100)

# Train models for all domains
all_results = train_all_domains(datasets, learning_rate=0.01, epochs=150)


# ==================== SECTION 6: TRAINING LOSS VISUALIZATION ====================
print("\n" + "="*100)
print("SECTION 6: TRAINING LOSS VISUALIZATION AND CONVERGENCE ANALYSIS")
print("="*100 + "\n")

visualize_training_loss(all_results)


# ==================== SECTION 7: PERFORMANCE METRICS COMPARISON ====================
print("\n" + "="*100)
print("SECTION 7: PERFORMANCE METRICS COMPARISON AND ANALYSIS")
print("="*100 + "\n")

print_comprehensive_metrics(all_results)
visualize_metrics_comparison(all_results)


# ==================== SECTION 8: SITUATED LEARNING ANALYSIS ====================
print("\n" + "="*100)
print("SECTION 8: SITUATED LEARNING ANALYSIS: DOMAIN-SPECIFIC DISCUSSIONS")
print("="*100)

print_situated_learning_analysis()


# ==================== SECTION 9: FINAL RESULTS SUMMARY ====================
print("\n" + "="*100)
print("SECTION 9: FINAL RESULTS SUMMARY")
print("="*100)

assignment_results = display_assignment_results(all_results)


# ==================== SECTION 10: EXPORT INSTRUCTIONS ====================
print("\n" + "="*100)
print("SECTION 10: EXPORT INSTRUCTIONS AND SUBMISSION GUIDE")
print("="*100 + "\n")

print_export_guide()

print("\n✓✓✓ ASSIGNMENT COMPLETE ✓✓✓")
print("\nProject Structure:")
print("  main.py              - This main orchestrator")
print("  models.py            - ML models (LogisticRegression, LinearRegression, Softmax, MLP)")
print("  metrics.py           - Evaluation metrics")
print("  preprocessing.py     - Data preprocessing pipeline")
print("  datasets.py          - Dataset creation")
print("  training.py          - Model training for all domains")
print("  visualization.py     - Loss and metrics visualization")
print("  analysis.py          - Situated learning analysis")
print("  results.py           - Final results summary")
print("\nTo run again: python main.py")
print("\n" + "="*100)
