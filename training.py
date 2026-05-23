"""
Model training and evaluation across all domains
"""

import time
import numpy as np
from models import LogisticRegression, MultiLayerPerceptron
from preprocessing import preprocess_data
from metrics import EvaluationMetrics


def train_domain_models(domain_key, dataset_info, learning_rate=0.01, epochs=150):
    """
    Train baseline model and MLP for a domain
    """
    print("\n" + "="*80)
    print(f"TRAINING DOMAIN: {dataset_info['name']}")
    print("="*80)
    
    df = dataset_info['data']
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Initialize results dictionary for this domain
    domain_results = {
        'name': dataset_info['name'],
        'train_size': X_train.shape[0],
        'test_size': X_test.shape[0],
        'num_features': X_train.shape[1],
        'models': {}
    }
    
    # ==================== BASELINE MODEL (Logistic Regression) ====================
    print("\n" + "-"*80)
    print("BASELINE MODEL: Logistic Regression")
    print("-"*80)
    
    start_time = time.time()
    
    baseline_model = LogisticRegression(learning_rate=learning_rate, epochs=epochs)
    baseline_model.fit(X_train, y_train)
    
    baseline_train_time = time.time() - start_time
    
    # Predictions
    y_pred_baseline = baseline_model.predict(X_test)
    
    # Metrics
    baseline_metrics = {
        'accuracy': EvaluationMetrics.accuracy(y_test, y_pred_baseline),
        'precision': EvaluationMetrics.precision(y_test, y_pred_baseline),
        'recall': EvaluationMetrics.recall(y_test, y_pred_baseline),
        'f1_score': EvaluationMetrics.f1_score(y_test, y_pred_baseline),
        'train_time': baseline_train_time
    }
    
    print(f"\nBaseline Model Results:")
    print(f"  Accuracy:  {baseline_metrics['accuracy']:.4f}")
    print(f"  Precision: {baseline_metrics['precision']:.4f}")
    print(f"  Recall:    {baseline_metrics['recall']:.4f}")
    print(f"  F1-Score:  {baseline_metrics['f1_score']:.4f}")
    print(f"  Training Time: {baseline_train_time:.4f}s")
    
    domain_results['models']['baseline'] = {
        'model': baseline_model,
        'metrics': baseline_metrics,
        'predictions': y_pred_baseline,
        'y_test': y_test
    }
    
    # ==================== MLP MODEL ====================
    print("\n" + "-"*80)
    print("ADVANCED MODEL: Multi-Layer Perceptron (MLP)")
    print("-"*80)
    
    start_time = time.time()
    
    mlp_model = MultiLayerPerceptron(
        hidden_layers=[64, 32],
        learning_rate=learning_rate,
        epochs=epochs,
        task='binary',
        num_classes=2
    )
    mlp_model.fit(X_train, y_train)
    
    mlp_train_time = time.time() - start_time
    
    # Predictions
    y_pred_mlp = mlp_model.predict(X_test)
    
    # Metrics
    mlp_metrics = {
        'accuracy': EvaluationMetrics.accuracy(y_test, y_pred_mlp),
        'precision': EvaluationMetrics.precision(y_test, y_pred_mlp),
        'recall': EvaluationMetrics.recall(y_test, y_pred_mlp),
        'f1_score': EvaluationMetrics.f1_score(y_test, y_pred_mlp),
        'train_time': mlp_train_time
    }
    
    print(f"\nMLP Model Results:")
    print(f"  Accuracy:  {mlp_metrics['accuracy']:.4f}")
    print(f"  Precision: {mlp_metrics['precision']:.4f}")
    print(f"  Recall:    {mlp_metrics['recall']:.4f}")
    print(f"  F1-Score:  {mlp_metrics['f1_score']:.4f}")
    print(f"  Training Time: {mlp_train_time:.4f}s")
    
    domain_results['models']['mlp'] = {
        'model': mlp_model,
        'metrics': mlp_metrics,
        'predictions': y_pred_mlp,
        'y_test': y_test
    }
    
    # ==================== COMPARATIVE ANALYSIS ====================
    print("\n" + "-"*80)
    print("COMPARATIVE ANALYSIS")
    print("-"*80)
    
    acc_improvement = (mlp_metrics['accuracy'] - baseline_metrics['accuracy']) * 100
    time_overhead = (mlp_train_time - baseline_train_time) / baseline_train_time * 100
    
    print(f"\nAccuracy Improvement (MLP vs Baseline): {acc_improvement:+.2f}%")
    print(f"Training Time Overhead (MLP): {time_overhead:+.2f}%")
    
    if acc_improvement > 0 and time_overhead < 200:
        print("✓ MLP provides better accuracy with reasonable computational cost")
    elif acc_improvement > 0 and time_overhead >= 200:
        print("⚠ MLP improves accuracy but requires significant computational resources")
    else:
        print("⚠ Baseline model is more efficient for this domain")
    
    domain_results['comparative_analysis'] = {
        'accuracy_improvement': acc_improvement,
        'time_overhead': time_overhead
    }
    
    print("\n" + "="*80)
    
    return domain_results


def train_all_domains(datasets, learning_rate=0.01, epochs=150):
    """Train models for all domains and return results"""
    print("\n\nTRAINING ALL DOMAINS...\n")
    
    all_results = {}
    for domain_key, dataset_info in datasets.items():
        domain_results = train_domain_models(domain_key, dataset_info, learning_rate, epochs)
        all_results[domain_key] = domain_results
    
    print("\n✓ All domains trained successfully!")
    return all_results
