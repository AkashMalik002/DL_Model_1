"""
Visualization functions for training progress and metrics comparison
"""

import numpy as np
import matplotlib.pyplot as plt


def visualize_training_loss(all_results):
    """
    Create comprehensive loss visualization for all domains
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    axes = axes.flatten()

    for idx, (domain_key, results) in enumerate(all_results.items()):
        ax = axes[idx]
        
        # Extract loss histories
        baseline_loss = results['models']['baseline']['model'].loss_history
        mlp_loss = results['models']['mlp']['model'].loss_history
        
        # Plot
        epochs_range = range(1, len(baseline_loss) + 1)
        ax.plot(epochs_range, baseline_loss, 'b-', linewidth=2, label='Baseline (Logistic Regression)', alpha=0.7)
        ax.plot(epochs_range, mlp_loss, 'r-', linewidth=2, label='MLP (Hidden: [64, 32])', alpha=0.7)
        
        # Formatting
        ax.set_xlabel('Epoch', fontsize=10, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=10, fontweight='bold')
        ax.set_title(f"{results['name'].split(':')[1].strip()}\nTraining Loss Convergence", 
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add convergence annotation
        final_baseline_loss = baseline_loss[-1]
        final_mlp_loss = mlp_loss[-1]
        ax.text(0.98, 0.97, f'Final Loss\nBaseline: {final_baseline_loss:.4f}\nMLP: {final_mlp_loss:.4f}',
                transform=ax.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Remove extra subplot
    axes[-1].remove()

    plt.suptitle('Training Loss Convergence Across All Domains', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()

    print("✓ Training loss visualization completed")


def visualize_metrics_comparison(all_results):
    """
    Create comprehensive metrics comparison across domains
    """
    # Prepare data for visualization
    domains_names = []
    baseline_acc = []
    mlp_acc = []
    baseline_prec = []
    mlp_prec = []
    baseline_recall = []
    mlp_recall = []
    baseline_f1 = []
    mlp_f1 = []

    for domain_key, results in all_results.items():
        domains_names.append(results['name'].split(':')[1].strip())
        
        baseline_metrics = results['models']['baseline']['metrics']
        mlp_metrics = results['models']['mlp']['metrics']
        
        baseline_acc.append(baseline_metrics['accuracy'])
        mlp_acc.append(mlp_metrics['accuracy'])
        baseline_prec.append(baseline_metrics['precision'])
        mlp_prec.append(mlp_metrics['precision'])
        baseline_recall.append(baseline_metrics['recall'])
        mlp_recall.append(mlp_metrics['recall'])
        baseline_f1.append(baseline_metrics['f1_score'])
        mlp_f1.append(mlp_metrics['f1_score'])

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    metrics_data = [
        ('Accuracy', baseline_acc, mlp_acc),
        ('Precision', baseline_prec, mlp_prec),
        ('Recall', baseline_recall, mlp_recall),
        ('F1-Score', baseline_f1, mlp_f1)
    ]

    for idx, (metric_name, baseline_vals, mlp_vals) in enumerate(metrics_data):
        ax = axes[idx // 2, idx % 2]
        x = np.arange(len(domains_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline', alpha=0.8, color='skyblue')
        bars2 = ax.bar(x + width/2, mlp_vals, width, label='MLP', alpha=0.8, color='salmon')
        
        ax.set_xlabel('Domain', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_name} Comparison Across Domains', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([name[:15] for name in domains_names], rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.0])
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle('Model Performance Comparison: Baseline vs MLP', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print("✓ Performance metrics visualization completed")


def print_detailed_metrics(all_results):
    """Print detailed metrics summary tables for each domain"""
    import pandas as pd
    
    for domain_key, results in all_results.items():
        print("\n" + "-"*80)
        print(f"DETAILED METRICS: {results['name']}")
        print("-"*80)
        
        # Create summary dataframe
        baseline_metrics = results['models']['baseline']['metrics']
        mlp_metrics = results['models']['mlp']['metrics']
        
        summary_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Training Time (s)'],
            'Baseline': [
                f"{baseline_metrics['accuracy']:.4f}",
                f"{baseline_metrics['precision']:.4f}",
                f"{baseline_metrics['recall']:.4f}",
                f"{baseline_metrics['f1_score']:.4f}",
                f"{baseline_metrics['train_time']:.4f}"
            ],
            'MLP': [
                f"{mlp_metrics['accuracy']:.4f}",
                f"{mlp_metrics['precision']:.4f}",
                f"{mlp_metrics['recall']:.4f}",
                f"{mlp_metrics['f1_score']:.4f}",
                f"{mlp_metrics['train_time']:.4f}"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Improvement analysis
        print(f"\nImprovement Analysis:")
        print(f"  Accuracy Improvement: {(mlp_metrics['accuracy'] - baseline_metrics['accuracy'])*100:+.2f}%")
        print(f"  Precision Improvement: {(mlp_metrics['precision'] - baseline_metrics['precision'])*100:+.2f}%")
        print(f"  Recall Improvement: {(mlp_metrics['recall'] - baseline_metrics['recall'])*100:+.2f}%")
        print(f"  F1-Score Improvement: {(mlp_metrics['f1_score'] - baseline_metrics['f1_score'])*100:+.2f}%")
        print(f"  Time Overhead: {(mlp_metrics['train_time']/baseline_metrics['train_time']):.2f}x")


def print_comprehensive_metrics(all_results):
    """Print comprehensive performance metrics comparison"""
    print("\n" + "="*100)
    print("COMPREHENSIVE PERFORMANCE METRICS COMPARISON")
    print("="*100)
    
    print_detailed_metrics(all_results)
